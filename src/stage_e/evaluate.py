"""CASAS Stage E: per-home models, one protocol for all.

For every target home and model (and seed): fit on the home's train days; per-channel error scale from its val days;
the score aggregation (max / mean / top3, smoothing) chosen on anomalies injected into VAL; then on held-out data:
  - forecast error (test), skill vs persistence;
  - injected anomalies in TEST (AUROC per type);
  - cross-user deviation: day score (mean window score) of the home's own TEST days vs other homes' test days mapped
    onto its channels ("impostor" days); AUROC per home, and the share of impostor/own days above the 95th
    percentile of the home's own VAL day scores.
Results are written per (home, model, seed) as JSON, so an interrupted run resumes where it stopped.
"""
import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import data, inject, models, train

log = logging.getLogger("ca")


def trainable(name):
    return name in ("lstm", "lstm_ae") or name.startswith("gdn")


def _auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else np.nan


def injected(ds, cfg, split, offset):
    e = cfg["stage_e"]["inject"]
    out = {}
    for k, typ in enumerate(e["types"]):
        X, obs, segs, _ = inject.inject(ds, typ, cfg, e["seed"] + offset + k, split)
        if segs:
            Z = data.standardize(ds, X, obs)
            a = data.make_arrays(Z, obs, ds["cov"], ds["widx"][split], ds["w"], ds["h"])
            a["Zsrc"] = Z
            out[typ] = (a, segs)
    return out


def day_scores(s, days):
    d = pd.Series(np.where(np.isfinite(s), s, np.nan), index=pd.DatetimeIndex(days)).groupby(level=0).mean()
    return d.dropna()


def neighbours_same_room(A, ds):
    room = np.asarray(ds["room"])
    same = room[:, None] == room[None, :]
    np.fill_diagonal(same, False)
    N = len(room)
    return float((A & same).sum() / max(A.sum(), 1)), float(same.sum() / max(N * (N - 1), 1))


def run_home(target, others, cfg, out_dir):
    e = cfg["stage_e"]
    w, h = target["w"], target["h"]
    home = target["home"]
    hdir = Path(out_dir) / "runs" / home
    hdir.mkdir(parents=True, exist_ok=True)
    arr = {s: data.make_arrays(target["Z"], target["obs"], target["cov"], target["widx"][s], w, h) for s in ("train", "val", "test")}
    if min(len(arr[s]["y"]) for s in arr) == 0:
        log.warning("%s: an empty split, skipped", home)
        return []
    inj_val, inj_test = injected(target, cfg, "val", 1000), injected(target, cfg, "test", 0)
    rng = np.random.default_rng(e["impostor"]["seed"])
    imps, cover = {}, {}
    for o in others:
        a, share = data.impostor_arrays(target, o, cfg, rng)
        cover[o["home"]] = share
        if a is not None:
            imps[o["home"]] = a
    combos = [(g, int(k)) for g in e["score"]["channel_agg"] for k in e["score"]["smooth_steps"]]
    rows = []
    graphs = {}
    for name in e["models"]:
        seeds = (e["seeds"] if name.startswith("gdn") else e["seeds_other_trainable"]) if trainable(name) else [0]
        for seed in seeds:
            f = hdir / f"{name}_s{seed}.json"
            if f.exists() and not e["force"]:
                rows.append(json.loads(f.read_text()))
                continue
            m = models.build(name, target, cfg, seed)
            info = train.fit(m, arr["train"], arr["val"], cfg, seed)
            r = dict(home=home, model=name, seed=seed, n_channels=len(target["names"]), n_params=int(info["n_params"]),
                     epochs=int(info["epochs"]), train_s=round(float(info["train_s"]), 1),
                     n_train_days=target["split_info"]["train"][2], n_test_days=target["split_info"]["test"][2])
            E = {}
            for s in ("val", "test"):
                E[s], sq, cnt = train.errors(m, arr[s], target)
                r[f"{s}_mse"] = sq / max(cnt, 1)
            med = np.nan_to_num(np.nanmedian(E["val"], 0), nan=0.0)
            iqr = np.nanpercentile(E["val"], 75, 0) - np.nanpercentile(E["val"], 25, 0)
            spread = (np.nanpercentile(E["val"], 95, 0) - np.nanpercentile(E["val"], 5, 0)) / 4
            iqr = np.maximum(np.nan_to_num(np.maximum(iqr, spread), nan=1.0), e["score_scale_floor"])
            Zv = train.score(E["val"], med, iqr)[1]
            Zt = train.score(E["test"], med, iqr)[1]
            Zi = {(tag, typ): train.score(train.errors(m, a, target)[0], med, iqr)[1]
                  for tag, inj in (("val", inj_val), ("test", inj_test)) for typ, (a, _) in inj.items()}
            Zimp = {oh: train.score(train.errors(m, a, target)[0], med, iqr)[1] for oh, a in imps.items()}

            def S(Zm, t, g, k):
                return train.smooth(train.aggregate(Zm, g), t, k)

            def inj_auc(inj, tag, g, k):
                v = []
                for typ, (a, segs) in inj.items():
                    lab = inject.labels_for(a["t"], segs, h + w - 1 + k - 1)
                    keep = lab >= 0
                    v.append(_auc(lab[keep] == 1, S(Zi[(tag, typ)], a["t"], g, k)[keep]))
                return v

            best = None
            for g, k in combos:
                v = np.nanmean(inj_auc(inj_val, "val", g, k)) if inj_val else np.nan
                key = v if e["score"]["select"] == "val_injected" else float([g, k] == list(e["score"]["default"]))
                if best is None or np.nan_to_num(key, nan=-1) > best[0] + 1e-9:
                    best = (np.nan_to_num(key, nan=-1), g, k)
            _, g, k = best
            r["score_agg"], r["score_smooth"] = g, k
            for typ, v in zip(inj_test, inj_auc(inj_test, "test", g, k)):
                r[f"inj_{typ}_auroc"] = v
            r["inj_mean_auroc"] = float(np.nanmean([r[f"inj_{t}_auroc"] for t in inj_test])) if inj_test else np.nan
            # cross-user: day scores
            dv = day_scores(S(Zv, arr["val"]["t"], g, k), target["times"][arr["val"]["t"]].normalize())
            dt = day_scores(S(Zt, arr["test"]["t"], g, k), target["times"][arr["test"]["t"]].normalize())
            di = [day_scores(S(Zimp[oh], a["t"], g, k), a["day"]) for oh, a in imps.items()]
            di = pd.concat(di) if di else pd.Series(dtype=float)
            r["impostor_homes"], r["impostor_days"] = len(imps), int(len(di))
            r["impostor_matched_share_mean"] = float(np.mean([cover[o] for o in imps])) if imps else np.nan
            r["cross_user_day_auroc"] = _auc(np.r_[np.zeros(len(dt)), np.ones(len(di))], np.r_[dt.values, di.values])
            thr = float(np.quantile(dv.values, e["day_threshold_quantile"])) if len(dv) else np.nan
            r["own_test_day_flag_rate"] = float((dt.values > thr).mean()) if len(dt) else np.nan
            r["impostor_day_flag_rate"] = float((di.values > thr).mean()) if len(di) else np.nan
            per_imp = {oh: _auc(np.r_[np.zeros(len(dt)), np.ones(len(d))], np.r_[dt.values, d.values])
                       for oh, d in zip(imps, [day_scores(S(Zimp[oh], a["t"], g, k), a["day"]) for oh, a in imps.items()])}
            r["cross_user_auroc_by_impostor"] = per_imp
            if isinstance(m, models.GDN) and m.graph == "learned":
                A = m.neighbours()
                r["graph_same_room_share"], r["graph_same_room_chance"] = neighbours_same_room(A, target)
                r["graph_edges"] = A.astype(int).tolist()
            f.write_text(json.dumps(r, default=float))
            rows.append(r)
            log.info("  %-7s %-16s s%d mse val/test %.3f/%.3f inj %.3f cross-user %.3f (%d impostor homes) %s/k%d %.0fs",
                     home, name, seed, r["val_mse"], r["test_mse"], r["inj_mean_auroc"], r["cross_user_day_auroc"],
                     len(imps), g, k, r["train_s"])
    return rows


def seed_jaccard(rows):
    """Overlap of learned GDN edges between seeds of the same home and model."""
    out = []
    df = pd.DataFrame([r for r in rows if "graph_edges" in r])
    for (h, mname), d in df.groupby(["home", "model"]):
        A = [np.array(x, bool) for x in d.graph_edges]
        j = [float((a & b).sum() / max((a | b).sum(), 1)) for a, b in combinations(A, 2)]
        out.append(dict(home=h, model=mname, seed_jaccard=float(np.mean(j)) if j else np.nan))
    return pd.DataFrame(out)


PAIRS = [("gdn", "gdn_nograph"), ("gdn_hier", "gdn"), ("gdn_hier", "gdn_hier_random"),
         ("gdn_hier_group", "gdn"), ("gdn_hier_group", "gdn_hier_group_random"), ("gdn_hier", "lstm"), ("gdn", "var")]
METRICS = ("cross_user_day_auroc", "inj_mean_auroc", "inj_room_swap_auroc", "test_mse")


def paired(R):
    """Home-level paired comparison (seeds averaged within a home first): mean diff, bootstrap 95% CI over homes,
    homes where the first model is better, Wilcoxon signed-rank p."""
    from scipy.stats import wilcoxon
    out = []
    H = R.groupby(["home", "model"])[[m for m in METRICS if m in R]].mean()
    rng = np.random.default_rng(0)
    for a, b in PAIRS:
        if a not in H.index.get_level_values(1) or b not in H.index.get_level_values(1):
            continue
        A, B = H.xs(a, level=1), H.xs(b, level=1)
        common = A.index.intersection(B.index)
        row = dict(comparison=f"{a} - {b}", homes=len(common))
        for mt in METRICS:
            if mt not in A:
                continue
            d = (A.loc[common, mt] - B.loc[common, mt]).dropna().values
            if not len(d):
                continue
            bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
            better = (d < 0) if mt.endswith("mse") else (d > 0)
            try:
                p = float(wilcoxon(d).pvalue) if len(d) >= 6 and np.any(d != 0) else np.nan
            except ValueError:
                p = np.nan
            row.update({f"{mt}_diff": float(d.mean()), f"{mt}_ci_lo": float(np.quantile(bs, .025)),
                        f"{mt}_ci_hi": float(np.quantile(bs, .975)), f"{mt}_better": f"{int(better.sum())}/{len(d)}", f"{mt}_p": p})
        out.append(row)
    return pd.DataFrame(out)
