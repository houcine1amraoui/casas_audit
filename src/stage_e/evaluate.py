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
    return name in ("lstm", "lstm_ae") or name.startswith("gdn") or name.startswith("hae")


def calib_stats(s, hours, bin_h):
    """Per time-of-day bin: median and robust scale (IQR, floored) of clean VAL window scores."""
    b = hours // bin_h
    st = {}
    for k in np.unique(b):
        v = s[(b == k) & np.isfinite(s)]
        if len(v) >= 5:
            q1, q2, q3 = np.quantile(v, [.25, .5, .75])
            st[int(k)] = (q2, max(q3 - q1, 1e-3))
    return st


def calibrate(s, hours, st, bin_h):
    """Score relative to what is usual for this time of day (val), so that night-like activity in the day or
    day-like activity at night stands out. Bins without val statistics keep the raw score shifted by the global median."""
    if not st:
        return s
    b = hours // bin_h
    med = np.array([st.get(int(k), (np.nan, np.nan))[0] for k in b])
    sc = np.array([st.get(int(k), (np.nan, np.nan))[1] for k in b])
    gm, gs = np.median([v[0] for v in st.values()]), np.median([v[1] for v in st.values()])
    med, sc = np.where(np.isnan(med), gm, med), np.where(np.isnan(sc), gs, sc)
    return (s - med) / sc


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
    combos = [(g, int(k), cal) for g in e["score"]["channel_agg"] for k in e["score"]["smooth_steps"] for cal in e["score"]["calibration"]]
    bin_h = int(e["score"]["calibration_bin_hours"])
    hrs = {s: target["times"][arr[s]["t"]].hour.values for s in ("val", "test")}
    rows = []
    graphs = {}
    for name in e["models"]:
        seeds = (e["seeds"] if name.startswith(("gdn", "hae")) else e["seeds_other_trainable"]) if trainable(name) else [0]
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

            cstats = {}

            def S(Zm, t, g, k, cal="none", hours=None):
                s = train.smooth(train.aggregate(Zm, g), t, k)
                if cal == "hour":
                    if (g, k) not in cstats:       # fitted on the clean val windows of this model, aggregation and smoothing
                        cstats[(g, k)] = calib_stats(train.smooth(train.aggregate(Zv, g), arr["val"]["t"], k), hrs["val"], bin_h)
                    s = calibrate(s, hours, cstats[(g, k)], bin_h)
                return s

            def inj_auc(inj, tag, g, k, cal):
                v = []
                for typ, (a, segs) in inj.items():
                    lab = inject.labels_for(a["t"], segs, h + w - 1 + k - 1)
                    keep = lab >= 0
                    v.append(_auc(lab[keep] == 1, S(Zi[(tag, typ)], a["t"], g, k, cal, target["times"][a["t"]].hour.values)[keep]))
                return v

            best = None
            grid = {}
            for g, k, cal in combos:
                v = np.nanmean(inj_auc(inj_val, "val", g, k, cal)) if inj_val else np.nan
                grid[f"{g}/k{k}/{cal}"] = float(v)
                key = v if e["score"]["select"] == "val_injected" else float([g, k, cal] == list(e["score"]["default"]))
                if best is None or np.nan_to_num(key, nan=-1) > best[0] + 1e-9:
                    best = (np.nan_to_num(key, nan=-1), g, k, cal)
            _, g, k, cal = best
            r["score_agg"], r["score_smooth"], r["score_calibration"] = g, k, cal
            r["val_inj_grid"] = grid
            # the same model with the uncalibrated / hour-calibrated score, each with its own best agg+smoothing on val
            for c_ in e["score"]["calibration"]:
                sub = [(kk, vv) for kk, vv in grid.items() if kk.endswith("/" + c_)]
                if sub:
                    bk = max(sub, key=lambda x: np.nan_to_num(x[1], nan=-1))[0].split("/")
                    gg, kk_ = bk[0], int(bk[1][1:])
                    r[f"inj_mean_auroc_{c_}"] = float(np.nanmean(inj_auc(inj_test, "test", gg, kk_, c_))) if inj_test else np.nan
            for typ, v in zip(inj_test, inj_auc(inj_test, "test", g, k, cal)):
                r[f"inj_{typ}_auroc"] = v
            r["inj_mean_auroc"] = float(np.nanmean([r[f"inj_{t}_auroc"] for t in inj_test])) if inj_test else np.nan
            # cross-user: day scores
            dv = day_scores(S(Zv, arr["val"]["t"], g, k, cal, hrs["val"]), target["times"][arr["val"]["t"]].normalize())
            dt = day_scores(S(Zt, arr["test"]["t"], g, k, cal, hrs["test"]), target["times"][arr["test"]["t"]].normalize())
            di = [day_scores(S(Zimp[oh], a["t"], g, k, cal, a["hour"]), a["day"]) for oh, a in imps.items()]
            di = pd.concat(di) if di else pd.Series(dtype=float)
            r["impostor_homes"], r["impostor_days"] = len(imps), int(len(di))
            r["impostor_matched_share_mean"] = float(np.mean([cover[o] for o in imps])) if imps else np.nan
            r["cross_user_day_auroc"] = _auc(np.r_[np.zeros(len(dt)), np.ones(len(di))], np.r_[dt.values, di.values])
            thr = float(np.quantile(dv.values, e["day_threshold_quantile"])) if len(dv) else np.nan
            r["own_test_day_flag_rate"] = float((dt.values > thr).mean()) if len(dt) else np.nan
            r["impostor_day_flag_rate"] = float((di.values > thr).mean()) if len(di) else np.nan
            per_imp = {oh: _auc(np.r_[np.zeros(len(dt)), np.ones(len(d))], np.r_[dt.values, d.values])
                       for oh, d in zip(imps, [day_scores(S(Zimp[oh], a["t"], g, k, cal, a["hour"]), a["day"]) for oh, a in imps.items()])}
            r["cross_user_auroc_by_impostor"] = per_imp
            if isinstance(m, models.GDN) and m.graph == "learned":
                A = m.neighbours()
                r["graph_same_room_share"], r["graph_same_room_chance"] = neighbours_same_room(A, target)
                r["graph_edges"] = A.astype(int).tolist()
            f.write_text(json.dumps(r, default=float))
            rows.append(r)
            log.info("  %-7s %-16s s%d mse val/test %.3f/%.3f inj %.3f (none %.3f, hour %.3f) cross-user %.3f (%d homes) %s/k%d/%s %.0fs",
                     home, name, seed, r["val_mse"], r["test_mse"], r["inj_mean_auroc"], r.get("inj_mean_auroc_none", np.nan),
                     r.get("inj_mean_auroc_hour", np.nan), r["cross_user_day_auroc"], len(imps), g, k, cal, r["train_s"])
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


PAIRS = [("hae", "hae_random"), ("hae", "hae_flat"), ("hae", "lstm_ae"),
         ("gdn", "gdn_nograph"), ("gdn_hier", "gdn"), ("gdn_hier", "gdn_hier_random"),
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
