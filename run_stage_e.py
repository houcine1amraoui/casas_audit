"""CASAS Stage E: per-home models with a cross-user deviation evaluation.

python -m run_stage_e --project_root_dir ROOT [--set stage_e.homes=[tm008,tm010]] [--quick]
Needs cache/casas/steps/<home>.npz and outputs/casas_audit/home_summary.csv from run_audit (v6+).
Resumable: finished (home, model, seed) results in outputs/casas_stage_e/runs/ are reused; the report always uses
everything that is there.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config
from src.stage_e import data, evaluate, train

log = logging.getLogger("ca")
ORDER = ["persistence", "seasonal_naive", "var", "lstm", "lstm_ae", "hae", "hae_random", "hae_flat", "gdn_nograph", "gdn", "gdn_hier", "gdn_hier_random",
         "gdn_hier_group", "gdn_hier_group_random"]


def md(df, fmt=".3f"):
    return df.to_markdown(index=False, floatfmt=fmt) + "\n" if df is not None and len(df) else "_none_\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root_dir", required=True)
    ap.add_argument("--dataset_folder", default="")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--quick", action="store_true", help="smoke test: 3 homes, 1 seed, 3 epochs (results go to a separate run)")
    ap.add_argument("--report_only", action="store_true", help="rebuild the report from the saved results")
    a = ap.parse_args(argv)
    sets = list(a.set)
    if a.quick:
        sets = ["stage_e.seeds=[0]", "stage_e.train.max_epochs=3", "stage_e.inject.n_per_type=4", "stage_e.run_name=casas_stage_e_quick"] + sets
    cfg = load_config(a.project_root_dir, a.dataset_folder, a.config, sets)
    e = cfg["stage_e"]
    out = Path(a.project_root_dir) / "outputs" / e["run_name"]
    (out / "figures").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out / "run.log", mode="a")])
    log.info("device: %s", train.DEV)
    HS = pd.read_csv(Path(a.project_root_dir) / "outputs" / cfg["audit"]["run_name"] / "home_summary.csv")
    step_dir = Path(cfg["paths"]["cache_dir"]) / "steps"
    homes = sorted(HS.home[HS.enough_clean.astype(bool)])
    homes = [h for h in homes if (step_dir / f"{h}.npz").exists()]
    if not homes:
        raise SystemExit(f"no step cache in {step_dir}: rerun run_audit (v6+) first")
    targets = homes if not e["homes"] else [h for h in homes if h in set(e["homes"])]
    if a.quick:
        targets = targets[:3]
    ds = {h: data.build(h, data.load_steps(step_dir / f"{h}.npz"), cfg) for h in homes}
    log.info("%d homes with data; modelling %d", len(ds), len(targets))
    if not a.report_only:
        t0 = time.time()
        for i, h in enumerate(targets):
            log.info("[%d/%d] %s: %d channels, splits %s", i + 1, len(targets), h, len(ds[h]["names"]), ds[h]["split_info"])
            evaluate.run_home(ds[h], [ds[o] for o in homes if o != h], cfg, out)
        log.info("modelling done in %.0f min", (time.time() - t0) / 60)
    rows = [json.loads(f.read_text()) for f in sorted((out / "runs").glob("*/*.json"))]
    if not rows:
        raise SystemExit("no results yet")
    report(rows, ds, cfg, out)


def report(rows, ds, cfg, out):
    e = cfg["stage_e"]
    R = pd.DataFrame(rows)
    R = R[R.model.isin(e["models"])]
    R.drop(columns=[c for c in ("graph_edges", "cross_user_auroc_by_impostor", "val_inj_grid") if c in R]).to_csv(out / "runs.csv", index=False)
    # per-home skill vs persistence
    pers = R[R.model == "persistence"].set_index("home").test_mse
    R["skill_vs_persistence"] = 1 - R.test_mse / R.home.map(pers)
    H = R.groupby(["home", "model"]).mean(numeric_only=True).reset_index()             # seeds averaged within a home
    inj_cols = [c for c in R.columns if c.startswith("inj_") and c.endswith("_auroc")]
    cols = ["test_mse", "skill_vs_persistence", "cross_user_day_auroc", "own_test_day_flag_rate", "impostor_day_flag_rate",
            "inj_mean_auroc_none", "inj_mean_auroc_hour"] + inj_cols
    rng = np.random.default_rng(0)
    summ = []
    for mname, d in H.groupby("model"):
        row = dict(model=mname, homes=d.home.nunique())
        for c in cols:
            if c in d:
                row[c] = d[c].mean()
        v = d.cross_user_day_auroc.dropna().values
        if len(v):
            bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
            row["cross_user_ci_lo"], row["cross_user_ci_hi"] = np.quantile(bs, [.025, .975])
        summ.append(row)
    S = pd.DataFrame(summ)
    S["o"] = S.model.map({m: i for i, m in enumerate(ORDER)})
    S = S.sort_values("o").drop(columns="o")
    S.to_csv(out / "summary.csv", index=False)
    P = evaluate.paired(R)
    P.to_csv(out / "paired.csv", index=False)
    J = evaluate.seed_jaccard(rows)
    G = R[R.graph_same_room_share.notna()].groupby("model")[["graph_same_room_share", "graph_same_room_chance"]].mean().reset_index() \
        if "graph_same_room_share" in R else pd.DataFrame()
    if len(J):
        G = G.merge(J.groupby("model").seed_jaccard.mean().reset_index(), on="model", how="left")
    piv = H.pivot_table(index="home", columns="model", values="cross_user_day_auroc")
    piv = piv[[m for m in ORDER if m in piv.columns]]
    piv.to_csv(out / "cross_user_auroc_by_home.csv")
    if "score_calibration" not in R:
        R["score_calibration"] = "none"
    chosen = R.groupby(["score_agg", "score_smooth", "score_calibration"]).size().reset_index(name="runs")

    # ---- figures
    figs = []
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(S))
    ax.errorbar(S.cross_user_day_auroc, y, xerr=[S.cross_user_day_auroc - S.cross_user_ci_lo, S.cross_user_ci_hi - S.cross_user_day_auroc],
                fmt="o", color="#2b6cb0", ecolor="#999", capsize=3)
    ax.axvline(0.5, color="k", lw=.8, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(S.model); ax.invert_yaxis()
    ax.set_xlabel("cross-user day AUROC (own held-out days vs other homes' days); mean over homes, 95% bootstrap CI")
    ax.set_title("Does a home's model flag other homes' days as unusual?")
    fig.tight_layout(); fig.savefig(out / "figures" / "cross_user_auroc.png", dpi=110); plt.close(fig); figs.append("cross_user_auroc.png")
    fig, ax = plt.subplots(figsize=(1 + 0.7 * piv.shape[1], 0.25 * len(piv) + 1.5))
    im = ax.imshow(piv.values, cmap="viridis", vmin=0.4, vmax=1, aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=6)
    fig.colorbar(im, ax=ax, label="cross-user day AUROC"); ax.set_title("Per home")
    fig.tight_layout(); fig.savefig(out / "figures" / "cross_user_auroc_by_home.png", dpi=110); plt.close(fig); figs.append("cross_user_auroc_by_home.png")
    if len(P):
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(P) + 1.5))
        for i, (_, p) in enumerate(P.iterrows()):
            a_, b_ = p.comparison.split(" - ")
            d = (H[H.model == a_].set_index("home").cross_user_day_auroc - H[H.model == b_].set_index("home").cross_user_day_auroc).dropna()
            ax.scatter(d.values, np.full(len(d), i) + rng.uniform(-.15, .15, len(d)), s=10, color="#888")
            ax.plot([p.cross_user_day_auroc_ci_lo, p.cross_user_day_auroc_ci_hi], [i, i], color="#2b6cb0", lw=3)
            ax.plot(p.cross_user_day_auroc_diff, i, "o", color="#2b6cb0")
        ax.axvline(0, color="k", lw=.8)
        ax.set_yticks(range(len(P))); ax.set_yticklabels(P.comparison, fontsize=8); ax.invert_yaxis()
        ax.set_xlabel("difference in cross-user day AUROC per home (dots), mean and 95% CI (blue)")
        fig.tight_layout(); fig.savefig(out / "figures" / "paired_differences.png", dpi=110); plt.close(fig); figs.append("paired_differences.png")

    # ---- report
    L = []
    w = L.append
    w("# CASAS Stage E: per-home models and cross-user deviation\n")
    w(f"{R.home.nunique()} homes, models {', '.join(m for m in ORDER if m in set(R.model))}; GDN-family seeds {e['seeds']}, "
      f"LSTM seeds {e['seeds_other_trainable']}. 10-min steps, input {e['window_steps']} steps, forecast {e['horizon_steps']} steps ahead. "
      f"Per home: chronological split of clean days {e['split']}; normalisation from train, error scale and score choice from val "
      "(anomalies injected into val), everything else on held-out test days.\n")
    if e["train"]["max_epochs"] < 10 or len(e["seeds"]) < 2:
        w("**Smoke-test settings: not for interpretation.**\n")
    w("**Cross-user deviation**: each home's model scores its own test days (should look normal) and up to "
      f"{e['impostor']['days_per_home']} test days of every other home, mapped onto its sensors by name (room type + spot; "
      f"homes sharing < {e['impostor']['min_matched_share']:.0%} of the sensors are skipped). Unmatched sensors are unobserved, which "
      "makes other homes look more normal, so the AUROC is conservative. Day score = mean window score.\n")
    w(f"Sensor match between homes: mean {R.impostor_matched_share_mean.mean():.0%} of a home's sensors found in an impostor home; "
      f"mean {R.impostor_homes.mean():.1f} impostor homes and {R.impostor_days.mean():.0f} impostor days per home.\n")
    w("## 1. Results by model (mean over homes; seeds averaged within a home)\n")
    w(md(S[["model", "homes", "test_mse", "skill_vs_persistence", "cross_user_day_auroc", "cross_user_ci_lo", "cross_user_ci_hi",
            "own_test_day_flag_rate", "impostor_day_flag_rate"]]))
    w("`lstm_ae` reconstructs its input window, so its MSE and skill are not comparable with the forecasters.\n")
    w(f"Flag rates use a threshold at the {e['day_threshold_quantile']:.0%} quantile of the home's own val day scores "
      "(about 5% of own days expected without drift).\n")
    w("## 2. Injected anomalies (test split, AUROC)\n")
    w(md(S[["model"] + inj_cols]))
    w("Score settings chosen on val-injected anomalies (aggregation/smoothing/calibration: runs): "
      + ", ".join(f"{r.score_agg}/k{r.score_smooth}/{r.score_calibration}: {r.runs}" for r in chosen.itertuples()) + ".\n")
    if "inj_mean_auroc_hour" in S:
        w("Effect of hour-of-day calibration on injected anomalies (test, mean AUROC; each column uses the best "
          "aggregation and smoothing chosen on val for that calibration):\n\n" + md(S[["model", "inj_mean_auroc_none", "inj_mean_auroc_hour"]]))
    w("## 3. Flat vs hierarchical and other paired comparisons (per home, Wilcoxon over homes)\n")
    w("Difference = first model - second model; MSE: negative is better, AUROC: positive is better. `better` = homes where the "
      "first model is better. The hierarchy question is answered by `gdn_hier - gdn` (does the fixed room hierarchy add to the "
      "flat learned graph) and `gdn_hier - gdn_hier_random` (does the RIGHT grouping matter, at equal capacity); for the "
      "autoencoders `hae - hae_random` (right rooms vs random rooms, same sizes) and `hae - hae_flat` (room codes vs one "
      "shared code of the same total size).\n\n")
    show = ["comparison", "homes"] + [c for c in P.columns if c.startswith("cross_user") or c.startswith("inj_room_swap") or c.startswith("test_mse")]
    w(md(P[[c for c in show if c in P]]))
    w("Same for mean injected AUROC:\n\n" + md(P[["comparison"] + [c for c in P.columns if c.startswith("inj_mean")]]))
    w("## 4. Learned graphs\n")
    w(md(G))
    w("`graph_same_room_share` vs `graph_same_room_chance`: do learned top-k neighbours stay inside rooms more than chance; "
      "`seed_jaccard`: overlap of edges between seeds of the same home.\n")
    w("## 5. Per home (cross-user day AUROC)\n")
    w(md(piv.reset_index()))
    w("## Figures\n\n" + "\n".join(f"- `figures/{f}`" for f in figs) + "\n")
    (out / "stage_e_report.md").write_text("\n".join(L))
    log.info("report -> %s", out / "stage_e_report.md")


if __name__ == "__main__":
    main()
