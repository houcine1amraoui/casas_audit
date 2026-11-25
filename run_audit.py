"""CASAS free-living, step 3: data-quality and behaviour audit of the selected homes (binary motion/door events).

python -m run_audit --project_root_dir ROOT --dataset_folder DATA [--set audit.tiers=[core_single,multi_resident] ...]
Needs outputs/casas_inventory/ (step 1) and outputs/casas_selection/ (step 2).
Writes outputs/casas_audit/ (audit_report.md, CSVs, figures/) and cache/casas/features/<home>.npz.
"""
import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_inventory import locate
from src import consistency, home_audit, labels
from src.config import load_config

log = logging.getLogger("ca")


def md(df, fmt=".3g"):
    return df.to_markdown(index=False, floatfmt=fmt) + "\n" if df is not None and len(df) else "_none_\n"


def find_file(home, member, roots):
    if isinstance(member, str) and os.path.exists(member):
        return member
    for root in roots:
        if root is None:
            continue
        for d, _, fs in os.walk(root):
            for f in fs:
                stem = f.split(".")[0]
                if stem == home:
                    return os.path.join(d, f)
    return None


def _job(args):
    home, path, ws, we, smap, cfg = args
    return home_audit.audit_home(home, path, ws, we, smap, cfg)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root_dir", required=True)
    ap.add_argument("--dataset_folder", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)
    cfg = load_config(a.project_root_dir, a.dataset_folder, a.config, a.set)
    A = cfg["audit"]
    root = Path(a.project_root_dir) / "outputs"
    out = root / A["run_name"]
    (out / "figures" / "homes").mkdir(parents=True, exist_ok=True)
    feat_dir = Path(cfg["paths"]["cache_dir"]) / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out / "run.log", mode="a")])
    if not (root / "casas_selection" / "homes_selection.csv").exists() or not (root / "casas_selection" / "sensor_map.csv").exists():
        if not (root / cfg["run_name"] / "homes.csv").exists():
            raise SystemExit(f"step-1 inventory not found in {root / cfg['run_name']}: run run_inventory first")
        log.info("step-2 selection not found: running run_select first")
        import run_select
        run_select.main(["--project_root_dir", a.project_root_dir] + (["--config", a.config] if a.config else [])
                        + [x for s in a.set for x in ("--set", s)])
    H = pd.read_csv(root / "casas_selection" / "homes_selection.csv")
    SM = pd.read_csv(root / "casas_selection" / "sensor_map.csv")
    sel = H[H.tier.isin(A["tiers"])].sort_values("home").reset_index(drop=True)
    if sel.empty:
        raise SystemExit(f"no homes in tiers {A['tiers']}")
    cfg["source"] = cfg.get("source", "auto")
    cfg["zenodo"]["download"] = False
    files = locate(cfg)
    data_dir = files.get("data.zip")
    lab_dir = files.get("labeled_data.zip")
    jobs = []
    for r in sel.itertuples():
        p = find_file(r.home, r.member, [data_dir])
        if p is None:
            log.warning("%s: data file not found, skipped", r.home)
            continue
        jobs.append((r.home, p, r.win_start, r.win_end, SM[SM.home == r.home][["sensor", "room_type", "room", "spot"]], cfg))
    log.info("auditing %d homes (tiers %s)", len(jobs), A["tiers"])
    res = {}
    with ProcessPoolExecutor(max_workers=A["workers"] or os.cpu_count() or 2) as ex:
        for r in ex.map(_job, jobs):
            res[r["home"]] = r
            log.info("  %-8s events=%-9d days=%-4d active=%-4d usable streams=%d/%d", r["home"], r["n_events"], len(r["days"]),
                     int(r["active_mask"].sum()), len(r["streams"]), len(r["sensors"]))
    homes = sorted(res)
    S = pd.concat([res[h]["sensors"] for h in homes], ignore_index=True)
    D = pd.concat([res[h]["days"] for h in homes], ignore_index=True)
    G = pd.concat([res[h]["gaps"] for h in homes if len(res[h]["gaps"])], ignore_index=True) if any(len(res[h]["gaps"]) for h in homes) else pd.DataFrame()
    S.to_csv(out / "sensor_audit.csv", index=False)
    D.to_csv(out / "home_days.csv", index=False)
    G.to_csv(out / "silences.csv", index=False)
    for h in homes:
        r = res[h]
        np.savez_compressed(feat_dir / f"{h}.npz", days=r["day_index"].values.astype("datetime64[D]").astype(np.int64),
                            active=r["active_mask"], Xs=r["Xs"], Xg_share=r["Xg_share"], Xh_share=r["Xh_share"],
                            log_total=r["log_total"], Xs_names=np.array(r["Xs_names"]))

    # ---------------- home summary
    hs = []
    for h in homes:
        r, s = res[h], res[h]["sensors"]
        gh = G[G.home == h] if len(G) else pd.DataFrame(columns=["hours", "door_or_entry_before"])
        hs.append(dict(home=h, window_days=len(r["days"]), active_days=int(r["active_mask"].sum()),
                       median_daily_activations=float(np.median(r["days"].activations[r["active_mask"]])) if r["active_mask"].any() else 0,
                       streams=len(s), usable=int(s.usable.sum()), **{f"cls_{k}": int(v) for k, v in s.cls.value_counts().items()},
                       silences=len(gh), silences_after_door=int(gh.door_or_entry_before.sum()) if len(gh) else 0,
                       silence_hours=float(gh.hours.sum()) if len(gh) else 0.0, backwards=r["backwards"],
                       ignored_messages=sum(r["ignored"].values())))
    HS = pd.DataFrame(hs).fillna(0)
    HS.to_csv(out / "home_summary.csv", index=False)

    # ---------------- within-home consistency (sensor features and room-group features)
    W = A["within"]
    wrows = []
    for h in homes:
        r = res[h]
        m = r["active_mask"]
        for fs, X in (("sensors", r["Xs"]), ("room_groups", np.c_[r["Xg_share"], r["log_total"]])):
            out_w = consistency.within_home(X[m], r["day_index"][m], W["ref_days"], W["late_days"], W["n_perm"], W["seed"], A["std_floor"])
            wrows.append(dict(home=h, features=fs, **out_w))
    WT = pd.DataFrame(wrows)
    WT.to_csv(out / "within_home.csv", index=False)

    # ---------------- cross-home identification
    cohort = dict(zip(H.home, H.cohort))
    I = A["ident"]
    fsets = {"time_only": lambda r: np.c_[r["Xh_share"], r["log_total"]],
             "room_groups": lambda r: np.c_[r["Xg_share"], r["log_total"]],
             "time_and_room_groups": lambda r: np.c_[r["Xh_share"], r["Xg_share"], r["log_total"]]}
    ident_rows, ident_home, Dmats = [], [], {}
    for name, f in fsets.items():
        F = {h: f(res[h])[res[h]["active_mask"]] for h in homes}
        summ, R, Dm = consistency.identification(F, homes, cohort, A["std_floor"], I["profile_frac"], I["test_frac"])
        ident_rows.append(dict(features=name, **summ))
        ident_home.append(R.assign(features=name))
        Dmats[name] = Dm
    IT = pd.DataFrame(ident_rows)
    IH = pd.concat(ident_home, ignore_index=True)
    IT.to_csv(out / "identification.csv", index=False)
    IH.to_csv(out / "identification_by_home.csv", index=False)

    # ---------------- labels (homes in the audit that have a labelled file)
    LT, LS = [], []
    if lab_dir is not None:
        for h in homes:
            p = find_file(h, None, [lab_dir])
            if p is None:
                continue
            segs, E, unclosed = labels.read_segments(p, A["label_join_gap_s"])
            if segs.empty:
                continue
            sm = SM[SM.home == h]
            tab = labels.label_room_table(segs, E, dict(zip(sm.sensor, sm.room_type)), home_audit.group_of)
            LT.append(tab.assign(home=h))
            LS.append(dict(home=h, segments=len(segs), activities=segs.activity.nunique(), unclosed_markers=unclosed,
                           styles=",".join(f"{k}:{v}" for k, v in segs["style"].value_counts().items()),
                           first=pd.Timestamp(segs.start.min(), unit="s"), last=pd.Timestamp(segs.end.max(), unit="s")))
            segs.assign(home=h).to_csv(out / f"label_segments_{h}.csv", index=False)
    LT = pd.concat(LT, ignore_index=True) if LT else pd.DataFrame()
    LS = pd.DataFrame(LS)
    LT.to_csv(out / "label_room_shares.csv", index=False)
    LS.to_csv(out / "label_summary.csv", index=False)

    # ---------------- figures
    figs = figures(out / "figures", res, homes, HS, IT, IH, Dmats, LT, cohort, A)

    # ---------------- report
    L = []
    w = L.append
    w("# CASAS free-living: audit of the selected homes (step 3)\n")
    w(f"{len(homes)} homes (tiers {A['tiers']}), whole days inside each home's stable window. Streams: motion ON/OFF and door "
      "OPEN/CLOSE; an activation = ON or OPEN. Other messages (temperatures, 0/100, button taps) are counted and ignored.\n")
    w("## 1. Integrity\n")
    w(md(HS[["home", "window_days", "active_days", "median_daily_activations", "silences", "silences_after_door", "silence_hours",
             "backwards", "ignored_messages"]]))
    w(f"Silences = event-free stretches >= {A['silence_hours']} h; `silences_after_door` = preceded by a door or entry event "
      "(likely the resident away) rather than a system outage.\n")
    w("## 2. Sensors\n")
    w(md(S.groupby("cls").agg(streams=("stream", "size"), homes=("home", "nunique")).reset_index().sort_values("streams", ascending=False)))
    w(f"Classes: `silent_in_window` (no activation in the window), `rare` (median < {A['classify']['rare_median_daily']}/day, kept), "
      f"`stuck` (ON held >= {A['stuck_hours']} h for > {A['classify']['stuck_time_share']:.0%} of the time), `long_dropout` "
      f"(silent >= {A['classify']['dropout_run_days']} active days in a row), `intermittent` (silent on > {A['classify']['dropout_share']:.0%} "
      "of active days), `chatter`. Usable = ok + rare.\n\n")
    bad = S[~S.usable].sort_values(["home", "cls"])[["home", "stream", "room_type", "cls", "activations", "median_daily",
                                                      "longest_dropout_run", "stuck_time_share", "unpaired_on_share"]]
    w("Not usable:\n\n" + md(bad))
    w("ON-duration and pairing (usable motion streams): median ON " +
      f"{S[S.usable & ~S.is_door.astype(bool)].median_on_s.median():.1f} s; unpaired ON share median "
      f"{S[S.usable].unpaired_on_share.median():.3f}.\n")
    w("## 3. Within-home consistency (first vs last 28 active days)\n")
    for fs in ("sensors", "room_groups"):
        x = WT[(WT.features == fs) & (WT.status == "ok")]
        w(f"**{fs}**: {len(x)} homes; permutation p < 0.05 in {int((x.perm_p < 0.05).sum())}; median cross/within distance ratio "
          f"{(x.cross / x.within).median():.3f}; drift rho > 0 and p < 0.05 in {int(((x.drift_rho > 0) & (x.drift_p < 0.05)).sum())}; "
          f"median weekday ratio {x.weekday_ratio.median():.3f}.\n")
    w(md(WT[WT.features == "sensors"][["home", "n_days", "within", "cross", "perm_p", "drift_rho", "drift_p", "weekday_ratio"]]))
    w("## 4. Cross-home distinguishability\n")
    w(f"Profile = first {I['profile_frac']:.0%} of each home's active days, test = last {I['test_frac']:.0%} (gap between). Each test "
      "day is matched to the nearest home profile (standardised on all profile days). `time_only` has no room information "
      "(hourly shares of activations + daily volume), so layout cannot drive it; `room_groups` uses six room groups x six "
      "4-h bins (shares) present in every home.\n\n")
    w(md(IT))
    w("`top1_within_cohort` = identification among homes of the same deployment (e.g. all `tm`), which removes era and "
      "installation differences between deployments.\n")
    w("## 5. Labels\n")
    if len(LS):
        w(md(LS))
        if len(LT):
            agg = LT.groupby("activity").apply(lambda d: pd.Series({g: np.average(d[g].fillna(0), weights=d.activations)
                                                                    for g in home_audit.GROUPS if g in d}), include_groups=False)
            agg["homes"] = LT.groupby("activity").home.nunique()
            min_h = min(3, LT.home.nunique())
            agg = agg[agg.homes >= min_h].reset_index().sort_values("homes", ascending=False)
            w(f"Share of activations inside each activity's labelled segments by room group (activities in >= {min_h} homes). "
              "Sleep should sit in Bedroom, Toilet/Personal_Hygiene in Bathroom, Cook_* in Kitchen if the room parsing is right:\n\n" + md(agg, ".2f"))
    else:
        w("No labelled files for the audited homes.\n")
    w("## Figures\n\n" + "\n".join(f"- `figures/{f}`" for f in figs) + "\n")
    (out / "audit_report.md").write_text("\n".join(L))
    log.info("done -> %s", out)


def figures(fd, res, homes, HS, IT, IH, Dmats, LT, cohort, A):
    figs = []
    # activity over each home's window (relative day)
    fig, ax = plt.subplots(figsize=(12, 0.22 * len(homes) + 1.5))
    mx = max(len(res[h]["days"]) for h in homes)
    M = np.full((len(homes), mx), np.nan)
    for i, h in enumerate(homes):
        v = res[h]["days"].activations.values.astype(float)
        med = np.median(v[v > 0]) if (v > 0).any() else 1
        M[i, :len(v)] = np.clip(v / med, 0, 2)
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=2, interpolation="nearest")
    ax.set_yticks(range(len(homes))); ax.set_yticklabels(homes, fontsize=6)
    ax.set_xlabel("day in stable window"); fig.colorbar(im, ax=ax, label="daily activations / home median")
    ax.set_title("Daily activity per home (dark = silent day)")
    fig.tight_layout(); fig.savefig(fd / "homes_daily_activity.png", dpi=110); plt.close(fig); figs.append("homes_daily_activity.png")
    # identification
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(IT))
    ax.bar(x - 0.2, IT.top1, 0.4, label="top-1, all homes"); ax.bar(x + 0.2, IT.top1_within_cohort, 0.4, label="top-1, within deployment")
    ax.axhline(IT.chance_top1.iat[0], color="k", ls="--", lw=.8); ax.text(len(IT) - .5, IT.chance_top1.iat[0], " chance", fontsize=7, va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(IT.features, fontsize=8); ax.set_ylim(0, 1); ax.legend(fontsize=7)
    ax.set_title("Which home is this day from? (test days vs home profiles)")
    fig.tight_layout(); fig.savefig(fd / "identification.png", dpi=110); plt.close(fig); figs.append("identification.png")
    for name, Dm in Dmats.items():
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(Dm.values, cmap="magma")
        ax.set_xticks(range(len(Dm))); ax.set_xticklabels(Dm.columns, rotation=90, fontsize=5)
        ax.set_yticks(range(len(Dm))); ax.set_yticklabels(Dm.index, fontsize=5)
        fig.colorbar(im, ax=ax); ax.set_title(f"Distance between home profiles ({name})")
        f = f"profile_distances_{name}.png"
        fig.tight_layout(); fig.savefig(fd / f, dpi=110); plt.close(fig); figs.append(f)
    # per-home sensor x day heatmaps
    if A["figures_per_home"]:
        for h in homes:
            r = res[h]
            s = r["sensors"].sort_values(["group", "stream"])
            names = s.stream.tolist()
            mat = np.zeros((len(names), len(r["days"])))
            cols = {n: i for i, n in enumerate(r["all_streams"])}
            for i, n in enumerate(names):
                if n in cols:
                    mat[i] = r["Xall"][:, cols[n]]
            fig, ax = plt.subplots(figsize=(12, 0.2 * len(names) + 1.5))
            ax.imshow(np.log1p(mat), aspect="auto", cmap="viridis", interpolation="nearest")
            ax.set_yticks(range(len(names))); ax.set_yticklabels([f"{n} [{c}]" for n, c in zip(names, s.cls)], fontsize=5)
            ax.set_title(f"{h}: activations per sensor and day (log); [class]"); ax.set_xlabel("day in window")
            fig.tight_layout(); fig.savefig(fd / "homes" / f"{h}.png", dpi=100); plt.close(fig)
        figs.append("homes/<home>.png")
    return figs


if __name__ == "__main__":
    main()
