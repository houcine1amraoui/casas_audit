"""CASAS free-living, step 2: home structure, duplicates and cohort proposal (from the step-1 inventory; no raw data read).

python -m run_select --project_root_dir ROOT [--dataset_folder DATA] [--set selection.min_window_active_days=84 ...]
Reads outputs/casas_inventory/ ; writes outputs/casas_selection/: selection_report.md, homes_selection.csv,
sensor_map.csv (sensor -> room type, room, spot, kind), duplicates.csv.
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.structure import parse_name, residents

log = logging.getLogger("ca")


def md(df, fmt=".3g"):
    return df.to_markdown(index=False, floatfmt=fmt) + "\n" if len(df) else "_none_\n"


def duplicates(days, min_overlap=7, min_identical=0.9):
    """Pairs of home ids whose daily event counts are identical on >= min_identical of >= min_overlap shared days."""
    piv = days.pivot_table(index="day", columns="home", values="events")
    P, homes = piv.values, list(piv.columns)
    rows = []
    for i in range(len(homes)):
        a = P[:, i]
        for j in range(i + 1, len(homes)):
            b = P[:, j]
            m = ~np.isnan(a) & ~np.isnan(b)
            if m.sum() >= min_overlap:
                same = float((a[m] == b[m]).mean())
                if same >= min_identical:
                    rows.append(dict(home_a=homes[i], home_b=homes[j], shared_days=int(m.sum()), identical_share=round(same, 3)))
    return pd.DataFrame(rows, columns=["home_a", "home_b", "shared_days", "identical_share"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root_dir", required=True)
    ap.add_argument("--dataset_folder", default="")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)
    cfg = load_config(a.project_root_dir, a.dataset_folder, a.config, a.set)
    sel = cfg["selection"]
    inv = Path(a.project_root_dir) / "outputs" / cfg["run_name"]
    out = Path(a.project_root_dir) / "outputs" / "casas_selection"
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out / "run.log", mode="a")])
    H = pd.read_csv(inv / "homes.csv")
    S = pd.read_csv(inv / "sensors.csv")
    D = pd.read_csv(inv / "days.csv.gz")
    tabs = sorted(inv.glob("zenodo_table_*.csv"))
    res = residents(tabs[0]) if tabs else {}
    labeled = set(pd.read_csv(inv / "labeled_homes.csv").home) if (inv / "labeled_homes.csv").exists() else None
    fp = pd.read_csv(inv / "floorplans.csv") if (inv / "floorplans.csv").exists() else pd.DataFrame(columns=["home_guess"])
    floor = set(fp.home_guess) if len(fp) else None

    # ---- sensor map (the hierarchy: home -> room -> spot sensor)
    p = S.sensor.map(parse_name)
    S["room_type"], S["room"], S["spot"] = p.str[0], p.str[1], p.str[2]
    S["has_motion"] = S.messages.astype(str).str.contains(r"(?:^|,)ON:")
    S["has_door"] = S.messages.astype(str).str.contains(r"(?:^|,)OPEN:")
    S.to_csv(out / "sensor_map.csv", index=False)
    inroom = S[S.has_motion & ~S.room_type.isin(["Perimeter", "Other"])]
    g = inroom.groupby(["home", "room"]).size()
    st = pd.DataFrame({"n_rooms": g.groupby("home").size(), "rooms_2plus": (g >= 2).groupby("home").sum(),
                       "max_sensors_per_room": g.groupby("home").max(), "n_motion_in_rooms": inroom.groupby("home").size()})
    H = H.merge(st, left_on="home", right_index=True, how="left").fillna({"n_rooms": 0, "rooms_2plus": 0})
    H["coarse_names"] = H.n_rooms.gt(0) & (H.rooms_2plus == 0)           # one name per room (old mapping, e.g. hh)
    H["residents"] = H.home.map(res).fillna("unknown")
    H["cohort"] = H.home.str.extract(r"^([a-z]+)")[0]
    H["has_labels"] = H.home.isin(labeled) if labeled is not None else np.nan
    H["has_floorplan"] = H.home.isin(floor) if floor is not None else np.nan

    # ---- duplicates: keep the id with more active days
    dup = duplicates(D)
    drop = {}
    for r in dup.itertuples():
        aa, bb = H.set_index("home").loc[[r.home_a, r.home_b], "active_days"]
        keep, lose = (r.home_a, r.home_b) if aa >= bb else (r.home_b, r.home_a)
        drop[lose] = keep
    dup["kept"] = [r.home_a if r.home_b in drop and drop[r.home_b] == r.home_a else r.home_b for r in dup.itertuples()]
    dup.to_csv(out / "duplicates.csv", index=False)
    H["duplicate_of"] = H.home.map(drop)

    # ---- stable window: span where every core sensor (in-room motion sensor covering >= core_cover of the home span) reports
    S2 = S.merge(H[["home", "first", "last", "span_days"]].rename(columns={"first": "hf", "last": "hl", "span_days": "hspan"}), on="home")
    S2["cover"] = S2.span_days / S2.hspan.clip(lower=1e-9)
    core = S2[S2.has_motion & ~S2.room_type.isin(["Perimeter", "Other"]) & (S2.cover >= sel["core_cover"])]
    win = core.groupby("home").agg(win_start=("first", "max"), win_end=("last", "min"), n_core=("sensor", "size"))
    H = H.merge(win, left_on="home", right_index=True, how="left")
    D["day"] = pd.to_datetime(D.day)
    act = D[D.events >= cfg["inventory"]["active_day_min_events"]]
    wa, wr = [], []
    for r in H.itertuples():
        if pd.isna(r.win_start):
            wa.append(0); wr.append(0); continue
        s0, e0 = pd.Timestamp(r.win_start).normalize(), pd.Timestamp(r.win_end).normalize()
        dd = act[(act.home == r.home) & (act.day > s0) & (act.day < e0)].day.sort_values()   # first/last partial days excluded
        wa.append(len(dd))
        run = best = 0
        prev = None
        for x in dd:
            run = run + 1 if prev is not None and (x - prev).days == 1 else 1
            best = max(best, run); prev = x
        wr.append(best)
    H["window_active_days"], H["window_longest_run"] = wa, wr

    # ---- tiers
    struct = H.rooms_2plus >= sel["min_rooms_with_2plus"]
    enough = (H.window_active_days >= sel["min_window_active_days"]) & (H.window_longest_run >= sel["min_run_days"])
    clean = (H.bad_share < 0.01) & H.duplicate_of.isna()
    H["tier"] = "excluded"
    H.loc[clean & struct & enough & (H.residents == "1"), "tier"] = "core_single"
    H.loc[clean & struct & enough & H.residents.isin(["2", ">2"]), "tier"] = "multi_resident"
    H.loc[clean & struct & enough & (H.residents == "unknown"), "tier"] = "unknown_residents"
    why = []
    for r in H.itertuples():
        w = []
        if isinstance(r.duplicate_of, str): w.append(f"duplicate of {r.duplicate_of}")
        if r.bad_share >= 0.01: w.append("unparsable lines")
        if r.rooms_2plus < sel["min_rooms_with_2plus"]: w.append("coarse sensor names" if r.coarse_names else f"{int(r.rooms_2plus)} rooms with 2+ sensors")
        if r.window_active_days < sel["min_window_active_days"]: w.append(f"{r.window_active_days} active days in stable window")
        elif r.window_longest_run < sel["min_run_days"]: w.append(f"longest run {r.window_longest_run} d")
        why.append("; ".join(w))
    H["reason"] = why
    H.to_csv(out / "homes_selection.csv", index=False)

    # ---- report
    L = []
    w = L.append
    w("# CASAS free-living: home structure and cohort proposal (step 2)\n")
    w(f"{len(H)} home files from the inventory. Residents from the record table ({len(res)} homes listed). "
      f"Labels: {'not inventoried (rerun step 1 with the parent folder)' if labeled is None else len(labeled)}; "
      f"floorplans: {'not inventoried' if not floor else len(floor)}.\n")
    w("## Duplicated homes\n")
    w(f"{len(dup)} pairs of ids share identical daily event counts (same physical home under two ids). The id with more "
      "active days is kept; the other is excluded from every cohort, otherwise one home would count as two users.\n\n" + md(dup))
    w("## Sensor naming and hierarchy\n")
    w("Names parse as `<room type><letter><spot>` (e.g. `BedroomABed` = bedroom A, bed spot; `KitchenAArea` = whole kitchen A). "
      "This gives a natural hierarchy: spot sensor -> room -> home. Perimeter sensors (doors, entryways, windows) form their own group.\n\n")
    w(md(S.groupby("room_type").agg(sensors=("sensor", "size"), homes=("home", "nunique")).reset_index().sort_values("sensors", ascending=False)))
    w("Homes with **coarse names** (one name per room, e.g. `Kitchen`, `Bedroom`) cannot support the hierarchy: several physical sensors "
      "appear merged under one name (single names with millions of events). By cohort:\n\n")
    w(md(H.groupby("cohort").agg(homes=("home", "size"), coarse=("coarse_names", "sum"), median_rooms_2plus=("rooms_2plus", "median"),
                                 median_active_days=("active_days", "median")).reset_index()))
    w("## Cohort proposal\n")
    w(f"Rule: no duplicate, < 1% unparsable, >= {sel['min_rooms_with_2plus']} rooms with 2+ motion sensors, and inside the *stable window* "
      f"(span where every core sensor reports; core = in-room motion sensor covering >= {sel['core_cover']:.0%} of the home's span) "
      f">= {sel['min_window_active_days']} active days with a run of >= {sel['min_run_days']} consecutive active days.\n\n")
    w(md(H.groupby("tier").agg(homes=("home", "size")).reset_index()))
    cols = ["home", "residents", "win_start", "win_end", "window_active_days", "window_longest_run", "n_sensors", "n_core", "n_rooms",
            "rooms_2plus", "median_events_per_active_day", "has_labels", "has_floorplan"]
    for t in ("core_single", "multi_resident", "unknown_residents"):
        x = H[H.tier == t].sort_values("window_active_days", ascending=False)
        w(f"### {t} ({len(x)})\n\n" + md(x[cols]))
    w("### Excluded homes by reason\n")
    w(md(H[H.tier == "excluded"].assign(first_reason=lambda d: d.reason.str.split(";").str[0].str.replace(r"\d+", "N", regex=True))
         .groupby("first_reason").size().reset_index(name="homes").sort_values("homes", ascending=False)))
    (out / "selection_report.md").write_text("\n".join(L))
    log.info("tiers: %s -> %s", H.tier.value_counts().to_dict(), out)


if __name__ == "__main__":
    main()
