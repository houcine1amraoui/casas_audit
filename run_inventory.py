"""CASAS free-living, step 1: download + verify + inventory of every home. Nothing is filtered here.

python -m run_inventory --project_root_dir ROOT --dataset_folder DATA [--config f.yaml] [--set a.b=c ...]
Outputs in {ROOT}/outputs/casas_inventory/: inventory_report.md, homes.csv, sensors.csv, days.csv.gz, gaps.csv,
labeled_homes.csv, labeled_sensors.csv, floorplans.csv, zenodo_description.txt (+ zenodo_tables_*.csv if parsable).
"""
import argparse
import html
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

from src import download, inventory
from src.config import load_config

log = logging.getLogger("ca")


def md(df, n=None, fmt=".3g"):
    if df is None or len(df) == 0:
        return "_none_\n"
    d = df if n is None else df.head(n)
    return d.to_markdown(index=False, floatfmt=fmt) + "\n"


def describe_record(meta_path, out):
    """Save the record's description as text and try to pull its home/resident table(s)."""
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    desc = meta.get("metadata", {}).get("description", "")
    text = html.unescape(re.sub(r"<[^>]+>", " ", re.sub(r"</(p|tr|li|h\d)>|<br\s*/?>", "\n", desc)))
    text = re.sub(r"[ \t]+", " ", text)
    (out / "zenodo_description.txt").write_text(text)
    try:
        from io import StringIO
        tabs = pd.read_html(StringIO(desc))
        for i, t in enumerate(tabs):
            t.to_csv(out / f"zenodo_table_{i}.csv", index=False)
        log.info("record description: %d table(s) saved", len(tabs))
    except Exception as e:
        log.info("record description has no parsable html table (%s); see zenodo_description.txt", type(e).__name__)
    return text


def find_dir(root, name, depth=3):
    """First folder called `name` (case-insensitive) under root, breadth-first, at most `depth` levels down."""
    level = [Path(root)]
    for _ in range(depth + 1):
        nxt = []
        for d in level:
            if d.name.lower() == name.lower() and d != Path(root):
                return d
            try:
                nxt += sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith((".", "__MACOSX")))
            except PermissionError:
                pass
        level = nxt
    return None


def locate(cfg):
    """{'data.zip'|'labeled_data.zip'|'floorplans.zip': path to the folder or the zip}. Folders win in 'auto'."""
    mode = cfg["source"]
    root = Path(cfg["dataset_folder"])
    if mode in ("auto", "folders"):
        f = cfg["folders"]
        dirs = {}
        for k in ("data", "labeled", "floorplans"):
            d = find_dir(root, f[k])
            if d is None and root.name.lower() == f[k].lower():
                d = root
            if d is None:          # DATASET_FOLDER may point at data/ itself: look beside it
                d = find_dir(root.parent, f[k], depth=1)
            dirs[k] = d
        if dirs["data"] is not None:
            log.info("using unzipped folders (no download): %s", ", ".join(f"{k}={v}" for k, v in dirs.items()))
            download.fetch_metadata(cfg)
            return {n: dirs[k] for n, k in (("data.zip", "data"), ("labeled_data.zip", "labeled"), ("floorplans.zip", "floorplans"))
                    if dirs[k] is not None}
        if mode == "folders":
            raise FileNotFoundError(f"no folder named '{f['data']}' under {root} (searched 3 levels deep)")
        log.info("no '%s' folder under %s: falling back to zips", f["data"], root)
    return download.fetch(cfg)


def candidates(H, cfg):
    s = cfg["selection"]
    c = H.copy()
    c["meets_active_days"] = c.active_days >= s["min_active_days"]
    c["meets_run"] = c.longest_active_run >= s["min_run_days"]
    c["meets_sensors"] = c.n_sensors >= s["min_sensors"]
    c["meets_clean"] = c.bad_share < 0.01
    c["candidate"] = c[["meets_active_days", "meets_run", "meets_sensors", "meets_clean"]].all(axis=1)
    return c


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root_dir", required=True)
    ap.add_argument("--dataset_folder", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)
    cfg = load_config(a.project_root_dir, a.dataset_folder, a.config, a.set)
    out = Path(cfg["paths"]["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["local_dir"]).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out / "run.log", mode="a")])
    files = locate(cfg)
    desc = describe_record(Path(cfg["dataset_folder"]) / f"zenodo_{cfg['zenodo']['record']}.json", out)

    parts = cfg["inventory"]["parts"]
    if "data" in parts:
        H, S, D, G, other = inventory.run(files["data.zip"], cfg, "data")
        H = candidates(H, cfg)
        H.to_csv(out / "homes.csv", index=False)
        S.to_csv(out / "sensors.csv", index=False)
        D.to_csv(out / "days.csv.gz", index=False)
        G.to_csv(out / "gaps.csv", index=False)
    else:                                 # reuse the data inventory already in the output folder
        log.info("inventory.parts has no 'data': reusing %s/homes.csv etc.", out)
        H = candidates(pd.read_csv(out / "homes.csv"), cfg)
        S = pd.read_csv(out / "sensors.csv")
        other = []

    LH = LS = None
    if "labeled" in parts and "labeled_data.zip" in files:
        LH, LS, LD, LG, _ = inventory.run(files["labeled_data.zip"], cfg, "labeled")
        LH.to_csv(out / "labeled_homes.csv", index=False)
        LS.to_csv(out / "labeled_sensors.csv", index=False)
    elif (out / "labeled_homes.csv").exists():
        LH = pd.read_csv(out / "labeled_homes.csv")
    if "floorplans" in parts or not (out / "floorplans.csv").exists():
        FP = inventory.list_zip(files["floorplans.zip"]) if "floorplans.zip" in files else pd.DataFrame(columns=["name", "bytes"])
    else:
        FP = pd.read_csv(out / "floorplans.csv")
    FP["home_guess"] = FP.name.map(lambda n: inventory.home_id(n).split(".")[0])
    FP.to_csv(out / "floorplans.csv", index=False)

    # ---------------- report
    L = []
    w = L.append
    w("# CASAS free-living: inventory (step 1)\n")
    folder_mode = any(Path(p).is_dir() for p in files.values())
    w(f"Record {cfg['zenodo']['record']}. Source: " + ", ".join(f"`{p}`" for p in files.values())
      + (" (unzipped folders; md5 not checked)" if folder_mode else (" (md5 verified)" if cfg["zenodo"]["verify_md5"] else "")) + ".")
    w(f"\n**{len(H)} home files**, {int(H.lines.sum()):,} lines, {int(H.bad_lines.sum()):,} unparsable "
      f"({H.bad_share.max():.2%} worst home), {int(H.duplicate_home_id.sum())} files sharing a home id.\n")
    if other:
        w("Non-data files in data.zip: " + ", ".join(f"`{o}`" for o in other[:30]) + "\n")
    w("## Line layout and fields\n")
    lay = H.layout.astype(str).str.replace(r":\d+", "", regex=True)
    w(md(lay.value_counts().rename_axis("layout").reset_index(name="homes")))
    ex = H[H.extra_fields.astype(str) != H.extra_fields.astype(str).str.extract(r"^(0:\d+)$")[0].fillna("#")]
    w(f"Homes with extra fields after the message: {len(ex)}.\n\n" + md(ex[["home", "extra_fields", "top_extra"]], 15))
    w("## Coverage\n")
    w(md(H[["span_days", "active_days", "longest_active_run", "n_sensors", "median_events_per_active_day"]].describe().T.reset_index(), fmt=".1f"))
    w("## Sensor kinds (from message vocabulary)\n")
    k = S.groupby("kind").agg(sensors=("sensor", "size"), homes=("home", "nunique")).reset_index().sort_values("sensors", ascending=False)
    w(md(k))
    vals = sorted({kv.rsplit(":", 1)[0] for m in H.messages for kv in str(m).split(",") if kv})
    w("Message values across all homes (first 60): " + ", ".join(f"`{v}`" for v in vals[:60]) + "\n")
    w("## Data defects\n")
    w(md(H.sort_values("bad_share", ascending=False)[["home", "lines", "bad_lines", "bad_share", "timestamps_backwards", "max_backwards_s", "exact_duplicates", "bad_examples"]], 15))
    w("## Candidate homes (proposal)\n")
    s = cfg["selection"]
    w(f"Rule: >= {s['min_active_days']} active days (>= {cfg['inventory']['active_day_min_events']} events), longest active run >= {s['min_run_days']} days, "
      f">= {s['min_sensors']} sensors, < 1% unparsable lines. Residents per home are NOT in the files: take them from the record table "
      f"(`zenodo_description.txt`). **{int(H.candidate.sum())} of {len(H)} homes pass.**\n")
    cols = ["home", "first", "last", "span_days", "active_days", "longest_active_run", "n_sensors", "n_area_sensors",
            "sensor_kinds", "median_events_per_active_day", "gaps_ge_24h"]
    w(md(H[H.candidate].sort_values("active_days", ascending=False)[cols]))
    w("Failing homes by reason: " + ", ".join(f"{c[6:]}: {int((~H[c]).sum())}" for c in ("meets_active_days", "meets_run", "meets_sensors", "meets_clean")) + "\n")
    w("## Labels and floorplans\n")
    if LH is not None:
        lab = set(LH.home)
        w(f"Labelled files: {len(LH)}; candidate homes with labels: {int(H[H.candidate].home.isin(lab).sum())}. "
          "Label fields (first files):\n\n" + md(LH[["home", "lines", "first", "last", "extra_fields", "top_extra"]], 10))
    w(f"Floorplan files: {len(FP)}; candidate homes with a floorplan: {int(H[H.candidate].home.isin(set(FP.home_guess)).sum())}.\n")
    w("## Sensor names (first 5 candidate homes)\n")
    for h in H[H.candidate].sort_values("active_days", ascending=False).home.head(5):
        w(f"- **{h}**: " + ", ".join(f"{r.sensor} ({r.kind}, {r.events})" for r in S[S.home == h].itertuples()) + "\n")
    (out / "inventory_report.md").write_text("\n".join(L))
    log.info("done -> %s", out)


if __name__ == "__main__":
    main()
