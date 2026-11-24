"""Per-home inventory of CASAS event files, streamed straight from the zip (no extraction, no pandas on raw lines).

Per home: lines, parse failures (with examples), layout, time span, timestamps going backwards, exact duplicate
events, event-free gaps, active days, longest run of active days, sensors and their message vocabulary, extra
fields (labels). Per sensor: events, kind guessed from its messages, first/last time. Per day: events, sensors.
"""
import bz2
import gzip
import logging
import os
import re
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import PurePosixPath

import pandas as pd

from .parse import parse

log = logging.getLogger("ca")
DATA_EXT = (".txt", ".csv", ".tsv", ".dat", "")
COMPRESSED = (".gz", ".bz2")
ARCHIVES = (".zip",)


def home_id(member):
    stem = PurePosixPath(member).name
    for ext in COMPRESSED + (".txt", ".csv", ".tsv", ".dat"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
    return stem


def _skip(name):
    n = PurePosixPath(name).name
    return "__MACOSX" in name or n.startswith(".") or n.lower() in ("readme", "readme.txt", "license", "license.txt")


def sources(zip_path, extract_dir):
    """Data sources inside the zip -> list of (kind, container, member, size). Inner archives (.zip) are extracted
    to extract_dir and their files listed; .gz/.bz2 members are read compressed. Non-data files are reported apart."""
    out, other = [], []
    with zipfile.ZipFile(zip_path) as z:
        for i in z.infolist():
            if i.is_dir() or _skip(i.filename):
                continue
            suf = PurePosixPath(i.filename).suffix.lower()
            if suf in ARCHIVES:
                dest = os.path.join(extract_dir, PurePosixPath(i.filename).stem)
                if not os.path.isdir(dest):
                    os.makedirs(dest, exist_ok=True)
                    tmp = z.extract(i, extract_dir)
                    with zipfile.ZipFile(tmp) as inner:
                        inner.extractall(dest)
                    os.remove(tmp)
                for root, _, files in os.walk(dest):
                    for f in files:
                        fp = os.path.join(root, f)
                        if not _skip(fp) and (PurePosixPath(f).suffix.lower() in DATA_EXT + COMPRESSED):
                            out.append(("file", fp, fp, os.path.getsize(fp)))
            elif suf in DATA_EXT + COMPRESSED:
                out.append(("zip", str(zip_path), i.filename, i.file_size))
            else:
                other.append(i.filename)
    return out, other


def sources_dir(root, extract_dir):
    """Same as `sources`, for an already unzipped folder (walked recursively)."""
    out, other = [], []
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            fp = os.path.join(dirpath, f)
            if _skip(fp):
                continue
            suf = PurePosixPath(f).suffix.lower()
            if suf in ARCHIVES:
                dest = os.path.join(extract_dir, PurePosixPath(f).stem)
                if not os.path.isdir(dest):
                    os.makedirs(dest, exist_ok=True)
                    with zipfile.ZipFile(fp) as inner:
                        inner.extractall(dest)
                for r2, _, f2 in os.walk(dest):
                    for g in f2:
                        gp = os.path.join(r2, g)
                        if not _skip(gp) and PurePosixPath(g).suffix.lower() in DATA_EXT + COMPRESSED:
                            out.append(("file", gp, gp, os.path.getsize(gp)))
            elif suf in DATA_EXT + COMPRESSED:
                out.append(("file", fp, fp, os.path.getsize(fp)))
            else:
                other.append(os.path.relpath(fp, root))
    return out, other


def open_source(kind, container, member):
    if kind == "zip":
        z = zipfile.ZipFile(container)
        fh = z.open(member)
    else:
        z, fh = None, open(member, "rb")
    low = member.lower()
    if low.endswith(".gz"):
        fh = gzip.GzipFile(fileobj=fh)
    elif low.endswith(".bz2"):
        fh = bz2.BZ2File(fh)
    return z, fh


def sensor_kind(msgs):
    keys = {k.upper() for k in msgs}
    if keys <= {"ON", "OFF"}:
        return "motion_like"
    if keys <= {"OPEN", "CLOSE", "CLOSED"}:
        return "door_like"
    num = sum(v for k, v in msgs.items() if re.fullmatch(r"-?\d+(\.\d+)?", k))
    if num >= 0.9 * sum(msgs.values()):
        return "numeric"
    return "mixed:" + ",".join(sorted(keys)[:4])


def profile_member(args):
    kind, container, member, cfg = args
    inv = cfg["inventory"]
    hid = home_id(member)
    n = bad = header = back = dup = 0
    max_back = 0.0
    layouts = Counter()
    extra_n = Counter()
    extra_vals = Counter()
    bad_ex = []
    sens = defaultdict(lambda: {"n": 0, "msgs": Counter(), "first": None, "last": None})
    days = defaultdict(lambda: [0, set()])
    gaps = []
    prev_ts, prev_key = None, None
    first_ts = last_ts = None
    thr = sorted(inv["gap_hours"])
    z, fh = open_source(kind, container, member)
    try:
        for raw in fh:
                n += 1
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                r = parse(line)
                if r is None:
                    if n <= 3 and re.search(r"[A-Za-z]", line[:10]):
                        header += 1
                    else:
                        bad += 1
                        if len(bad_ex) < inv["max_bad_examples"]:
                            bad_ex.append(line[:160])
                    continue
                d, ts, s, msg, extra, lay = r
                layouts[lay] += 1
                extra_n[len(extra)] += 1
                if extra:
                    extra_vals[" ".join(extra)[:60]] += 1
                key = (ts, s, msg)
                if key == prev_key:
                    dup += 1
                if prev_ts is not None:
                    dt = ts - prev_ts
                    if dt < 0:
                        back += 1
                        max_back = max(max_back, -dt)
                    elif dt >= thr[0] * 3600:
                        gaps.append((dt, prev_ts))
                prev_ts, prev_key = ts, key
                first_ts = ts if first_ts is None else min(first_ts, ts)
                last_ts = ts if last_ts is None else max(last_ts, ts)
                e = sens[s]
                e["n"] += 1
                e["msgs"][msg] += 1
                e["first"] = ts if e["first"] is None else min(e["first"], ts)
                e["last"] = ts if e["last"] is None else max(e["last"], ts)
                dd = days[d]
                dd[0] += 1
                dd[1].add(s)

    finally:
        fh.close()
        if z is not None:
            z.close()
    def iso(t):
        if t is None:
            return None
        return pd.Timestamp(round((t - date(1970, 1, 1).toordinal() * 86400) * 1e6), unit="us").isoformat()

    parsed = n - bad - header
    dayrows = [dict(home=hid, day=d, events=v[0], sensors=len(v[1])) for d, v in sorted(days.items())]
    act = [r for r in dayrows if r["events"] >= inv["active_day_min_events"]]
    # longest run of consecutive active days
    best = cur = 0
    prev = None
    for r in act:
        o = date.fromisoformat(r["day"]).toordinal()
        cur = cur + 1 if prev is not None and o == prev + 1 else 1
        best = max(best, cur)
        prev = o
    span_days = (last_ts - first_ts) / 86400 if first_ts is not None else 0
    allmsgs = Counter()
    for e in sens.values():
        allmsgs.update(e["msgs"])
    kinds = Counter(sensor_kind(e["msgs"]) for e in sens.values())
    names = sorted(sens)
    home = dict(home=hid, member=member, lines=n, parsed=parsed, header_lines=header, bad_lines=bad,
                bad_share=bad / max(n, 1), layout=",".join(f"{k}:{v}" for k, v in layouts.most_common()),
                first=iso(first_ts), last=iso(last_ts), span_days=round(span_days, 1),
                calendar_days=len(dayrows), active_days=len(act), longest_active_run=best,
                active_share_of_span=len(act) / max(span_days, 1),
                median_events_per_active_day=float(pd.Series([r["events"] for r in act]).median()) if act else 0,
                n_sensors=len(sens), n_area_sensors=sum(1 for s in names if s.endswith("Area")),
                sensor_kinds=",".join(f"{k}:{v}" for k, v in kinds.most_common()),
                messages=",".join(f"{k}:{v}" for k, v in allmsgs.most_common(inv["top_messages"])),
                n_message_values=len(allmsgs),
                timestamps_backwards=back, max_backwards_s=round(max_back, 3), exact_duplicates=dup,
                extra_fields=",".join(f"{k}:{v}" for k, v in sorted(extra_n.items())),
                top_extra=" | ".join(f"{k} ({v})" for k, v in extra_vals.most_common(6)),
                bad_examples=" || ".join(bad_ex))
    for h in thr:
        home[f"gaps_ge_{h}h"] = sum(1 for g, _ in gaps if g >= h * 3600)
    longest = sorted(gaps, reverse=True)[:10]
    gap_rows = [dict(home=hid, start=iso(t0), hours=round(g / 3600, 2)) for g, t0 in longest]
    sensor_rows = [dict(home=hid, sensor=s, events=e["n"], kind=sensor_kind(e["msgs"]),
                        messages=",".join(f"{k}:{v}" for k, v in e["msgs"].most_common(inv["top_messages"])),
                        first=iso(e["first"]), last=iso(e["last"]),
                        span_days=round((e["last"] - e["first"]) / 86400, 1)) for s, e in sorted(sens.items())]
    return home, sensor_rows, dayrows, gap_rows


def run(zip_path, cfg, tag):
    ex_dir = os.path.join(cfg["paths"]["local_dir"], tag)
    src, other = (sources_dir(zip_path, ex_dir) if os.path.isdir(zip_path) else sources(zip_path, ex_dir))
    log.info("%s: %d data files, %d other files%s", tag, len(src), len(other), (": " + ", ".join(other[:10])) if other else "")
    jobs = [(k, c, m, cfg) for k, c, m, _ in sorted(src, key=lambda x: -x[3])]   # biggest first
    workers = cfg["inventory"]["workers"] or os.cpu_count() or 2
    homes, sensors, days, gaps = [], [], [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (h, s, d, g) in enumerate(ex.map(profile_member, jobs)):
            homes.append(h); sensors += s; days += d; gaps += g
            log.info("  [%3d/%d] %-14s lines=%-10d bad=%-6d sensors=%-3d active_days=%-5d run=%d",
                     i + 1, len(jobs), h["home"], h["lines"], h["bad_lines"], h["n_sensors"], h["active_days"], h["longest_active_run"])
    H = pd.DataFrame(homes).sort_values("home")
    dupl = H.home[H.home.duplicated(keep=False)].unique()
    if len(dupl):
        log.warning("%s: home id appears in several files (kept separate, check!): %s", tag, ", ".join(map(str, dupl)))
    H["duplicate_home_id"] = H.home.isin(dupl)
    return H, pd.DataFrame(sensors), pd.DataFrame(days), pd.DataFrame(gaps), other


def list_zip(zip_path):
    if os.path.isdir(zip_path):
        rows = [dict(name=os.path.relpath(os.path.join(d, f), zip_path), bytes=os.path.getsize(os.path.join(d, f)))
                for d, _, fs in os.walk(zip_path) for f in sorted(fs) if not _skip(os.path.join(d, f))]
        return pd.DataFrame(rows, columns=["name", "bytes"])
    with zipfile.ZipFile(zip_path) as z:
        return pd.DataFrame([dict(name=i.filename, bytes=i.file_size) for i in z.infolist() if not i.is_dir()])
