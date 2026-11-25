"""Activity labels in the labelled CASAS files: two styles, possibly mixed in one file.

  marker style: `... Sleep="begin"` / `... Sleep="end"` on the event lines where an activity starts / ends
  event style:  `... Toilet` on every event line that belongs to the activity
Segments: markers give [begin, end]; event-style labels are joined while the same label repeats with gaps
<= join_gap_s. Output: segments (activity, start, end) in epoch seconds, and per-event labels for sanity checks.
"""
import re
from datetime import date

import pandas as pd

from .inventory import open_source
from .parse import parse

EPOCH0 = date(1970, 1, 1).toordinal() * 86400
MARK = re.compile(r'^([A-Za-z_]+)\s*=\s*"?(begin|end)"?$', re.I)


def read_segments(path, join_gap_s=1800):
    z, fh = open_source("file", path, path)
    segs, open_m = [], {}
    cur = None                                   # event-style segment being built: [act, start, last]
    ev = []
    try:
        for raw in fh:
            r = parse(raw.decode("utf-8", "replace").strip())
            if r is None:
                continue
            _, ts, s, msg, extra, _ = r
            ts -= EPOCH0
            lab = " ".join(extra).strip()
            act = None
            if lab:
                m = MARK.match(lab)
                if m:
                    act, what = m.group(1), m.group(2).lower()
                    if what == "begin":
                        open_m[act] = ts
                    elif act in open_m:
                        segs.append((act, open_m.pop(act), ts, "marker"))
                else:
                    act = lab.split()[0]
                    if cur and cur[0] == act and ts - cur[2] <= join_gap_s:
                        cur[2] = ts
                    else:
                        if cur:
                            segs.append((cur[0], cur[1], cur[2], "event"))
                        cur = [act, ts, ts]
            ev.append((ts, s, msg))
        if cur:
            segs.append((cur[0], cur[1], cur[2], "event"))
    finally:
        fh.close()
    S = pd.DataFrame(segs, columns=["activity", "start", "end", "style"])
    S["minutes"] = (S.end - S.start) / 60
    E = pd.DataFrame(ev, columns=["t", "sensor", "msg"])
    return S, E, len(open_m)


def label_room_table(S, E, room_type_of, group_of):
    """Share of motion activations inside each activity's segments by room group (do labels and rooms agree?)."""
    on = E[E.msg.str.upper().isin(["ON", "OPEN"])].sort_values("t").reset_index(drop=True)
    on["group"] = on.sensor.map(lambda s: group_of(room_type_of.get(s, "Other")))
    rows = []
    for act, g in S.groupby("activity"):
        mask = pd.Series(False, index=on.index)
        tt = on.t.values
        for r in g.itertuples():
            lo, hi = tt.searchsorted(r.start), tt.searchsorted(r.end, side="right")
            mask.iloc[lo:hi] = True
        sub = on[mask]
        if len(sub):
            sh = sub.group.value_counts(normalize=True)
            rows.append(dict(activity=act, segments=len(g), activations=len(sub), **{k: float(v) for k, v in sh.items()}))
    return pd.DataFrame(rows)
