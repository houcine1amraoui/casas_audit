"""Step 3, one home: integrity + sensor checks on binary event streams inside the home's stable window, and daily features.

Streams: motion = ON/OFF messages, door = OPEN/CLOSE messages (a name carrying both gives two streams, the door one
named `<name>#door`). Everything else (temperatures, 0/100, button taps) is counted and ignored.
An *activation* = ON (motion) or OPEN (door).
"""
import os
from collections import Counter, defaultdict
from datetime import date

import numpy as np
import pandas as pd

from .inventory import open_source
from .parse import parse

EPOCH0 = date(1970, 1, 1).toordinal() * 86400
GROUPS = ["Bedroom", "Bathroom", "Kitchen", "LivingRoom", "Transit", "OtherRooms"]
GROUP_OF = {"Bedroom": "Bedroom", "Bathroom": "Bathroom", "Kitchen": "Kitchen", "DiningRoom": "Kitchen",
            "LivingRoom": "LivingRoom", "Lounge": "LivingRoom", "Hallway": "Transit", "Hall": "Transit",
            "Entryway": "Transit", "Perimeter": "Transit"}


def group_of(room_type):
    return GROUP_OF.get(room_type, "OtherRooms")


def read_events(path, start_s, end_s):
    """-> DataFrame t (epoch s), stream, act (1 activation / 0 release), plus counts of ignored messages."""
    kind = "file"
    z, fh = open_source(kind, path, path)
    T, S, A = [], [], []
    ignored = Counter()
    back = 0
    prev = None
    try:
        for raw in fh:
            r = parse(raw.decode("utf-8", "replace").strip())
            if r is None:
                continue
            _, ts, s, msg, _, _ = r
            ts -= EPOCH0
            if ts < start_s or ts >= end_s:
                continue
            m = msg.upper()
            if m in ("ON", "OFF"):
                T.append(ts); S.append(s); A.append(1 if m == "ON" else 0)
            elif m in ("OPEN", "CLOSE", "CLOSED"):
                T.append(ts); S.append(s + "#door"); A.append(1 if m == "OPEN" else 0)
            else:
                ignored[msg if not msg.replace(".", "", 1).lstrip("-").isdigit() else "<number>"] += 1
            if prev is not None and ts < prev:
                back += 1
            prev = ts
    finally:
        fh.close()
    df = pd.DataFrame({"t": np.asarray(T, float), "stream": S, "act": np.asarray(A, np.int8)})
    df = df.sort_values("t", kind="mergesort").reset_index(drop=True)
    return df, ignored, back


def audit_home(home, path, win_start, win_end, smap, cfg):
    """smap: DataFrame of this home's sensors (sensor, room_type, room, spot). Returns dict of results."""
    a = cfg["audit"]
    d0 = pd.Timestamp(win_start).normalize() + pd.Timedelta(days=1)          # whole days strictly inside the window
    d1 = pd.Timestamp(win_end).normalize()                                    # exclusive
    start_s, end_s = d0.value / 1e9, d1.value / 1e9
    ev, ignored, back = read_events(path, start_s, end_s)
    days = pd.date_range(d0, d1 - pd.Timedelta(days=1))
    nd = len(days)
    room = dict(zip(smap.sensor, smap.room_type))
    roomid = dict(zip(smap.sensor, smap.room))
    ev["base"] = ev.stream.str.replace("#door", "", regex=False)
    ev["room_type"] = ev.base.map(room).fillna("Other")
    ev.loc[ev.stream.str.endswith("#door"), "room_type"] = "Perimeter"
    ev["group"] = ev.room_type.map(group_of)
    ev["day"] = ((ev.t - start_s) // 86400).astype(int)
    ev["hour"] = ((ev.t - start_s) % 86400 // 3600).astype(int)
    on = ev[ev.act == 1]
    streams = sorted(ev.stream.unique())

    # ---------------- home-level days
    day_act = np.bincount(on.day, minlength=nd)[:nd]
    active = day_act >= cfg["inventory"]["active_day_min_events"]
    # silent gaps >= gap_h inside the window, and what preceded them (door -> likely absence)
    gaps = []
    if len(ev):
        tt = ev.t.values
        dt = np.diff(tt)
        for i in np.flatnonzero(dt >= a["silence_hours"] * 3600):
            prev_rows = ev.iloc[max(0, i - 5):i + 1]
            door_before = bool(prev_rows.stream.str.endswith("#door").any() | prev_rows.room_type.eq("Perimeter").any())
            gaps.append(dict(home=home, start=pd.Timestamp(tt[i], unit="s"), hours=dt[i] / 3600,
                             last_sensor=ev.stream.iat[i], door_or_entry_before=door_before))
    # ---------------- per stream
    rows = []
    for s in streams:
        e = ev[ev.stream == s]
        o = e[e.act == 1]
        per_day = np.bincount(o.day, minlength=nd)[:nd]
        med = float(np.median(per_day[active])) if active.any() else 0.0
        dropout = active & (per_day == 0) & (med >= a["dropout_min_median"])
        run = best = 0
        for x in dropout:
            run = run + 1 if x else 0
            best = max(best, run)
        # ON -> next OFF durations of the same stream
        tt, aa = e.t.values, e.act.values
        on_idx = np.flatnonzero(aa == 1)
        nxt = on_idx + 1
        ok = nxt < len(aa)
        paired = np.zeros(len(on_idx), bool)
        paired[ok] = aa[nxt[ok]] == 0
        dur = tt[nxt[paired]] - tt[on_idx[paired]] if paired.any() else np.array([])
        stuck = dur >= a["stuck_hours"] * 3600
        rows.append(dict(home=home, stream=s, base=e.base.iat[0], room_type=e.room_type.iat[0], group=e.group.iat[0],
                         room=roomid.get(e.base.iat[0], "Other"), is_door=s.endswith("#door"),
                         activations=int(len(o)), releases=int((e.act == 0).sum()),
                         unpaired_on_share=float(1 - paired.mean()) if len(on_idx) else np.nan,
                         median_on_s=float(np.median(dur)) if len(dur) else np.nan,
                         p99_on_s=float(np.quantile(dur, .99)) if len(dur) else np.nan,
                         stuck_cycles=int(stuck.sum()), stuck_time_share=float(dur[stuck].sum() / max(nd * 86400, 1)),
                         median_daily=med, active_day_share=float((per_day[active] > 0).mean()) if active.any() else 0.0,
                         dropout_days=int(dropout.sum()), longest_dropout_run=int(best),
                         max_daily=int(per_day.max()) if nd else 0))
    S = pd.DataFrame(rows)
    # sensors from the map that never fire inside the window
    silent = sorted(set(smap.sensor) - set(ev.base))
    for s in silent:
        S = pd.concat([S, pd.DataFrame([dict(home=home, stream=s, base=s, room_type=room.get(s, "Other"),
                                             group=group_of(room.get(s, "Other")), room=roomid.get(s, "Other"),
                                             is_door=False, activations=0, median_daily=0.0, active_day_share=0.0)])],
                      ignore_index=True)
    c = a["classify"]
    cls = []
    for r in S.itertuples():
        if r.activations == 0:
            k = "silent_in_window"
        elif r.median_daily < c["rare_median_daily"]:
            k = "rare"
        elif (getattr(r, "stuck_time_share", 0) or 0) > c["stuck_time_share"]:
            k = "stuck"
        elif (getattr(r, "longest_dropout_run", 0) or 0) >= c["dropout_run_days"]:
            k = "long_dropout"
        elif (getattr(r, "dropout_days", 0) or 0) > c["dropout_share"] * max(active.sum(), 1):
            k = "intermittent"
        elif (getattr(r, "max_daily", 0) or 0) > c["chatter_daily"]:
            k = "chatter"
        else:
            k = "ok"
        cls.append(k)
    S["cls"] = cls
    S["usable"] = S.cls.isin(["ok", "rare"])

    # ---------------- daily features
    nb = a["n_bins"]
    per = 24 // nb
    use = S[S.usable & (S.activations > 0)].stream.tolist()
    o = on[on.stream.isin(use)]
    sid = {s: i for i, s in enumerate(use)}
    Xs = np.zeros((nd, len(use) * nb), np.float32)                         # sensor x bin counts (within-home)
    if len(o):
        np.add.at(Xs, (o.day.values, o.stream.map(sid).values * nb + o.hour.values // per), 1)
    Xg = np.zeros((nd, len(GROUPS) * nb), np.float32)                      # room group x bin counts (cross-home)
    gid = {g: i for i, g in enumerate(GROUPS)}
    if len(o):
        np.add.at(Xg, (o.day.values, o.group.map(gid).values * nb + o.hour.values // per), 1)
    Xh = np.zeros((nd, 24), np.float32)                                    # hour-of-day counts (layout-free)
    if len(o):
        np.add.at(Xh, (o.day.values, o.hour.values), 1)
    tot = Xh.sum(1, keepdims=True)
    allid = {s: i for i, s in enumerate(streams)}                          # every stream, for the figures
    Xall = np.zeros((nd, len(streams)), np.float32)
    if len(on):
        np.add.at(Xall, (on.day.values, on.stream.map(allid).values), 1)
    D = pd.DataFrame(dict(home=home, day=days, activations=day_act, active=active, usable_activations=tot[:, 0].astype(int)))
    return dict(home=home, days=D, sensors=S, gaps=pd.DataFrame(gaps), ignored=dict(ignored), backwards=back,
                n_events=len(ev), Xs=np.log1p(Xs), Xs_names=[f"{s}@{b * per:02d}h" for s in use for b in range(nb)],
                Xg_share=Xg / np.maximum(tot, 1), Xh_share=Xh / np.maximum(tot, 1), log_total=np.log1p(tot[:, 0]),
                active_mask=active, day_index=days, streams=use, all_streams=streams, Xall=Xall)
