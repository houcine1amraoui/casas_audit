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


def stream_stats(ev, on, streams, day_sel, active, cfg, home, room, roomid, smap):
    """Per-stream statistics over the days in day_sel (bool array over the full window)."""
    a = cfg["audit"]
    nd = len(day_sel)
    act_sel = active & day_sel
    rows = []
    for s in streams:
        e = ev[(ev.stream == s) & day_sel[ev.day.values]]
        if e.empty:
            continue
        o = e[e.act == 1]
        per_day = np.bincount(o.day, minlength=nd)[:nd]
        med = float(np.median(per_day[act_sel])) if act_sel.any() else 0.0
        dropout = act_sel & (per_day == 0) & (med >= a["dropout_min_median"])
        run = best = 0
        for i in np.flatnonzero(day_sel):
            run = run + 1 if dropout[i] else 0
            best = max(best, run)
        tt, aa = e.t.values, e.act.values
        on_idx = np.flatnonzero(aa == 1)
        nxt = on_idx + 1
        ok = nxt < len(aa)
        paired = np.zeros(len(on_idx), bool)
        paired[ok] = aa[nxt[ok]] == 0
        dur = tt[nxt[paired]] - tt[on_idx[paired]] if paired.any() else np.array([])
        stuck = dur >= a["stuck_hours"] * 3600
        sel_days = np.flatnonzero(day_sel)
        fired = sel_days[per_day[sel_days] > 0]
        late_start = int(fired[0] - sel_days[0]) if len(fired) else 0          # days from window start to first activation
        early_stop = int(sel_days[-1] - fired[-1]) if len(fired) else 0
        rows.append(dict(home=home, stream=s, late_start_days=late_start, early_stop_days=early_stop, base=e.base.iat[0], room_type=e.room_type.iat[0], group=e.group.iat[0],
                         room=roomid.get(e.base.iat[0], "Other"), is_door=s.endswith("#door"),
                         activations=int(len(o)), releases=int((e.act == 0).sum()),
                         unpaired_on_share=float(1 - paired.mean()) if len(on_idx) else np.nan,
                         median_on_s=float(np.median(dur)) if len(dur) else np.nan,
                         p99_on_s=float(np.quantile(dur, .99)) if len(dur) else np.nan,
                         stuck_cycles=int(stuck.sum()), stuck_time_share=float(dur[stuck].sum() / max(day_sel.sum() * 86400, 1)),
                         median_daily=med, active_day_share=float((per_day[act_sel] > 0).mean()) if act_sel.any() else 0.0,
                         dropout_days=int(dropout.sum()), longest_dropout_run=int(best),
                         max_daily=int(per_day[day_sel].max()) if day_sel.any() else 0))
    S = pd.DataFrame(rows)
    # binary (ON/OFF or OPEN/CLOSE) sensors of the map that never fire on these days; numeric sensors are not streams
    binary = set(smap.sensor[smap.binary]) if "binary" in smap else set(smap.sensor)
    silent = sorted(binary - set(S.base if len(S) else []))
    if silent:
        S = pd.concat([S, pd.DataFrame([dict(home=home, stream=x, base=x, room_type=room.get(x, "Other"), group=group_of(room.get(x, "Other")),
                                             room=roomid.get(x, "Other"), is_door=False, activations=0, median_daily=0.0,
                                             active_day_share=0.0) for x in silent])], ignore_index=True)
    c = a["classify"]
    cls = []
    for r in S.itertuples():
        if r.activations == 0:
            k = "silent_in_window"
        elif max(getattr(r, "late_start_days", 0) or 0, getattr(r, "early_stop_days", 0) or 0) > c["partial_days"]:
            k = "partial_presence"      # installed, renamed or removed inside the window: a step change, not behaviour
        elif r.median_daily < c["rare_median_daily"]:
            k = "rare"
        elif (not r.is_door) and (r.stuck_time_share or 0) > c["stuck_time_share"]:   # a door may stay open: not a fault
            k = "stuck"
        elif (r.longest_dropout_run or 0) >= c["dropout_run_days"]:
            k = "long_dropout"
        elif (r.dropout_days or 0) > c["dropout_share"] * max(act_sel.sum(), 1):
            k = "intermittent"
        elif (r.max_daily or 0) > c["chatter_daily"]:
            k = "chatter"
        else:
            k = "ok"
        cls.append(k)
    S["cls"] = cls
    S["usable"] = S.cls.isin(["ok", "rare"])
    return S


def clean_run(Xall, streams, active, cfg):
    """Longest run of days where >= min_alive_share of the core motion streams fire (core: median >= dropout_min_median on
    active days of the full window); stretches of <= gap_tolerance bad days are bridged but not used. Capped at max_days
    clean days. Returns (bool mask over the window of the clean days used, info dict)."""
    a, cw = cfg["audit"], cfg["audit"]["clean"]
    nd = len(active)
    med = np.median(Xall[active], 0) if active.any() else np.zeros(len(streams))
    core = np.array([(not s.endswith("#door")) and m >= a["dropout_min_median"] for s, m in zip(streams, med)])
    if core.sum() == 0:
        return active.copy(), dict(core_streams=0, clean_days=int(active.sum()), run_start=0, run_end=nd)
    alive = (Xall[:, core] > 0).mean(1)
    good = active & (alive >= cw["min_alive_share"])
    best, i = (0, 0, 0), 0
    while i < nd:
        if not good[i]:
            i += 1
            continue
        j, last_good, bad = i, i, 0
        while j < nd:
            if good[j]:
                last_good, bad = j, 0
            else:
                bad += 1
                if bad > cw["gap_tolerance"]:
                    break
            j += 1
        n_good = int(good[i:last_good + 1].sum())
        if n_good > best[0]:
            best = (n_good, i, last_good + 1)
        i = last_good + 1
    _, s0, s1 = best
    mask = np.zeros(nd, bool)
    mask[s0:s1] = good[s0:s1]
    if cw["max_days"] and mask.sum() > cw["max_days"]:
        keep = np.flatnonzero(mask)[: cw["max_days"]]
        mask[:] = False
        mask[keep] = True
    return mask, dict(core_streams=int(core.sum()), clean_days=int(mask.sum()), run_start=int(s0), run_end=int(s1),
                      good_days_full=int(good.sum()))


def audit_home(home, path, win_start, win_end, smap, cfg):
    """smap: DataFrame of this home's sensors (sensor, room_type, room, spot[, binary]). Integrity and sensor classes are
    reported for the whole stable window AND for the clean sub-window; features come from the clean sub-window only."""
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

    day_act = np.bincount(on.day, minlength=nd)[:nd]
    active = day_act >= cfg["inventory"]["active_day_min_events"]
    gaps = []
    if len(ev):
        tt = ev.t.values
        dt = np.diff(tt)
        for i in np.flatnonzero(dt >= a["silence_hours"] * 3600):
            prev_rows = ev.iloc[max(0, i - 5):i + 1]
            door_before = bool(prev_rows.stream.str.endswith("#door").any() | prev_rows.room_type.eq("Perimeter").any())
            gaps.append(dict(home=home, start=pd.Timestamp(tt[i], unit="s"), hours=dt[i] / 3600,
                             last_sensor=ev.stream.iat[i], door_or_entry_before=door_before))
    allid = {s: i for i, s in enumerate(streams)}
    Xall = np.zeros((nd, len(streams)), np.float32)
    if len(on):
        np.add.at(Xall, (on.day.values, on.stream.map(allid).values), 1)

    S_full = stream_stats(ev, on, streams, np.ones(nd, bool), active, cfg, home, room, roomid, smap)
    clean, cinfo = clean_run(Xall, streams, active, cfg)
    S = stream_stats(ev, on, streams, clean, active, cfg, home, room, roomid, smap)
    S = S.merge(S_full[["stream", "cls"]].rename(columns={"cls": "cls_full_window"}), on="stream", how="left")

    # ---------------- daily features (clean days only)
    nb = a["n_bins"]
    per = 24 // nb
    use = S[S.usable & (S.activations > 0)].stream.tolist()
    o = on[on.stream.isin(use) & clean[on.day.values]]
    sid = {s: i for i, s in enumerate(use)}
    Xs = np.zeros((nd, len(use) * nb), np.float32)
    if len(o):
        np.add.at(Xs, (o.day.values, o.stream.map(sid).values * nb + o.hour.values // per), 1)
    Xg = np.zeros((nd, len(GROUPS) * nb), np.float32)
    gid = {g: i for i, g in enumerate(GROUPS)}
    if len(o):
        np.add.at(Xg, (o.day.values, o.group.map(gid).values * nb + o.hour.values // per), 1)
    Xh = np.zeros((nd, 24), np.float32)
    if len(o):
        np.add.at(Xh, (o.day.values, o.hour.values), 1)
    tot = Xh.sum(1, keepdims=True)
    res_min = a["step_minutes"]
    spd = 1440 // res_min
    oa = on[on.stream.isin(use)]
    Xstep = np.zeros((nd * spd, len(use)), np.float32)
    if len(oa):
        st = (oa.day.values * spd + ((oa.t.values - start_s) % 86400 // (res_min * 60)).astype(int))
        np.add.at(Xstep, (st, oa.stream.map(sid).values), 1)
    meta = S.set_index("stream").loc[use] if use else S.iloc[:0].set_index("stream")
    sp = dict(zip(smap.sensor, smap.spot))
    D = pd.DataFrame(dict(home=home, day=days, activations=day_act, active=active, clean=clean, usable_activations=tot[:, 0].astype(int)))
    return dict(home=home, days=D, sensors=S, sensors_full=S_full, gaps=pd.DataFrame(gaps), ignored=dict(ignored), backwards=back,
                n_events=len(ev), clean_info=cinfo, Xs=np.log1p(Xs), Xs_names=[f"{s}@{b * per:02d}h" for s in use for b in range(nb)],
                Xg_share=Xg / np.maximum(tot, 1), Xh_share=Xh / np.maximum(tot, 1), log_total=np.log1p(tot[:, 0]),
                active_mask=clean, day_index=days, streams=use, all_streams=streams, Xall=Xall, active_full=active,
                Xstep=Xstep, step_minutes=res_min, stream_room_type=meta.room_type.astype(str).tolist() if use else [],
                stream_room=meta.room.astype(str).tolist() if use else [], stream_group=meta.group.astype(str).tolist() if use else [],
                stream_spot=[sp.get(s.replace("#door", ""), s) for s in use], stream_is_door=[s.endswith("#door") for s in use])
