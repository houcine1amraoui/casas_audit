"""Synthetic CASAS-like homes for the full pipeline (inventory -> select -> audit), folder layout:
OUT/casas/{data,labeled_data,floorplans}/ + OUT/casas/zenodo_15708568.json (resident table).

6 single-resident homes with different routines; planted defects:
  sh01: bedroom bed sensor stuck ON for ~5 h every night from day 30  -> stuck
  sh02: kitchen stove sensor dead for days 20-40                      -> long_dropout
  sh03: away 3 days (door, then silence) + 1-day system outage        -> silences, one after a door
  sh04: routine shift (wake 2 h later) from day 45                     -> within-home drift
        + OfficeAChair installed on day 40                            -> partial_presence
  sh05: labels, marker style (Sleep="begin"/"end")
  sh06: labels, event style (every event labelled); also a duplicate of itself as sh99 (same data)
"""
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

root = Path(sys.argv[1]) / "casas"
for d in ("data", "labeled_data", "floorplans"):
    (root / d).mkdir(parents=True, exist_ok=True)
DAYS = 70
SENS = {"Bedroom": ["BedroomAArea", "BedroomABed", "BedroomADoor"], "Bathroom": ["BathroomAArea", "BathroomASink", "BathroomAToilet"],
        "Kitchen": ["KitchenAArea", "KitchenASink", "KitchenAStove", "KitchenARefrigerator"], "LivingRoom": ["LivingRoomAArea", "LivingRoomAChair"],
        "Transit": ["HallwayA", "MainEntryway"], "Office": ["OfficeAArea", "OfficeAChair"]}


def home_events(hid, seed, wake, sleep, kitchen_w, office_w, defects):
    rng = random.Random(seed)
    t0 = datetime(2019, 3, 1)
    ev, lab = [], []
    for d in range(DAYS):
        day = t0 + timedelta(days=d)
        if "away" in defects and 25 <= d < 28:
            if d == 25:
                ev.append((day + timedelta(hours=9), "MainDoor", "OPEN")); ev.append((day + timedelta(hours=9, seconds=20), "MainDoor", "CLOSE"))
            continue
        if "outage" in defects and d == 50:
            continue
        w = wake + (2 if ("shift" in defects and d >= 45) else 0) + rng.uniform(-.5, .5)
        s = sleep + rng.uniform(-.5, .5)
        # night: bed + occasional bathroom
        t = day
        while t < day + timedelta(hours=w):
            room = "Bathroom" if rng.random() < .05 else "Bedroom"
            sname = rng.choice(SENS[room][:2]) if room == "Bedroom" else rng.choice(SENS[room])
            if "stuck_bed" in defects and d >= 31 and sname == "BedroomABed" and (t - day).total_seconds() < 4.5 * 3600:
                sname = "BedroomAArea"            # a stuck-ON sensor sends nothing until it releases
            ev.append((t, sname, "ON")); lab.append((t, "Sleep"))
            t += timedelta(seconds=rng.uniform(1, 8)); ev.append((t, sname, "OFF"))
            t += timedelta(minutes=rng.expovariate(1 / 25))
        t = day + timedelta(hours=w)
        end = day + timedelta(hours=s)
        while t < end:
            h = (t - day).total_seconds() / 3600
            p = {"Kitchen": kitchen_w * (2 if 7 <= h < 9 or 17 <= h < 19 else 1), "LivingRoom": 1.5 if h > 18 else .7,
                 "Bathroom": .5, "Office": office_w, "Transit": .3, "Bedroom": .2}
            room = rng.choices(list(p), weights=list(p.values()))[0]
            sname = rng.choice(SENS[room])
            if "dead_stove" in defects and sname == "KitchenAStove" and 20 <= d < 40:
                sname = "KitchenAArea"
            if "shift" in defects and sname == "OfficeAChair" and d < 40:    # chair sensor installed on day 40
                sname = "OfficeAArea"
            ev.append((t, sname, "ON")); lab.append((t, {"Kitchen": "Cook", "LivingRoom": "Watch_TV", "Bathroom": "Toilet",
                                                       "Office": "Work_At_Desk", "Transit": "Walk", "Bedroom": "Dress"}[room]))
            dur = rng.uniform(1, 10)
            if "stuck_bed" in defects and sname == "BedroomABed":
                dur = rng.uniform(1, 10)
            t2 = t + timedelta(seconds=dur); ev.append((t2, sname, "OFF"))
            if rng.random() < .02:
                ev.append((t2, "MainDoor", "OPEN")); ev.append((t2 + timedelta(seconds=15), "MainDoor", "CLOSE"))
            t = t2 + timedelta(minutes=rng.expovariate(1 / 3))
        if "stuck_bed" in defects and d >= 30:
            st = day + timedelta(hours=23, minutes=30)
            ev.append((st, "BedroomABed", "ON")); ev.append((st + timedelta(hours=5), "BedroomABed", "OFF"))
            ev = [e for e in ev if not (e[1] == "BedroomABed" and st < e[0] < st + timedelta(hours=5))]
    ev.sort()
    return ev, lab


def fmt(ev, labels=None, style=None):
    """style None: no labels; 'event': every event carries its activity; 'marker': Act="begin"/"end" on run edges."""
    lab = dict(labels or [])
    acts, cur = [], None
    for t, s, m in ev:                     # OFF / door events inherit the activity of the last labelled ON
        cur = lab.get(t, cur) if m == "ON" else cur
        acts.append(cur)
    lines = []
    for i, (t, s, m) in enumerate(ev):
        base = f"{t.strftime('%Y-%m-%d %H:%M:%S.%f')},{s},{m}"
        a = acts[i]
        if style == "event" and a:
            base += "," + a
        elif style == "marker" and a:
            first = i == 0 or acts[i - 1] != a
            last = i == len(ev) - 1 or acts[i + 1] != a
            if first and not last:
                base += f',{a}="begin"'
            elif last and not first:
                base += f',{a}="end"'
        lines.append(base)
    return "\n".join(lines) + "\n"


cfgs = {"sh01": (7, 22, 1.0, .2, ["stuck_bed"]), "sh02": (6, 21, 2.0, .1, ["dead_stove"]), "sh03": (8, 23, .7, 1.5, ["away", "outage"]),
        "sh04": (7.5, 22.5, 1.2, .6, ["shift"]), "sh05": (9, 24, .8, .3, []), "sh06": (6.5, 21.5, 1.5, 1.0, [])}
for i, (h, (wk, sl, kw, ow, df)) in enumerate(cfgs.items()):
    ev, lab = home_events(h, i, wk, sl, kw, ow, df)
    (root / "data" / f"{h}.csv").write_text(fmt(ev))
    if h == "sh06":
        (root / "data" / "sh99.csv").write_text(fmt(ev))
        (root / "labeled_data" / f"{h}.csv").write_text(fmt(ev, lab, "event"))
    if h == "sh05":
        (root / "labeled_data" / f"{h}.csv").write_text(fmt(ev, lab, "marker"))
(root / "floorplans" / "sh01.png").write_bytes(b"png")
desc = ('<table><tr><th>Home(s)</th><th>#Residents</th></tr><tr><td>sh01-sh06, sh99</td><td>1</td>'
        '<td>sh: synthetic</td></tr></table>')
(root / "zenodo_15708568.json").write_text(json.dumps({"metadata": {"description": desc}}))
print("ok", root)
