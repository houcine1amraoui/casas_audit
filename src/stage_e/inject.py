"""Synthetic behavioural anomalies injected into the TEST split (feature space, before scaling).

Each type is injected into its own copy of the data, so per-type detection is measured separately.
A segment is kept only if it really changes the input (mean |delta z| >= min_effect on some channel);
otherwise injecting, e.g., inactivity into a period that is already inactive would create false labels.

types
  night_activity : event channels between 00:00-05:00 replaced by the same day's values 12 h later
  inactivity     : event channels set to zero during 09:00-20:00 (sensors still reporting)
  appliance_on   : one plug held at its train p99.5 level (sustained use at an arbitrary time)
  routine_shift  : all channels replaced by the values 4 h earlier (right behaviour, wrong time)
  room_swap      : one room's channels replaced by that room 12 h away; other rooms unchanged
                   (each channel stays plausible, the cross-room context does not; the structural case)
"""
import numpy as np

TYPES = ["night_activity", "inactivity", "appliance_on", "routine_shift", "room_swap"]


def _hours(times):
    return times.hour.values + times.minute.values / 60


def inject(ds, kind, cfg, seed, split="test"):
    e = cfg["stage_e"]["inject"]
    rng = np.random.default_rng(seed)
    X, obs = ds["X"].copy(), ds["obs"].copy()
    spd = ds["spd"]
    L = int(round(e["segment_hours"] * spd / 24))
    w = ds["w"] + ds.get("h", 1)
    T, N = X.shape
    hrs = _hours(ds["times"])
    test_t = set(ds["widx"][split].tolist())
    ev = np.flatnonzero(ds["family"] == "events")
    plugs = np.flatnonzero(ds["kind"] == "current")
    tr = ds["split"] == "train"
    rooms = {r: np.flatnonzero(ds["room"] == r) for r in np.unique(ds["room"])}
    rooms = {r: v for r, v in rooms.items() if len(v) >= 2}
    good = ds["good_step"]
    segs, info = [], []
    cand = rng.permutation(sorted(test_t))
    used = np.zeros(T, bool)

    def ok_range(a, b):   # source data must come from the same split (no test data copied into val anomalies)
        return 0 <= a and b <= T and good[a:b].all() and (ds["split"][a:b] == split).all()

    for s in cand:
        if len(segs) >= e["n_per_type"]:
            break
        a, b = int(s), int(s) + L
        if not all((t in test_t) for t in range(a, b)):          # whole segment = valid test targets
            continue
        if used[max(a - w - e["gap_steps"], 0):min(b + e["gap_steps"], T)].any():
            continue
        h = hrs[a]
        cols, src = None, None
        if kind == "night_activity":
            if not (0 <= h and hrs[b - 1] < 5 and h < 5) or len(ev) == 0:
                continue
            cols, src = ev, a + spd // 2
        elif kind == "inactivity":
            if not (9 <= h and hrs[b - 1] < 20) or len(ev) == 0:
                continue
            cols = ev
        elif kind == "appliance_on":
            if len(plugs) == 0:
                break
            cols = np.array([rng.choice(plugs)])
        elif kind == "routine_shift":
            cols, src = np.arange(N), a - spd // 6
        elif kind == "room_swap":
            if not rooms:
                break
            r = list(rooms)[rng.integers(len(rooms))]
            cols, src = rooms[r], a + (spd // 2) * (1 if rng.random() < .5 else -1)
        new = X[a:b][:, cols].copy()
        nob = obs[a:b][:, cols].copy()
        if src is not None:
            if not ok_range(src, src + L) or used[max(src, 0):src + L].any():
                continue
            new, nob = X[src:src + L][:, cols].copy(), obs[src:src + L][:, cols].copy()
        elif kind == "inactivity":
            new[:] = 0.0; nob[:] = True
        elif kind == "appliance_on":
            c = cols[0]
            hi = np.nanpercentile(X[tr, c], 99.5)
            new[:] = hi; nob[:] = True
        # effect check in z units (unobserved -> 0 = train mean)
        z_old = np.where(obs[a:b][:, cols], (X[a:b][:, cols] - ds["mu"][cols]) / ds["sd"][cols], 0)
        z_new = np.where(nob, (new - ds["mu"][cols]) / ds["sd"][cols], 0)
        eff = np.nanmax(np.nanmean(np.abs(z_new - z_old), 0))
        if not eff >= e["min_effect"]:
            continue
        X[a:b, cols] = np.where(nob, new, np.nan)
        obs[a:b, cols] = nob
        used[a:b] = True
        segs.append((a, b))
        info.append(dict(type=kind, start=str(ds["times"][a]), steps=L, channels=";".join(ds["names"][c] for c in cols), effect_z=float(eff)))
    return X, obs, segs, info


def labels_for(targets, segs, after):
    """1 = target step inside a segment; -1 = clean target whose inputs (or smoothed score) still see the segment,
    i.e. t in [b, b + after) with after = horizon + window - 1 + smoothing - 1 (excluded); 0 = clean."""
    lab = np.zeros(len(targets), int)
    for a, b in segs:
        inside = (targets >= a) & (targets < b)
        lab[inside] = 1
        touch = (targets >= a) & (targets < b + after) & ~inside
        lab[touch & (lab == 0)] = -1
    return lab
