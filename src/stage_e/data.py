"""CASAS Stage E data: one model-ready dataset per home from the audit's step cache (cache/casas/steps/<home>.npz).

Channels = the home's usable streams (audit, clean window). Value per step = log1p(activations in the step).
A step is usable only on a clean day; windows never touch a non-clean day. Chronological split of clean days:
train / val / test (fractions). Normalisation is fitted on train only. Other homes' days are mapped onto this home's
channels by sensor name (`align`) so this home's model can score them (cross-user deviation).
"""
import numpy as np
import pandas as pd

KEYS_DOC = "channel key = (room type, spot, is door); same room letter preferred"


def load_steps(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def build(home, Z, cfg):
    e = cfg["stage_e"]
    res = int(Z["step_minutes"])
    spd = 1440 // res
    days = pd.to_datetime(Z["days"].astype("datetime64[D]"))
    clean = Z["clean"].astype(bool)
    nd = len(days)
    X = np.log1p(Z["X"].astype(np.float32))
    good_step = np.repeat(clean, spd)
    obs = np.repeat(good_step[:, None], X.shape[1], 1)
    X = np.where(obs, X, np.nan)
    # chronological split over clean days
    cd = np.flatnonzero(clean)
    f = e["split"]
    n = len(cd)
    n_tr, n_va = int(round(f["train"] * n)), int(round(f["val"] * n))
    lab = np.full(nd, "none", object)
    lab[cd[:n_tr]], lab[cd[n_tr:n_tr + n_va]], lab[cd[n_tr + n_va:]] = "train", "val", "test"
    split = np.repeat(lab, spd)
    info = {s: (str(days[cd_s].min().date()), str(days[cd_s].max().date()), int(len(cd_s)))
            for s in ("train", "val", "test") for cd_s in [np.flatnonzero(lab == s)] if len(cd_s)}
    tr = split == "train"
    mu = np.nanmean(X[tr], 0)
    sd = np.maximum(np.nan_to_num(np.nanstd(X[tr], 0)), e["std_floor"])
    times = days[0] + pd.to_timedelta(np.arange(nd * spd) * res, unit="min")
    w, h = int(e["window_steps"]), int(e["horizon_steps"])
    ds = dict(name=home, home=home, X=X, obs=obs, mu=mu.astype(np.float32), sd=sd.astype(np.float32), split=split, split_info=info,
              names=[str(s) for s in Z["streams"]], room=np.array(Z["room"]).astype(str), group=np.array(Z["group"]).astype(str),
              room_type=np.array(Z["room_type"]).astype(str), spot=np.array(Z["spot"]).astype(str), is_door=np.array(Z["is_door"]).astype(bool),
              family=np.array(["events"] * X.shape[1]), kind=np.array(["motion"] * X.shape[1]),
              times=times, good_step=good_step, spd=spd, w=w, h=h, days=days, clean=clean, day_split=lab)
    ds["Z"] = standardize(ds, X, obs)
    ds["cov"] = covariates(times)
    ds["widx"] = window_index(obs, good_step, split, w, h)
    return ds


def standardize(ds, X, obs):
    Zs = np.clip((X - ds["mu"]) / ds["sd"], -10, 10)
    return np.where(obs, Zs, 0.0).astype(np.float32)


def covariates(times):
    hh = (times.hour.values * 60 + times.minute.values) / 1440.0
    return np.stack([np.sin(2 * np.pi * hh), np.cos(2 * np.pi * hh), (times.dayofweek.values >= 5).astype(float)], 1).astype(np.float32)


def window_index(obs, good_step, split, w, h):
    """Target t, inputs t-h-w+1..t-h; every step from t-h-w+1 to t on a clean day."""
    T = len(good_step)
    cg = np.r_[0, np.cumsum(good_step.astype(int))]
    t = np.arange(w + h - 1, T)
    ok = (cg[t + 1] - cg[t - h - w + 1]) == (h + w)
    return {s: t[ok & (split[t] == s)] for s in ("train", "val", "test")}


def make_arrays(Z, obs, cov, idx, w, h=1):
    ii = idx[:, None] + np.arange(-h - w + 1, -h + 1)[None, :]
    ir = idx[:, None] + np.arange(-w + 1, 1)[None, :]
    return dict(x=Z[ii], m=obs[ii].astype(np.float32), c=cov[np.c_[ii, idx]], y=Z[idx], ym=obs[idx].astype(np.float32), t=idx,
                xr=Z[ir], mr=obs[ir].astype(np.float32), cr=cov[np.c_[ir, idx]])


def align_map(target, source):
    """For each target channel, the source channel index (or -1): same (room type, spot, door) key, same room letter first."""
    out, how = [], []
    for j in range(len(target["names"])):
        key = (target["room_type"][j], target["spot"][j], bool(target["is_door"][j]))
        cand = [k for k in range(len(source["names"]))
                if (source["room_type"][k], source["spot"][k], bool(source["is_door"][k])) == key]
        if not cand:
            out.append(-1); how.append("none"); continue
        same = [k for k in cand if source["room"][k] == target["room"][j]]
        out.append((same or sorted(cand, key=lambda k: source["room"][k]))[0]); how.append("same_room" if same else "same_type")
    return np.array(out), how


def impostor_arrays(target, source, cfg, rng):
    """Up to n_days test days of `source`, mapped onto `target`'s channels and normalised with target's train stats.
    Unmatched channels are unobserved (mask 0, not scored). Returns arrays + matched share, or None."""
    m, _ = align_map(target, source)
    if (m >= 0).mean() < cfg["stage_e"]["impostor"]["min_matched_share"]:
        return None, float((m >= 0).mean())
    T = source["X"].shape[0]
    X = np.full((T, len(m)), np.nan, np.float32)
    obs = np.zeros((T, len(m)), bool)
    for j, k in enumerate(m):
        if k >= 0:
            X[:, j], obs[:, j] = source["X"][:, k], source["obs"][:, k]
    Zs = standardize(target, X, obs)
    idx = source["widx"]["test"]
    if not len(idx):
        return None, float((m >= 0).mean())
    tdays = source["times"][idx].normalize().values.astype("datetime64[D]").astype(np.int64)   # day numbers
    ud = np.unique(tdays)
    nmax = cfg["stage_e"]["impostor"]["days_per_home"]
    pick = rng.choice(ud, size=min(nmax, len(ud)), replace=False)
    idx = idx[np.isin(tdays, pick)]
    if not len(idx):
        return None, float((m >= 0).mean())
    a = make_arrays(Zs, obs, source["cov"], idx, target["w"], target["h"])
    a["Zsrc"] = Zs
    a["day"] = source["times"][idx].normalize().values
    a["hour"] = source["times"][idx].hour.values
    return a, float((m >= 0).mean())
