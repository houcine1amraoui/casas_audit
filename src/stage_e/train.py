"""Uniform fit / error interface for numpy baselines and torch models."""
import copy
import time

import numpy as np
import pandas as pd
import torch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _t(a):
    return torch.as_tensor(a, device=DEV)


KEYS = ("x", "m", "c", "y", "ym", "xr", "mr", "cr")


def _loss(model, b):
    if model.kind == "forecast":
        p = model(b["x"], b["m"], b["c"])
        return ((p - b["y"]) ** 2 * b["ym"]).sum() / b["ym"].sum().clamp_min(1)
    p = model(b["xr"], b["mr"], b["cr"])
    return ((p - b["xr"]) ** 2 * b["mr"]).sum() / b["mr"].sum().clamp_min(1)


def _batches(a, bs, shuffle, rng):
    n = len(a["y"])
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        yield {k: _t(a[k][j]) for k in KEYS}


def fit(model, tr, va, cfg, seed):
    """Returns info dict (epochs, best val loss, train seconds, n_params, history)."""
    t0 = time.time()
    if not isinstance(model, torch.nn.Module):
        info = model.fit(tr, va)
        info.update(train_s=time.time() - t0, n_params=getattr(model, "n_params", 0), epochs=0)
        return info
    hp = cfg["stage_e"]["train"]
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(hp["max_epochs"]):
        model.train()
        tl, nb = 0.0, 0
        for b in _batches(tr, hp["batch_size"], True, rng):
            opt.zero_grad()
            loss = _loss(model, b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
            opt.step()
            tl += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            vl = [float(_loss(model, b)) * len(b["y"]) for b in _batches(va, 1024, False, rng)]
        vl = sum(vl) / len(va["y"])
        hist.append((tl / max(nb, 1), vl))
        if vl < best - 1e-5:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return dict(epochs=len(hist), best_val_loss=best, train_s=time.time() - t0,
                n_params=sum(p.numel() for p in model.parameters()), history=hist)


def errors(model, a, ds):
    """Per-window, per-channel absolute error (NaN where the channel is unobserved) and the masked squared-error
    sum/count (for MSE). Forecasters: error at the target step. Autoencoder: mean over the window's observed steps."""
    if not isinstance(model, torch.nn.Module):
        p = model.predict(a, ds)
        d = p - a["y"]
        E = np.where(a["ym"] > 0, np.abs(d), np.nan)
        return E, float(((d ** 2) * a["ym"]).sum()), float(a["ym"].sum())
    outs = []
    with torch.no_grad():
        for i in range(0, len(a["y"]), 2048):
            b = {k: _t(a[k][i:i + 2048]) for k in KEYS}
            outs.append((model(b["x"], b["m"], b["c"]) if model.kind == "forecast" else model(b["xr"], b["mr"], b["cr"])).cpu().numpy())
    p = np.concatenate(outs) if outs else np.zeros_like(a["y"])
    if model.kind == "forecast":
        d = p - a["y"]
        E = np.where(a["ym"] > 0, np.abs(d), np.nan)
        return E, float(((d ** 2) * a["ym"]).sum()), float(a["ym"].sum())
    d = p - a["xr"]
    cnt = a["mr"].sum(1)
    with np.errstate(all="ignore"):
        E = np.where(cnt > 0, (np.abs(d) * a["mr"]).sum(1) / cnt, np.nan)
    return E, float(((d ** 2) * a["mr"]).sum()), float(a["mr"].sum())


def score(E, med, iqr):
    """GDN-style: per-channel robust normalisation (val median / IQR), max over observed channels."""
    with np.errstate(all="ignore"):
        Zs = (E - med) / iqr
        s = np.nanmax(Zs, 1)
    return np.where(np.isnan(s), -np.inf, s), Zs


def aggregate(Zs, how):
    """Channel aggregation of normalised errors per window: max | mean | topK (mean of the K largest)."""
    with np.errstate(all="ignore"):
        if how == "max":
            s = np.nanmax(Zs, 1)
        elif how == "mean":
            s = np.nanmean(Zs, 1)
        elif how.startswith("top"):
            k = int(how[3:])
            srt = -np.sort(-np.where(np.isnan(Zs), -np.inf, Zs), 1)[:, :k]
            srt = np.where(np.isinf(srt), np.nan, srt)
            s = np.nanmean(srt, 1)
        else:
            raise ValueError(how)
    return s


def smooth(s, t, k):
    """Causal rolling mean of the last k window scores, restarted wherever target steps are not consecutive."""
    s = pd.Series(np.where(np.isfinite(s), s, np.nan))
    if k <= 1:
        return s.fillna(s.min() if s.notna().any() else 0).values
    blk = np.r_[0, np.cumsum(np.diff(t) != 1)]
    out = s.groupby(blk).transform(lambda x: x.rolling(k, min_periods=1).mean())
    return out.fillna(out.min() if out.notna().any() else 0).values
