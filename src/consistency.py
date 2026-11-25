"""Within-home consistency and cross-home distinguishability on daily features."""
import numpy as np
import pandas as pd
from scipy import stats as sps


def masked_rms(A, B):
    """Pairwise RMS distance between rows of A and B (no missing values here)."""
    return np.sqrt(np.maximum(((A[:, None, :] - B[None, :, :]) ** 2).mean(2), 0))


def zfit(X, idx, floor):
    mu = X[idx].mean(0)
    sd = np.maximum(X[idx].std(0), floor)
    return (X - mu) / sd


def within_home(X, days, ref_n, late_n, n_perm, seed, floor):
    """Early (first ref_n good days) vs late (last late_n good days). X: good days only, in time order."""
    n = len(X)
    if n < ref_n + late_n or X.shape[1] == 0:
        return dict(status="too few days")
    Z = zfit(X, np.arange(ref_n), floor)
    ref, late = np.arange(ref_n), np.arange(n - late_n, n)
    sel = np.r_[ref, late]
    lab = np.r_[np.zeros(ref_n, int), np.ones(late_n, int)]
    D = masked_rms(Z[sel], Z[sel])

    def stat(l):
        a, b = l == 0, l == 1
        cross = D[np.ix_(a, b)].mean()
        within = np.r_[D[np.ix_(a, a)][np.triu_indices(a.sum(), 1)], D[np.ix_(b, b)][np.triu_indices(b.sum(), 1)]].mean()
        return cross - within, cross, within

    rng = np.random.default_rng(seed)
    obs, cross, within = stat(lab)
    p = (1 + sum(stat(rng.permutation(lab))[0] >= obs for _ in range(n_perm))) / (n_perm + 1)
    cen = Z[ref].mean(0)
    dref = np.sqrt(((Z - cen) ** 2).mean(1))
    rho, prho = sps.spearmanr(np.arange(n), dref)
    dow = pd.DatetimeIndex(days).dayofweek.values[sel]
    same = dow[:, None] == dow[None, :]
    iu = np.triu_indices(len(sel), 1)
    rr = float(D[iu][~same[iu]].mean() / D[iu][same[iu]].mean()) if same[iu].any() else np.nan
    return dict(status="ok", n_days=n, within=float(within), cross=float(cross), cross_minus_within=float(obs), perm_p=float(p),
                drift_rho=float(rho), drift_p=float(prho), weekday_ratio=rr)


def identification(F, homes, cohort, floor, profile_frac=0.5, test_frac=0.5):
    """F: home -> X (good days in time order). Profile = first profile_frac of a home's good days, test = last test_frac
    (a gap between them reduces the advantage of neighbouring days).
    Standardised on all profile days pooled. Returns summary + per-home rows + centroid distance matrix."""
    P, T = {}, {}
    for h in homes:
        X = F[h]
        n = len(X)
        P[h], T[h] = X[: max(1, int(round(profile_frac * n)))], X[n - max(1, int(round(test_frac * n))):]
    allP = np.vstack([P[h] for h in homes])
    mu, sd = allP.mean(0), np.maximum(allP.std(0), floor)
    C = np.vstack([((P[h] - mu) / sd).mean(0) for h in homes])
    rows, ranks, aucs = [], [], []
    for i, h in enumerate(homes):
        Zt = (T[h] - mu) / sd
        D = masked_rms(Zt, C)                                  # test days x homes
        order = np.argsort(D, 1)
        rank = np.argmax(order == i, 1) + 1
        own, other = D[:, i], np.delete(D, i, 1)
        auc = float((own[:, None] < other).mean())             # P(own centroid closer than another home's)
        same_c = [j for j, g in enumerate(homes) if cohort[g] == cohort[h]]
        Dc = D[:, same_c]
        rank_c = np.argmax(np.argsort(Dc, 1) == same_c.index(i), 1) + 1
        rows.append(dict(home=h, cohort=cohort[h], test_days=len(Zt), top1=float((rank == 1).mean()),
                         top5=float((rank <= 5).mean()), mrr=float((1 / rank).mean()), auroc_own_vs_other=auc,
                         cohort_size=len(same_c), top1_within_cohort=float((rank_c == 1).mean())))
        ranks.append(rank); aucs.append(auc)
    R = pd.DataFrame(rows)
    allr = np.concatenate(ranks)
    summ = dict(homes=len(homes), test_days=int(len(allr)), top1=float((allr == 1).mean()), top5=float((allr <= 5).mean()),
                chance_top1=1 / len(homes), mrr=float((1 / allr).mean()), auroc_own_vs_other=float(np.mean(aucs)),
                top1_within_cohort=float(np.average(R.top1_within_cohort, weights=R.test_days)),
                macro_top1=float(R.top1.mean()), macro_top1_within_cohort=float(R.top1_within_cohort.mean()),
                chance_within_cohort=float(np.average(1 / R.cohort_size, weights=R.test_days)))
    Dc = masked_rms(C, C)
    return summ, R, pd.DataFrame(Dc, index=homes, columns=homes)
