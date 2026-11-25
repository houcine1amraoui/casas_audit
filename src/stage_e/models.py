"""Stage E models. All forecasters predict the standardised value of every channel at step t from steps t-w..t-1
(plus the masks and time-of-day/weekend covariates); the autoencoder reconstructs the window.

  persistence      y_hat(t) = x(t-1)
  seasonal_naive   y_hat(t) = x(t - 1 day)
  var              ridge-regularised linear AR on the flattened window + covariates (lambda chosen on val)
  lstm             LSTM forecaster
  lstm_ae          LSTM autoencoder (reconstruction, scored separately)
  gdn_nograph      GDN with the graph removed (each node attends only to itself)        -> graph ablation
  gdn              GDN (Deng & Hooi 2021): learned top-k sensor graph + graph attention
  gdn_hier         GDN + fixed hierarchy: sensor -> group (room) -> home pooling, attention between groups
  gdn_hier_random  same, groups randomly re-assigned (same sizes)                       -> hierarchy control
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn


# ---------------------------------------------------------------- numpy baselines
class Persistence:
    kind = "forecast"
    n_params = 0

    def __init__(self, ds, cfg): pass

    def fit(self, tr, va): return {}

    def predict(self, a, ds):
        return a["x"][:, -1, :]


class SeasonalNaive:
    kind = "forecast"
    n_params = 0

    def __init__(self, ds, cfg): pass

    def fit(self, tr, va): return {}

    def predict(self, a, ds):
        src = a["t"] - ds["spd"]
        Zs = a.get("Zsrc", ds["Z"])     # injected copies pass their own Z
        ok = (src >= 0)
        out = np.zeros_like(a["y"])
        out[ok] = Zs[src[ok]]
        return out


class RidgeVAR:
    kind = "forecast"

    def __init__(self, ds, cfg):
        self.lams = cfg["stage_e"]["var_lambdas"]
        self.res = residual_mask(ds, cfg)        # [N]: 1 for level channels when residual forecasting is on

    @staticmethod
    def _feat(a):
        n = len(a["x"])
        return np.c_[a["x"].reshape(n, -1), a["c"][:, -1, :], np.ones((n, 1), np.float32)].astype(np.float64)

    def fit(self, tr, va):
        A, Ym = self._feat(tr), tr["ym"]
        Y = (tr["y"] - self.res * tr["x"][:, -1, :]).astype(np.float64)
        Av = self._feat(va)
        G = A.T @ A
        best = None
        for lam in self.lams:
            Wt = np.linalg.solve(G + lam * len(A) * np.eye(G.shape[0]), A.T @ Y)
            err = (((Av @ Wt) + self.res * va["x"][:, -1, :] - va["y"]) ** 2 * va["ym"]).sum() / va["ym"].sum()
            if best is None or err < best[0]:
                best = (err, lam, Wt)
        self.W, self.lam = best[2], best[1]
        self.n_params = int(self.W.size)
        return dict(lambda_=self.lam)

    def predict(self, a, ds):
        return (self._feat(a) @ self.W + self.res * a["x"][:, -1, :]).astype(np.float32)


# ---------------------------------------------------------------- torch models
class LSTMForecaster(nn.Module):
    kind = "forecast"

    def __init__(self, N, w, hp):
        super().__init__()
        H = hp["hidden"]
        self.rnn = nn.LSTM(2 * N + 3, H, num_layers=hp["layers"], batch_first=True, dropout=hp["dropout"] if hp["layers"] > 1 else 0)
        self.out = nn.Sequential(nn.Linear(H + 3, H), nn.ReLU(), nn.Linear(H, N))
        self.register_buffer("res", torch.as_tensor(hp.get("residual_mask", np.zeros(N)), dtype=torch.float32))

    def forward(self, x, m, c):
        h, _ = self.rnn(torch.cat([x, m, c[:, :-1]], -1))
        return self.out(torch.cat([h[:, -1], c[:, -1]], -1)) + self.res * x[:, -1]


class LSTMAE(nn.Module):
    kind = "recon"

    def __init__(self, N, w, hp):
        super().__init__()
        H, D = hp["hidden"], hp["latent"]
        self.w = w
        self.enc = nn.LSTM(2 * N + 3, H, batch_first=True)
        self.lat = nn.Linear(H, D)
        self.dec = nn.LSTM(D + 3, H, batch_first=True)
        self.out = nn.Linear(H, N)

    def forward(self, x, m, c):
        h, _ = self.enc(torch.cat([x, m, c[:, :-1]], -1))
        z = self.lat(h[:, -1])
        d, _ = self.dec(torch.cat([z[:, None, :].expand(-1, self.w, -1), c[:, :-1]], -1))
        return self.out(d)


class HierAE(nn.Module):
    """Hierarchical autoencoder: sensor -> room -> home.
    Encoder: each sensor's window (values + mask) -> sensor feature (shared weights + sensor embedding); sensors pooled
    into their room (mean) -> room code (room_dim); rooms pooled into the home (mean) -> home code (latent).
    Decoder: each sensor's window is rebuilt from the home code, ITS room's code, its sensor embedding and the window's
    time of day. The bottleneck is the codes, so room-level structure must carry what the home code does not.
    groups = room id per sensor (LongTensor [N]); a random assignment gives the control, and one single group with
    room_dim x n_rooms gives the flat version with the same total code size."""
    kind = "recon"

    def __init__(self, N, w, hp, groups, room_dim):
        super().__init__()
        d, H, D = hp["emb_dim"], hp["hidden"], hp["latent"]
        self.w = w
        G = int(groups.max()) + 1
        self.register_buffer("groups", groups.long())
        P = torch.zeros(G, N)
        P[groups.long(), torch.arange(N)] = 1.0
        self.register_buffer("pool", P / P.sum(1, keepdim=True))
        self.emb = nn.Embedding(N, d)
        self.sens = nn.Sequential(nn.Linear(2 * w, d), nn.ReLU())
        self.room = nn.Sequential(nn.Linear(d, d), nn.ReLU())
        self.room_code = nn.Linear(d, room_dim)
        self.home_code = nn.Linear(d, D)
        self.dec = nn.Sequential(nn.Linear(D + room_dim + d + 3, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, w))

    def forward(self, x, m, c):
        B, w, N = x.shape
        e = self.emb.weight[None].expand(B, -1, -1)                         # [B,N,d]
        hs = self.sens(torch.cat([x, m], 1).transpose(1, 2)) + e            # [B,N,d]
        hr = self.room(self.pool @ hs)                                      # [B,G,d]
        rc = self.room_code(hr)                                             # [B,G,room_dim]
        hc = self.home_code(hr.mean(1))                                     # [B,D]
        t0 = c[:, 0][:, None, :].expand(-1, N, -1)                          # time of day of the window start
        z = torch.cat([hc[:, None, :].expand(-1, N, -1), rc[:, self.groups], e, t0], -1)
        return self.dec(z).transpose(1, 2)                                  # [B,w,N]


class GDN(nn.Module):
    """Graph Deviation Network (Deng & Hooi, AAAI 2021), compact dense implementation.
    graph: 'learned' (top-k cosine neighbours of the sensor embeddings, re-computed every step),
           'none' (self only). groups (LongTensor [N]) enables the fixed-hierarchy head."""
    kind = "forecast"

    def __init__(self, N, w, hp, graph="learned", groups=None):
        super().__init__()
        d = hp["emb_dim"]
        self.N, self.k, self.graph = N, min(hp["topk"], N - 1), graph
        self.emb = nn.Embedding(N, d)
        nn.init.kaiming_uniform_(self.emb.weight, a=5 ** 0.5)
        self.lin = nn.Linear(2 * w, d, bias=False)
        self.att_i = nn.Parameter(torch.randn(2 * d) * 0.1)
        self.att_j = nn.Parameter(torch.randn(2 * d) * 0.1)
        self.norm = nn.LayerNorm(d)
        self.hier = groups is not None
        extra = 0
        if self.hier:
            self.register_buffer("groups", groups.long())
            G = int(groups.max()) + 1
            self.G = G
            self.gemb = nn.Embedding(G, d)
            self.glin = nn.Linear(d, d, bias=False)
            self.gatt_i = nn.Parameter(torch.randn(2 * d) * 0.1)
            self.gatt_j = nn.Parameter(torch.randn(2 * d) * 0.1)
            self.gnorm = nn.LayerNorm(d)
            P = torch.zeros(G, N)
            P[groups.long(), torch.arange(N)] = 1.0
            self.register_buffer("pool", P / P.sum(1, keepdim=True))
            extra = 2 * d
        H = hp["hidden"]
        self.out = nn.Sequential(nn.Linear(d + extra + 3, H), nn.ReLU(), nn.Linear(H, 1))
        self.register_buffer("res", torch.as_tensor(hp.get("residual_mask", np.zeros(N)), dtype=torch.float32))

    def adjacency(self):
        N = self.N
        eye = torch.eye(N, dtype=torch.bool, device=self.emb.weight.device)
        if self.graph == "none":
            return eye
        v = Fn.normalize(self.emb.weight, dim=1)
        sim = v @ v.T
        sim = sim.masked_fill(eye, -1e9)
        nb = sim.topk(self.k, dim=1).indices
        A = torch.zeros(N, N, dtype=torch.bool, device=sim.device)
        A[torch.arange(N, device=sim.device)[:, None], nb] = True
        return A | eye                                   # row i = neighbours whose information flows to i

    def forward(self, x, m, c):
        B = x.shape[0]
        inp = torch.cat([x, m], 1).transpose(1, 2)        # [B, N, 2w]
        hx = self.lin(inp)                                # [B, N, d]
        v = self.emb.weight[None].expand(B, -1, -1)
        g = torch.cat([v, hx], -1)                        # [B, N, 2d]
        e = Fn.leaky_relu((g @ self.att_i)[:, :, None] + (g @ self.att_j)[:, None, :], 0.2)
        A = self.adjacency()
        e = e.masked_fill(~A[None], float("-inf"))
        alpha = torch.softmax(e, -1)
        z = Fn.relu(self.norm(alpha @ hx))                # [B, N, d]
        feats = [z * v]
        if self.hier:
            r = self.pool @ z                             # [B, G, d] mean over group members
            u = self.gemb.weight[None].expand(B, -1, -1)
            hr = self.glin(r)
            gg = torch.cat([u, hr], -1)
            ge = Fn.leaky_relu((gg @ self.gatt_i)[:, :, None] + (gg @ self.gatt_j)[:, None, :], 0.2)
            r2 = Fn.relu(self.gnorm(torch.softmax(ge, -1) @ hr))   # [B, G, d] (all groups attend to all)
            home = r2.mean(1, keepdim=True).expand(-1, self.N, -1)
            feats += [r2[:, self.groups], home]
        ct = c[:, -1][:, None, :].expand(-1, self.N, -1)
        return self.out(torch.cat(feats + [ct], -1)).squeeze(-1) + self.res * x[:, -1]

    def neighbours(self):
        with torch.no_grad():
            A = self.adjacency().cpu().numpy()
        np.fill_diagonal(A, False)
        return A


def residual_mask(ds, cfg):
    """Residual (predict change from the last step) only for level channels: states such as pressure or plug current
    persist from one 10-min step to the next; event counts do not, so persistence is a poor starting point for them."""
    return (np.asarray(ds["family"]) == "levels").astype(np.float32) * float(cfg["stage_e"]["residual"])


def group_ids(ds, level, seed=None):
    labels = ds["room"] if level == "room" else ds["group"]
    uniq = sorted(set(labels))
    g = np.array([uniq.index(x) for x in labels])
    if seed is not None:
        g = np.random.default_rng(10_000 + seed).permutation(g)   # same group sizes, random membership
    return torch.tensor(g), uniq


def build(name, ds, cfg, seed):
    hp = dict(cfg["stage_e"]["hp"], residual_mask=residual_mask(ds, cfg))
    N, w = len(ds["names"]), ds["w"]
    level = cfg["stage_e"]["hierarchy_level"]
    if name == "persistence":
        return Persistence(ds, cfg)
    if name == "seasonal_naive":
        return SeasonalNaive(ds, cfg)
    if name == "var":
        return RidgeVAR(ds, cfg)
    if name == "lstm":
        return LSTMForecaster(N, w, hp)
    if name == "lstm_ae":
        return LSTMAE(N, w, hp)
    if name == "gdn":
        return GDN(N, w, hp, "learned")
    if name == "gdn_nograph":
        return GDN(N, w, hp, "none")
    if name.startswith("hae"):        # hae (rooms) | hae_random (same sizes, random members) | hae_flat (one group)
        g, uniq = group_ids(ds, level, seed if name == "hae_random" else None)
        rd = int(hp.get("room_dim", 8))
        if name == "hae_flat":
            return HierAE(N, w, hp, torch.zeros(N, dtype=torch.long), rd * len(uniq))
        return HierAE(N, w, hp, g, rd)
    if name.startswith("gdn_hier"):   # gdn_hier[_group][_random]: room level unless _group; _random = shuffled membership
        lvl = "group" if "_group" in name else level
        return GDN(N, w, hp, "learned", group_ids(ds, lvl, seed if name.endswith("_random") else None)[0])
    raise ValueError(name)
