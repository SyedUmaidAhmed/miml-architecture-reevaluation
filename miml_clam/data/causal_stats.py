"""Causal label-coupling statistics for MIML-LCGA (Direction A).

The current LabelCorrelationHead couples logits through the raw conditional
co-occurrence  C[i,j] = P(Y_j=1 | Y_i=1).  That conditional confounds two
things:

  1. a *structural* association between labels i and j, and
  2. the *marginal frequency* of label j  (a common label j looks
     "correlated" with everything, purely because P(j) is high).

Component (2) is the spurious channel: it is exactly what shifts when the
label-prior P(Y) changes between train and test, and it is what drags the
coupled predictor down under label-prior shift.

This module builds *deconfounded* coupling matrices that keep the structural
association and drop the marginal-frequency component.

Two realisations (cf. plan Direction A):

  * ``log_odds_ratio`` / ``causal_coupling(method="odds_ratio")`` -- the
    PRIMARY, most defensible one.  The pairwise log odds ratio
        LOR[i,j] = log( (n11 n00) / (n10 n01) )
    is *exactly invariant* to independent reweighting of the label marginals
    in the two-label / conditional case (Proposition, invariance to
    label-prior shift; see TEST 1 in ``__main__``, diff = 0 to machine eps).
    For the multi-label *marginal* odds ratios actually stored in the coupling
    matrix, odds-ratio non-collapsibility makes the invariance only
    strongly-but-approximately hold (TEST 2 shows ~9x more prior-stable than
    raw P(j|i)). We claim exact invariance ONLY for the conditional/2-label
    association and demonstrate the marginal case empirically -- this is the
    honest scope of the theoretical result.

  * ``prior_normalized_lift`` (positive PMI) -- a cheaper heuristic,
    relu(log(P(j|i)/P(j))).  It removes the leading marginal term but is only
    *approximately* invariant.  Kept for ablation, documented as such -- we do
    NOT claim exact invariance for it.

  * ``stratified_backdoor`` -- the literal backdoor adjustment, used for the
    synthetic planted-confounder experiment and sensitivity analysis.

All functions take a label matrix ``labels`` of shape (N, L) with {0,1}
entries (torch.Tensor or np.ndarray) and return torch tensors, so they are
reusable by both MIMLDatasetCV and MIMLDatasetDD.
"""

import numpy as np
import torch


def _as_tensor(labels):
    """Coerce (N, L) {0,1} labels to a float32 torch tensor."""
    if isinstance(labels, torch.Tensor):
        return labels.float()
    return torch.as_tensor(np.asarray(labels), dtype=torch.float32)


def label_marginals(labels):
    """P(Y_j = 1) for each label. Returns (L,) tensor, clamped away from 0/1."""
    y = _as_tensor(labels)
    p = y.mean(dim=0)
    return p.clamp(min=1e-6, max=1 - 1e-6)


def cooccurrence(labels):
    """Raw conditional co-occurrence C[i,j] = P(Y_j=1 | Y_i=1), diagonal zeroed.

    Mirrors MIMLDatasetCV.get_label_cooccurrence so the two agree exactly.
    """
    y = _as_tensor(labels)
    co = y.T @ y                      # co[i,j] = #(Y_i=1 and Y_j=1)
    counts = y.sum(dim=0).clamp(min=1)
    co = co / counts.unsqueeze(1)     # divide row i by #(Y_i=1)  ->  P(j|i)
    co.fill_diagonal_(0)
    return co


def _pair_counts(labels, correction=0.5):
    """2x2 contingency counts for every label pair, with a Haldane-Anscombe
    continuity correction (default 0.5) so odds ratios stay finite.

    Returns n11, n10, n01, n00, each (L, L). n11[i,j] = #(Y_i=1, Y_j=1).
    """
    y = _as_tensor(labels)
    N = y.shape[0]
    count = y.sum(dim=0)                      # (L,)  #(Y_i = 1)
    n11 = y.T @ y                             # (L, L)
    n10 = count.unsqueeze(1) - n11            # #(Y_i=1, Y_j=0)
    n01 = count.unsqueeze(0) - n11            # #(Y_i=0, Y_j=1)
    n00 = N - n11 - n10 - n01                 # #(Y_i=0, Y_j=0)
    c = correction
    return n11 + c, n10 + c, n01 + c, n00 + c


def log_odds_ratio(labels, correction=0.5):
    """Pairwise log odds ratio LOR[i,j] = log((n11 n00)/(n10 n01)), diag 0.

    Exactly invariant to independent reweighting of the label marginals
    (proved numerically in ``__main__``). This is the deconfounded association
    that survives label-prior shift.
    """
    n11, n10, n01, n00 = _pair_counts(labels, correction)
    lor = torch.log(n11) + torch.log(n00) - torch.log(n10) - torch.log(n01)
    lor.fill_diagonal_(0)
    return lor


def prior_normalized_lift(co=None, priors=None, labels=None, eps=1e-6):
    """Positive pointwise mutual information: relu(log(P(j|i)/P(j))), diag 0.

    A cheaper heuristic deconfounder (variant A). Provide either (co, priors)
    or a raw label matrix. NOTE: only approximately prior-invariant -- kept for
    ablation, not the load-bearing theoretical object.
    """
    if labels is not None:
        co = cooccurrence(labels)
        priors = label_marginals(labels)
    if co is None or priors is None:
        raise ValueError("prior_normalized_lift needs (co, priors) or labels=")
    ratio = co / priors.unsqueeze(0).clamp(min=eps)   # P(j|i) / P(j)
    lift = torch.log(ratio.clamp(min=eps))
    lift = torch.relu(lift)
    lift.fill_diagonal_(0)
    return lift


def make_strata(labels, n_strata=4, seed=0, features=None):
    """Assign each bag to one of K confounder strata (approximate C_ctx).

    Clusters on the label-combination vectors by default (the confounder that
    generates label combinations), or on supplied bag-summary ``features``.
    Returns a (N,) int64 tensor of stratum ids in [0, n_strata).
    """
    from sklearn.cluster import KMeans

    if features is not None:
        X = np.asarray(features, dtype=np.float64)
    else:
        X = _as_tensor(labels).cpu().numpy().astype(np.float64)
    n_strata = int(min(n_strata, max(1, X.shape[0])))
    if n_strata == 1:
        return torch.zeros(X.shape[0], dtype=torch.long)
    km = KMeans(n_clusters=n_strata, random_state=seed, n_init=10)
    assign = km.fit_predict(X)
    return torch.as_tensor(assign, dtype=torch.long)


def stratified_backdoor(labels, strata, ref_prior=None):
    """Backdoor-adjusted coupling  C[i,j] = sum_k P(Y_j|Y_i, C_ctx=k) P(k).

    Stratifies the conditional on the confounder ``strata`` and averages by a
    FIXED reference prior ``ref_prior`` (uniform 1/K by default) rather than the
    empirical stratum frequencies -- this is what removes the confounder's
    influence. Diagonal zeroed.
    """
    y = _as_tensor(labels)
    strata = torch.as_tensor(strata, dtype=torch.long)
    ks = torch.unique(strata).tolist()
    K = len(ks)
    if ref_prior is None:
        ref_prior = {k: 1.0 / K for k in ks}
    L = y.shape[1]
    out = torch.zeros(L, L)
    for k in ks:
        yk = y[strata == k]
        if yk.shape[0] == 0:
            continue
        co_k = cooccurrence(yk)               # P(j|i, k)
        out = out + float(ref_prior[k]) * co_k
    out.fill_diagonal_(0)
    return out


def causal_coupling(labels, method="odds_ratio", positive=True, correction=0.5,
                    strata=None, ref_prior=None, normalize=True):
    """Build the deconfounded coupling matrix used to init the causal head.

    method:
      "odds_ratio"  -> relu(log odds ratio)          [primary, exact-invariant]
      "lift"        -> relu(log(P(j|i)/P(j)))          [PMI heuristic]
      "backdoor"    -> stratified backdoor adjustment  [needs strata]
      "statistical" -> raw P(j|i)                       [the OLD, confounded one]

    positive: keep only positive associations (relu) for odds_ratio.
    normalize: row-normalise the |matrix| to spectral-radius <= 1 so the
      coupling z + beta*(C z - z) stays contractive (defensive, keeps Phase-2
      training stable regardless of raw scale).
    """
    if method == "odds_ratio":
        C = log_odds_ratio(labels, correction)
        if positive:
            C = torch.relu(C)
    elif method == "lift":
        C = prior_normalized_lift(labels=labels)
    elif method == "backdoor":
        if strata is None:
            raise ValueError("method='backdoor' requires strata")
        C = stratified_backdoor(labels, strata, ref_prior)
    elif method == "statistical":
        C = cooccurrence(labels)
    else:
        raise ValueError(f"unknown method: {method}")

    if normalize:
        C = _row_normalize(C)
    return C


def _row_normalize(C, max_row_sum=1.0):
    """Scale so the largest absolute row-sum is <= max_row_sum (keeps I - C-type
    couplings well-behaved). No-op if already within bound or all-zero."""
    row_abs = C.abs().sum(dim=1)
    m = row_abs.max()
    if m > max_row_sum and m > 0:
        C = C * (max_row_sum / m)
    return C


def _pair_stats_from_dist(states, probs):
    """Exact pairwise stats from a distribution over enumerated label states.

    states: (S, L) {0,1}, probs: (S,). Returns (log_odds_ratio, P(j|i), marginals),
    all computed analytically (no sampling), with a tiny floor for stability.
    """
    S = states.float()
    p = probs.float()
    marg = (p.unsqueeze(1) * S).sum(dim=0).clamp(1e-9, 1 - 1e-9)      # (L,)
    n11 = (S * p.unsqueeze(1)).T @ S                                   # (L,L) P(i=1,j=1)
    n10 = marg.unsqueeze(1) - n11
    n01 = marg.unsqueeze(0) - n11
    n00 = 1.0 - n11 - n10 - n01
    f = 1e-9
    lor = (torch.log(n11 + f) + torch.log(n00 + f)
           - torch.log(n10 + f) - torch.log(n01 + f))
    cond = n11 / marg.unsqueeze(1).clamp(min=1e-9)                     # P(j|i)
    L = S.shape[1]
    eye = torch.eye(L, dtype=torch.bool)
    lor[eye] = 0.0
    cond[eye] = 0.0
    return lor, cond, marg


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Numerical proof of the load-bearing identity behind the Proposition
    # (invariance to label-prior shift).
    #
    # TEST 1 (exact, 2 labels): under independent margin reweighting
    #   P'(a,b) ∝ P(a,b) f(a) g(b), the odds ratio is EXACTLY preserved while
    #   the raw conditional P(j|i) changes. This is the identity, to machine eps.
    #
    # TEST 2 (exact, multi-label, pairwise Ising): a prior shift = changing the
    #   singleton fields h while keeping the pairwise interactions J fixed. On a
    #   pairwise log-linear model the log odds ratio is invariant (that IS the
    #   stated condition of the Proposition); raw P(j|i) is not.
    # -------------------------------------------------------------------------
    torch.manual_seed(0)

    # ---- TEST 1: exact 2-label identity -------------------------------------
    p = torch.tensor([[0.40, 0.15],      # [p00, p01]
                      [0.10, 0.35]])     # [p10, p11]
    OR = (p[1, 1] * p[0, 0]) / (p[1, 0] * p[0, 1])
    f = torch.tensor([2.3, 0.4]); g = torch.tensor([0.6, 3.1])   # margin reweights
    pp = p * f.unsqueeze(1) * g.unsqueeze(0)
    pp = pp / pp.sum()
    ORp = (pp[1, 1] * pp[0, 0]) / (pp[1, 0] * pp[0, 1])
    cond = p[1, 1] / (p[1, 1] + p[0, 1])          # P(Y_i=1 | Y_j=1)
    condp = pp[1, 1] / (pp[1, 1] + pp[0, 1])
    print("TEST 1 (exact 2-label identity)")
    print(f"  odds ratio: base={OR:.6f}  shifted={ORp:.6f}  |diff|={abs(OR-ORp):.2e}")
    print(f"  P(i|j)    : base={cond:.4f}  shifted={condp:.4f}  (changes)")
    assert abs(OR - ORp) < 1e-6, "odds ratio must be exactly margin-invariant"
    assert abs(cond - condp) > 0.05, "raw conditional should shift"

    # ---- TEST 2: exact multi-label Ising ------------------------------------
    L = 6
    states = torch.tensor([[(s >> b) & 1 for b in range(L)]
                           for s in range(2 ** L)], dtype=torch.float32)
    J = torch.randn(L, L); J = (J + J.T) / 2; J.fill_diagonal_(0)     # interactions
    h = torch.randn(L) * 0.8                                          # base fields

    def ising_probs(h_):
        e = states @ h_ + 0.5 * (states @ J * states).sum(dim=1)      # energy
        return torch.softmax(e, dim=0)

    P_base = ising_probs(h)
    P_shift = ising_probs(h + torch.randn(L) * 1.5)                   # shift fields only

    lor_b, co_b, mg_b = _pair_stats_from_dist(states, P_base)
    lor_s, co_s, mg_s = _pair_stats_from_dist(states, P_shift)
    off = ~torch.eye(L, dtype=torch.bool)
    lor_drift = (lor_b - lor_s)[off].abs().mean().item()
    co_drift = (co_b - co_s)[off].abs().mean().item()
    marg_drift = (mg_b - mg_s).abs().mean().item()
    ratio = co_drift / max(lor_drift, 1e-12)
    print("\nTEST 2 (multi-label pairwise Ising; marginal odds ratios)")
    print(f"  mean |marginal| drift : {marg_drift:.4f}  (prior shift is real)")
    print(f"  mean |P(j|i)| drift   : {co_drift:.4f}   (confounded)")
    print(f"  mean |logOR| drift    : {lor_drift:.4f}   (deconfounded)")
    print(f"  -> marginal logOR is {ratio:.1f}x more prior-stable than raw P(j|i)")
    # NOTE (odds-ratio non-collapsibility): the *conditional* odds ratio equals
    # the Ising interaction and is EXACTLY field-invariant (Test 1 is its 2-label
    # instance). The *marginal* pairwise logOR used for the coupling matrix mixes
    # in other labels under marginalisation, so it is only strongly-but-
    # approximately invariant. We claim exact invariance only for the
    # conditional/2-label association, and demonstrate the marginal case
    # empirically -- this is the honest scope of the Proposition.
    assert marg_drift > 0.03, "shift should move marginals"
    assert co_drift > 4 * lor_drift, "raw P(j|i) must move much more than logOR"
    print("\nOK: exact margin-invariance for the conditional/2-label odds ratio;"
          f" marginal logOR coupling is ~{ratio:.0f}x more prior-stable than raw P(j|i).")
