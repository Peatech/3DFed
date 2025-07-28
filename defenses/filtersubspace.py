import torch
import logging
import os
import numpy as np
import sklearn.metrics.pairwise as smp
from defenses.fedavg import FedAvg

logger = logging.getLogger('logger')

class FilterSubspace(FedAvg):
    """
    Defense based on filter-subspace similarity:
      - computes cosine distances over each client's filter-atom layers
      - flags clients whose distance > tau
      - returns per-client weights (zeros for flagged) and flagged IDs
    """

    def __init__(self, params):
        super().__init__(params)
        self.tau = params.fs_tau       # similarity threshold
        self.layer_names = params.fs_layer_names  # list of conv layers to check

    def aggr(self, weight_accumulator, global_model):
    """
    Robust cosine-distance screen (median centre + MAD z-scores).
    Down-weights or drops clients whose atom subspace is an outlier.
    """
    # ───────────────────────────── BEGIN PATCH ─────────────────────────────
    n_clients = self.params.fl_total_participants
    atom_vecs = []

    # 1)  Load each client's saved update and flatten the chosen *.atoms
    for cid in range(n_clients):
        update_path = f'{self.params.folder_path}/saved_updates/update_{cid}.pth'
        upd = torch.load(update_path, map_location='cpu')

        flattened_parts = []
        for p_name, tensor in upd.items():
            if any(layer in p_name for layer in self.layer_names):
                flattened_parts.append(tensor.flatten().numpy())
        if not flattened_parts:
            raise ValueError(f"[FilterSubspace] No atoms found for client {cid}. "
                             f"Check fs_layer_names={self.layer_names}")

        atom_vecs.append(np.concatenate(flattened_parts))

    atom_vecs = np.stack(atom_vecs, axis=0)      #  (n, dim)

    # 2)  Robust reference = coordinate-wise median vector
    ref_vec = np.median(atom_vecs, axis=0, keepdims=True)   # shape (1, dim)

    # 3)  Cosine distance of each client to the median reference
    #     dist_i = 1 - cos(atom_i, ref)
    dists = 1.0 - smp.cosine_similarity(atom_vecs, ref_vec).ravel()

    # 4)  MAD z-score
    med = np.median(dists)
    mad = np.median(np.abs(dists - med)) + 1e-6
    z_scores = 0.6745 * np.abs(dists - med) / mad    # robust z

    # 5)  Convert z to weights   (τ default 2.5)
    w = np.clip(self.tau - z_scores, a_min=0.0, a_max=None)
    if w.sum() == 0.0:          # extreme fallback
        w += 1e-3
    w /= w.sum()

    flagged = [cid for cid, wi in enumerate(w) if wi == 0.0]

    logger.info(
        f"[FilterSubspace] dist={dists.round(3).tolist()}  "
        f"weights={w.round(3).tolist()}  flagged={flagged}"
    )
    if flagged:
        logger.warning(f"[FilterSubspace]  dropped clients → {flagged}")

    # 6)  Apply weights in FedAvg accumulation
    for cid in range(n_clients):
        update_path = f'{self.params.folder_path}/saved_updates/update_{cid}.pth'
        upd = torch.load(update_path, map_location='cpu')
        scale = float(w[cid])
        for name, tensor in upd.items():
            if self.check_ignored_weights(name):
                continue
            weight_accumulator[name].add_(tensor.to(self.params.device) * scale)

    return weight_accumulator
    # ────────────────────────────── END PATCH ──────────────────────────────

