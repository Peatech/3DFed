import logging
from typing import List
import os
import numpy as np
import torch
import sklearn.metrics.pairwise as smp
from defenses.fedavg import FedAvg
from utils.robust import mad_zscores

logger = logging.getLogger("logger")


class FilterSubspace(FedAvg):
    """
    Defence based on filter-subspace similarity (median + MAD z–score).
    Screens **only the clients selected in this FL round**.
    """

    def __init__(self, params):
        super().__init__(params)
        self.tau: float = params.fs_tau                  # z-threshold (≈2–3)
        self.layer_names: List[str] = params.fs_layer_names  # conv layers that contain `.atoms`

    # ──────────────────────────────────────────────────────────────────────
    def _flatten_atoms(self, update_state: dict) -> np.ndarray:
        """
        Concatenate all *.atoms tensors from the chosen layers in ONE client update.
        """
        parts = []
        for p_name, tensor in update_state.items():
            if any(layer in p_name for layer in self.layer_names):
                parts.append(tensor.flatten().cpu().numpy())
        if not parts:
            raise ValueError(f"[FilterSubspace] None of {self.layer_names} found in update!")
        return np.concatenate(parts)

    # ──────────────────────────────────────────────────────────────────────
    def aggr(
        self,
        weight_accumulator: dict,
        global_model,                  # (unused, but kept for FedAvg API)
        client_ids: List[int],         # <-- list of selected IDs this round
    ):
        """
        1. Load updates ONLY for `client_ids`.
        2. Compute cosine-distance to median atom vector.
        3. Derive MAD z-scores -> weights.
        4. Accumulate with those weights.
        """
        # 1️⃣  Load & flatten atoms
        atom_vecs, update_paths = [], []
        for cid in client_ids:
            up_path = os.path.join(
                self.params.folder_path, "saved_updates", f"update_{cid}.pth"
            )
            upd = torch.load(up_path, map_location="cpu")
            atom_vecs.append(self._flatten_atoms(upd))
            update_paths.append(up_path)
        atom_vecs = np.stack(atom_vecs, axis=0)          # (n_sel, dim)

        # 2️⃣  Robust reference: coordinate-wise median
        ref_vec = np.median(atom_vecs, axis=0, keepdims=True)  # (1, dim)

        # 3️⃣  Cosine distances  d_i = 1 – cos(atom_i, ref)
        dists = 1.0 - smp.cosine_similarity(atom_vecs, ref_vec).ravel()

        # 4️⃣  MAD-based z-scores  -> weights
        z = mad_zscores(torch.tensor(dists)).numpy()
        w = np.clip(self.tau - z, a_min=0.0, a_max=None)
        if w.sum() == 0.0:                          # extreme fallback
            w += 1e-3
        w /= w.sum()

        flagged = [cid for cid, wi in zip(client_ids, w) if wi == 0.0]

        logger.info(
            f"[FilterSubspace] dist={dists.round(3).tolist()} "
            f"weights={w.round(3).tolist()} flagged={flagged}"
        )
        if flagged:
            logger.warning(f"[FilterSubspace]  dropped clients → {flagged}")

        # 5️⃣  Accumulate with weights
        for cid, up_path, scale in zip(client_ids, update_paths, w):
            upd = torch.load(up_path, map_location="cpu")
            for name, tensor in upd.items():
                if self.check_ignored_weights(name):
                    continue
                weight_accumulator[name].add_(tensor.to(self.params.device) * float(scale))

        return weight_accumulator
