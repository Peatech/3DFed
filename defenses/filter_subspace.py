# defenses/filter_subspace.py
"""
Filter‑Subspace Outlier Defence
--------------------------------
• Clients keep shared α‑coeffs fixed (already the case in 3DFed’s models that
  inherit from `AtomConv2d` / `NCModel`); they only update atoms.
• Server receives each client's atoms, computes cosine‑distance to a robust
  reference (median vector), applies MAD‑z threshold.
• Returns aggregation weights + list of flagged client IDs.
"""

from typing import List, Dict
import logging
import torch

from defenses.defense import Defense           # 3DFed base‑class
from utils.robust import mad_zscores

logger = logging.getLogger("logger")

# ---------------------------------------------------------------------------

def _flatten_atoms(state_dict: Dict[str, torch.Tensor],
                   layers: List[str]) -> torch.Tensor:
    """
    Concatenate flattened `atoms` tensors from the chosen layers.
    Expects keys like 'conv1.atoms'  or  'layer1.0.conv1.atoms'.
    """
    vecs = []
    for name in layers:
        key = f"{name}.atoms"
        if key not in state_dict:
            raise KeyError(f"[FilterSubspace] param '{key}' not found. "
                           f"Check fs_layers list.")
        vecs.append(state_dict[key].flatten())
    return torch.cat(vecs).cpu()        # 1‑D tensor on CPU


class FilterSubspaceDefense(Defense):
    """Called by Helper.before_agg each FL round."""

    def __init__(self, params):
        super().__init__(params)
        # Configurable via YAML
        self.layers = params.fs_layers          # List[str]
        self.tau    = params.fs_tau             # float

    # ------------------------------------------------------------------
    def before_agg(self,
                   clients_updates: List[Dict[str, torch.Tensor]],
                   client_ids: List[int]):
        """
        Parameters
        ----------
        clients_updates : list of each client’s delta state‑dict
        client_ids      : list[int]  – same order as updates

        Returns
        -------
        weights  : list[float]  – aggregation weights (sum=1)
        flagged  : list[int]    – IDs whose weight==0
        """
        # 1. build flattened atom vectors
        atom_vecs = []
        for upd in clients_updates:
            atom_vecs.append(_flatten_atoms(upd, self.layers))
        atom_vecs = torch.stack(atom_vecs)          # (n, L)

        # 2. robust reference = element‑wise median
        ref = atom_vecs.median(dim=0).values

        # 3. cosine distance (1 – cosine similarity)
        dists = 1 - torch.nn.functional.cosine_similarity(
            atom_vecs, ref.unsqueeze(0), eps=1e-8)

        # 4. MAD‑z scores & weights
        z = mad_zscores(dists)
        w = torch.clamp(self.tau - z, min=0.0)
        if torch.allclose(w, torch.zeros_like(w)):   # all outliers? soften
            w += 1e-3
        w /= w.sum()

        flagged = [cid for cid, wi in zip(client_ids, w) if wi == 0.0]

        # 5. pretty log
        logger.info(f"[FilterSubspace]  dist={dists.round(3).tolist()}  "
                    f"weights={w.round(3).tolist()}  flagged={flagged}")

        return w.tolist(), flagged
