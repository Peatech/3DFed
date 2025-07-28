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
        # 1) Load all updates
        n = self.params.fl_total_participants
        all_updates = []
        for i in range(n):
            path = f'{self.params.folder_path}/saved_updates/update_{i}.pth'
            upd = torch.load(path)
            # extract just the filter-atom parameters
            vec = []
            for name, tensor in upd.items():
                if any(ln in name for ln in self.layer_names):
                    vec.append(tensor.detach().cpu().numpy().ravel())
            all_updates.append(np.concatenate(vec))
        all_updates = np.stack(all_updates, axis=0)

        # 2) compute pairwise cosine similarities
        sims = smp.cosine_similarity(all_updates)
        # for each client, average similarity to others
        avg_sim = sims.mean(axis=1)

        # 3) determine weights and flagged clients
        weights = np.clip(avg_sim / self.tau, 0, 1)   # simple linear scaling
        flagged = [i for i, s in enumerate(avg_sim) if s < self.tau]
        distances = 1 - avg_sim 

        logger.info(f"[FilterSubspace] dist={1-avg_sim.tolist()}  weights={weights.tolist()}  flagged={flagged}")
        logger.warning(f"[Round {self.params.current_round}]  dropped clients → {flagged}")

        # 4) apply weights in accumulation
        for i in range(n):
            path = f'{self.params.folder_path}/saved_updates/update_{i}.pth'
            upd = torch.load(path)
            scale = weights[i]
            for name, tensor in upd.items():
                if self.check_ignored_weights(name):
                    continue
                weight_accumulator[name].add_(tensor.to(self.params.device) * scale)

        return weight_accumulator
