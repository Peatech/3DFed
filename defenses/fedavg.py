import math
from typing import List, Any, Dict
import torch
import logging
import os
from utils.parameters import Params

logger = logging.getLogger('logger')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class FedAvg:
    params: Params
    ignored_weights = ['num_batches_tracked']#['tracked', 'running']

    def __init__(self, params: Params) -> None:
        self.params = params
        self.participating_user_ids = None  # Track actual participating users

    def set_participating_users(self, user_ids):
        """Set the actual participating user IDs for this round."""
        self.participating_user_ids = user_ids

    # FedAvg aggregation
    def aggr(self, weight_accumulator, _):
        # Use actual participating user IDs if available, otherwise fall back to range
        user_ids_to_check = (self.participating_user_ids if self.participating_user_ids 
                            else range(self.params.fl_no_models))
        
        for user_id in user_ids_to_check:
            updates_name = '{0}/saved_updates/update_{1}.pth'\
                .format(self.params.folder_path, user_id)
            if os.path.exists(updates_name):
                loaded_params = torch.load(updates_name)
                self.accumulate_weights(weight_accumulator, \
                    {key:loaded_params[key].to(self.params.device) for \
                        key in loaded_params})
            else:
                # logger.warning(f"Update file {updates_name} not found, skipping client {user_id}")
                continue

    def accumulate_weights(self, weight_accumulator, local_update):
        for name, value in local_update.items():
            weight_accumulator[name].add_(value)
    
    def get_update_norm(self, local_update):
        squared_sum = 0
        for name, value in local_update.items():
            if 'tracked' in name or 'running' in name:
                continue
            squared_sum += torch.sum(torch.pow(value, 2)).item()
        update_norm = math.sqrt(squared_sum)
        return update_norm

    def add_noise(self, sum_update_tensor: torch.Tensor, sigma):
        noised_layer = torch.FloatTensor(sum_update_tensor.shape)
        noised_layer = noised_layer.to(self.params.device)
        noised_layer.normal_(mean=0, std=sigma)
        sum_update_tensor.add_(noised_layer)

    def check_ignored_weights(self, name) -> bool:
        for ignored in self.ignored_weights:
            if ignored in name:
                return True

        return False
