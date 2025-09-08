"""
FedAvgCKA Defense Implementation

This module implements the FedAvgCKA defense mechanism for federated learning
as described in the paper. It uses Centered Kernel Alignment (CKA) to measure
similarity between client model activations and filter out potentially malicious clients.
"""

import logging
import time
import copy
import math
from typing import Dict, List, Any, Tuple, Optional, Union
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from defenses.fedavg import FedAvg

logger = logging.getLogger('logger')


class FedAvgCKA(FedAvg):
    """FedAvgCKA defense mechanism using Centered Kernel Alignment."""

    def __init__(self, params):
        super().__init__(params)
        
        # FedAvgCKA specific parameters
        self.fedavgcka_enabled = getattr(params, 'fedavgcka_enabled', True)
        self.root_dataset_size = getattr(params, 'fedavgcka_root_dataset_size', 64)
        self.root_dataset_strategy = getattr(params, 'fedavgcka_root_dataset_strategy', 'class_balanced')
        self.layer_comparison = getattr(params, 'fedavgcka_layer_comparison', 'penultimate')
        self.trim_fraction = getattr(params, 'fedavgcka_trim_fraction', 0.3)
        self.log_scores = getattr(params, 'fedavgcka_log_scores', True)
        
        # Initialize components
        self.root_dataset_loader = None
        self.telemetry = []
        self.participating_user_ids = None  # Track actual participating users
        
    def initialize_fedavgcka(self, task, global_model, device='cpu'):
        """Initialize FedAvgCKA defense with root dataset and configuration."""
        try:
            logger.info("Initializing FedAvgCKA defense...")
            
            # Create root dataset
            self.root_dataset_loader = self._create_root_dataset(task)
            
            # Store references
            self.global_model = global_model
            self.device = device
            
            logger.info(f"FedAvgCKA initialized with root dataset of size {self.root_dataset_size}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize FedAvgCKA: {e}")
            self.fedavgcka_enabled = False
            return False
    
    def _create_root_dataset(self, task):
        """Create root dataset for CKA computation."""
        if self.root_dataset_strategy == 'class_balanced':
            return self._create_class_balanced_root_dataset(task)
        else:
            return self._create_random_root_dataset(task)
    
    def _create_class_balanced_root_dataset(self, task):
        """Create a class-balanced root dataset."""
        try:
            test_dataset = task.test_dataset
            num_classes = task.num_classes if hasattr(task, 'num_classes') else 10
            samples_per_class = max(1, self.root_dataset_size // num_classes)
            
            # Get indices for each class
            class_indices = {i: [] for i in range(num_classes)}
            
            for idx, (_, label) in enumerate(test_dataset):
                if isinstance(label, torch.Tensor):
                    label = label.item()
                if len(class_indices[label]) < samples_per_class:
                    class_indices[label].append(idx)
            
            # Collect balanced indices
            selected_indices = []
            for class_id, indices in class_indices.items():
                selected_indices.extend(indices[:samples_per_class])
            
            # Create subset and dataloader
            root_subset = Subset(test_dataset, selected_indices[:self.root_dataset_size])
            root_loader = DataLoader(root_subset, batch_size=min(32, self.root_dataset_size), shuffle=False)
            
            logger.info(f"Created class-balanced root dataset: {len(selected_indices)} samples across {num_classes} classes")
            return root_loader
            
        except Exception as e:
            logger.error(f"Failed to create class-balanced root dataset: {e}")
            return self._create_random_root_dataset(task)
    
    def _create_random_root_dataset(self, task):
        """Create a random root dataset."""
        try:
            test_dataset = task.test_dataset
            total_size = len(test_dataset)
            
            # Random sampling
            indices = np.random.choice(total_size, min(self.root_dataset_size, total_size), replace=False)
            root_subset = Subset(test_dataset, indices)
            root_loader = DataLoader(root_subset, batch_size=min(32, self.root_dataset_size), shuffle=False)
            
            logger.info(f"Created random root dataset: {self.root_dataset_size} samples")
            return root_loader
            
        except Exception as e:
            logger.error(f"Failed to create random root dataset: {e}")
            return None

    def set_participating_users(self, user_ids):
        """Set the actual participating user IDs for this round."""
        self.participating_user_ids = user_ids
    
    def aggr(self, weight_accumulator, global_model):
        """Override aggregation to apply FedAvgCKA filtering."""
        if not self.fedavgcka_enabled or self.root_dataset_loader is None:
            logger.warning("FedAvgCKA not enabled or initialized, falling back to FedAvg")
            return super().aggr(weight_accumulator, global_model)
        
        try:
            start_time = time.time()
            
            # Load client updates - use actual participating user IDs
            client_models = {}
            client_weights = {}
            
            # Use actual participating user IDs if available, otherwise fall back to range
            user_ids_to_check = (self.participating_user_ids if self.participating_user_ids 
                                else range(self.params.fl_no_models))
            
            for user_id in user_ids_to_check:
                updates_name = f'{self.params.folder_path}/saved_updates/update_{user_id}.pth'
                try:
                    loaded_params = torch.load(updates_name)
                    
                    # Create client model by applying update to global model
                    client_model = copy.deepcopy(global_model)
                    for name, data in loaded_params.items():
                        if not self.check_ignored_weights(name):
                            client_model.state_dict()[name].add_(data.to(self.params.device))
                    
                    client_models[user_id] = client_model
                    client_weights[user_id] = {key: loaded_params[key].to(self.params.device) for key in loaded_params}
                    
                except FileNotFoundError:
                    logger.warning(f"Update file {updates_name} not found, skipping client {user_id}")
                    continue
            
            if not client_models:
                logger.error("No client models loaded!")
                return super().aggr(weight_accumulator, global_model)
            
            logger.info(f"Applying FedAvgCKA filtering to {len(client_models)} clients...")
            
            # Apply FedAvgCKA filtering
            filtered_weights, telemetry = self._apply_fedavgcka_filter(
                client_models, client_weights, global_model
            )
            
            # Store telemetry
            telemetry['compute_time_s'] = time.time() - start_time
            telemetry['n_original'] = len(client_models)
            self.telemetry.append(telemetry)
            
            if self.log_scores:
                self._log_filtering_results(telemetry)
            
            # Accumulate filtered weights
            for client_id, weights in filtered_weights.items():
                self.accumulate_weights(weight_accumulator, weights)
            
            logger.info(f"FedAvgCKA aggregation complete. Used {len(filtered_weights)}/{len(client_models)} clients.")
            
        except Exception as e:
            logger.error(f"FedAvgCKA filtering failed: {e}")
            return super().aggr(weight_accumulator, global_model)
    
    def _apply_fedavgcka_filter(self, client_models, client_weights, global_model):
        """Apply FedAvgCKA filtering to select clients."""
        
        # Get layer name for comparison
        layer_name = self._get_penultimate_layer_name(global_model)
        if not layer_name:
            logger.error("Could not identify penultimate layer, fallback to all clients")
            return client_weights, {'error': 'No penultimate layer found'}
        
        logger.info(f"Extracting activations from layer: {layer_name}")
        
        # Extract activations for global model and clients
        global_activations = self._get_layer_activations(global_model, layer_name)
        
        client_activations = {}
        for client_id, client_model in client_models.items():
            try:
                activations = self._get_layer_activations(client_model, layer_name)
                if activations is not None:
                    client_activations[client_id] = activations
            except Exception as e:
                logger.warning(f"Failed to extract activations for client {client_id}: {e}")
        
        if not client_activations:
            logger.error("No client activations extracted!")
            return client_weights, {'error': 'No activations extracted'}
        
        logger.info(f"Computing pairwise CKA scores for {len(client_activations)} clients...")
        
        # Compute CKA scores
        cka_scores = {}
        for client_id, client_acts in client_activations.items():
            try:
                score = self._linear_cka(global_activations, client_acts)
                cka_scores[client_id] = score
            except Exception as e:
                logger.warning(f"CKA computation failed for client {client_id}: {e}")
                cka_scores[client_id] = 0.0  # Assign low score
        
        # Rank clients by CKA scores (higher is better/more similar)
        ranked_clients = sorted(cka_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top clients
        n_select = max(1, int(len(ranked_clients) * (1 - self.trim_fraction)))
        selected_clients = [client_id for client_id, _ in ranked_clients[:n_select]]
        excluded_clients = [client_id for client_id, _ in ranked_clients[n_select:]]
        
        logger.info(f"CKA ranking: selected {len(selected_clients)}, excluded {len(excluded_clients)}")
        
        # Filter weights
        filtered_weights = {client_id: client_weights[client_id] 
                          for client_id in selected_clients 
                          if client_id in client_weights}
        
        # Prepare telemetry
        telemetry = {
            'layer_name': layer_name,
            'cka_scores': cka_scores,
            'selected_clients': selected_clients,
            'excluded_clients': excluded_clients,
            'n_selected': len(selected_clients),
            'n_excluded': len(excluded_clients)
        }
        
        return filtered_weights, telemetry
    
    def _get_penultimate_layer_name(self, model):
        """Get the name of the penultimate layer."""
        layers = list(model.named_modules())
        
        # Look for common penultimate layer patterns
        for name, module in reversed(layers):
            if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d)):
                return name
            elif 'avgpool' in name.lower() or 'pool' in name.lower():
                return name
        
        # Fallback: look for the layer before the final classifier
        for i in range(len(layers) - 1, 0, -1):
            name, module = layers[i]
            if isinstance(module, nn.Linear):
                # Found classifier, return previous layer
                if i > 0:
                    prev_name = layers[i-1][0]
                    return prev_name
        
        # Final fallback: return the first conv or linear layer found
        for name, module in layers:
            if isinstance(module, (nn.Conv2d, nn.Linear)) and 'classifier' not in name and 'fc' not in name:
                return name
        
        return None
    
    def _get_layer_activations(self, model, layer_name):
        """Extract activations from a specific layer."""
        activations = []
        
        def hook_fn(module, input, output):
            # Handle different output types
            if isinstance(output, torch.Tensor):
                act = output.detach().cpu()
                # Flatten spatial dimensions but keep batch and channel dims
                if act.dim() > 2:
                    act = act.view(act.size(0), -1)  # (batch_size, features)
                activations.append(act)
        
        # Register hook
        target_module = None
        for name, module in model.named_modules():
            if name == layer_name:
                target_module = module
                break
        
        if target_module is None:
            logger.error(f"Layer {layer_name} not found in model")
            return None
        
        handle = target_module.register_forward_hook(hook_fn)
        
        try:
            model.eval()
            with torch.no_grad():
                for batch_idx, (data, _) in enumerate(self.root_dataset_loader):
                    data = data.to(self.device)
                    _ = model(data)
            
            # Remove hook
            handle.remove()
            
            if not activations:
                logger.error("No activations captured")
                return None
            
            # Concatenate all batches
            all_activations = torch.cat(activations, dim=0)  # (total_samples, features)
            return all_activations
            
        except Exception as e:
            handle.remove()
            logger.error(f"Error extracting activations: {e}")
            return None
    
    def _linear_cka(self, X, Y):
        """Compute Linear Centered Kernel Alignment (CKA) between two sets of activations."""
        try:
            # Ensure inputs are 2D: (n_samples, n_features)
            if X.dim() != 2 or Y.dim() != 2:
                raise ValueError(f"Expected 2D tensors, got X: {X.shape}, Y: {Y.shape}")
            
            # Convert to float for computation
            X = X.float()
            Y = Y.float()
            
            # Check for NaN or infinite values in inputs
            if torch.isnan(X).any() or torch.isinf(X).any():
                logger.warning("NaN or Inf values detected in X, returning 0.0")
                return 0.0
            if torch.isnan(Y).any() or torch.isinf(Y).any():
                logger.warning("NaN or Inf values detected in Y, returning 0.0")
                return 0.0
            
            # Center the data
            X_centered = X - X.mean(dim=0, keepdim=True)
            Y_centered = Y - Y.mean(dim=0, keepdim=True)
            
            # Check if centered data has zero variance (all identical activations)
            X_var = torch.var(X_centered)
            Y_var = torch.var(Y_centered)
            
            if X_var < 1e-12 or Y_var < 1e-12:
                logger.warning("Very low variance in activations, returning 0.0")
                return 0.0
            
            # Compute Gram matrices
            K_X = torch.mm(X_centered, X_centered.t())  # (n_samples, n_samples)
            K_Y = torch.mm(Y_centered, Y_centered.t())  # (n_samples, n_samples)
            
            # Check for numerical stability in Gram matrices
            trace_XX = torch.trace(torch.mm(K_X, K_X))
            trace_YY = torch.trace(torch.mm(K_Y, K_Y))
            
            # Add epsilon for numerical stability
            eps = 1e-12
            if trace_XX < eps or trace_YY < eps:
                logger.warning("Trace values too small, numerical instability detected")
                return 0.0
            
            # Compute CKA with numerical stability checks
            numerator = torch.trace(torch.mm(K_X, K_Y))
            denominator = torch.sqrt(trace_XX * trace_YY)
            
            if denominator < eps:
                logger.warning("Denominator too small in CKA computation")
                return 0.0
            
            cka = numerator / denominator
            
            # Final check for valid result
            if torch.isnan(cka) or torch.isinf(cka):
                logger.warning("CKA result is NaN or Inf, returning 0.0")
                return 0.0
                
            return float(cka.item())
            
        except Exception as e:
            logger.error(f"CKA computation error: {e}")
            return 0.0
    
    def _log_filtering_results(self, telemetry):
        """Log FedAvgCKA filtering results."""
        logger.info("FedAvgCKA filtering complete:")
        logger.info(f"  Selected: {telemetry['n_selected']} clients {telemetry['selected_clients']}")
        logger.info(f"  Excluded: {telemetry['n_excluded']} clients {telemetry['excluded_clients']}")
        
        if 'compute_time_s' in telemetry:
            logger.info(f"  Compute time: {telemetry['compute_time_s']:.2f} s")
        
        # Log CKA scores for excluded clients
        if telemetry['excluded_clients'] and self.log_scores:
            logger.info("  Excluded clients CKA scores:")
            for client_id in telemetry['excluded_clients']:
                score = telemetry['cka_scores'].get(client_id, 'N/A')
                logger.info(f"    Client {client_id}: {score:.4f}")
    
    def get_telemetry(self):
        """Get collected telemetry data."""
        return self.telemetry
    
    def reset_telemetry(self):
        """Reset telemetry data."""
        self.telemetry = []


# Standalone functions for compatibility
def linear_cka(X, Y):
    """Standalone function for computing Linear CKA."""
    defense = FedAvgCKA(type('MockParams', (), {})())  # Create mock instance
    return defense._linear_cka(X, Y)

def get_layer_activations(model, layer_name, root_loader, device='cpu'):
    """Standalone function for extracting layer activations."""
    defense = FedAvgCKA(type('MockParams', (), {})())
    defense.root_dataset_loader = root_loader
    defense.device = device
    return defense._get_layer_activations(model, layer_name)

def get_penultimate_layer_name(model):
    """Standalone function for getting penultimate layer name."""
    defense = FedAvgCKA(type('MockParams', (), {})())
    return defense._get_penultimate_layer_name(model)

def create_root_dataset(task, size=64, strategy='class_balanced'):
    """Standalone function for creating root dataset."""
    params = type('MockParams', (), {
        'fedavgcka_root_dataset_size': size,
        'fedavgcka_root_dataset_strategy': strategy
    })()
    defense = FedAvgCKA(params)
    return defense._create_root_dataset(task)

def apply_fedavgcka_filter(client_models, client_weights, params, root_loader, global_model=None, device='cpu'):
    """Standalone function for applying FedAvgCKA filtering."""
    defense = FedAvgCKA(params)
    defense.root_dataset_loader = root_loader
    defense.device = device
    
    if global_model is None:
        # Use first client model as reference if no global model provided
        global_model = next(iter(client_models.values()))
    
    return defense._apply_fedavgcka_filter(client_models, client_weights, global_model)
