import torch
import torch.nn as nn
import torch.optim as optim

class GSAMOptimizer:
    def __init__(self, model, base_optimizer, rho_max=0.05, rho_min=0.01, alpha=0.5, total_epochs=100):
        """
        Initializes GSAM optimizer.
        
        Parameters:
        - model: PyTorch model to train.
        - base_optimizer: Base optimizer, e.g., SGD or Adam.
        - rho_max: Maximum neighborhood size (perturbation radius).
        - rho_min: Minimum neighborhood size at the start of training.
        - alpha: Scaling factor for gradient perturbation.
        - total_epochs: Total number of training epochs to control the gradual increase of rho.
        """
        # Enable anomaly detection for debugging
        torch.autograd.set_detect_anomaly(True)
        self.model = model
        self.base_optimizer = base_optimizer
        self.rho_max = rho_max
        self.rho_min = rho_min
        self.alpha = alpha
        self.total_epochs = total_epochs

    def update_rho(self, current_epoch):
        """Calculates current rho based on the training progress."""
        rho = self.rho_min + (self.rho_max - self.rho_min) * (current_epoch / self.total_epochs)
        return rho
    def step(self, loss, current_epoch):
        """
        Performs a GSAM optimization step.
        
        Parameters:
        - loss: Computed loss for the batch (tensor, not float).
        - current_epoch: Current training epoch (used for gradual increase of rho).
        """
        # Compute initial gradients
        loss.backward(retain_graph=True)  # Compute gradients of the original loss, retain graph

        # Save current model parameters
        original_params = [p.clone() for p in self.model.parameters() if p.requires_grad]

        # Compute rho based on the current epoch
        rho = self.update_rho(current_epoch)

        # Apply perturbation (without in-place modification)
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    perturbation = self.alpha * rho * param.grad / (torch.norm(param.grad) + 1e-12)
                    param = param + perturbation  # Use copy_ instead of in-place add_

        # Compute gradients with perturbed parameters
        perturbed_loss = loss  # Keep loss as a tensor
        self.model.zero_grad()  # Clear previous gradients
        perturbed_loss.backward()  # Compute gradients at perturbed location

        # Restore original parameters and take an optimization step
        with torch.no_grad():
            for original, param in zip(original_params, self.model.parameters()):
                if param.requires_grad:
                    param = original  # Restore original parameters

        self.base_optimizer.step()  # Apply optimizer step
        self.base_optimizer.zero_grad()  # Reset gradients after update