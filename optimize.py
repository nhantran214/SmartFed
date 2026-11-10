import torch


class GSAMOptimizer:
    """Gradient Sharpness-Aware Minimization (SAM-style ascent then descent).

    Previous implementation enabled ``torch.autograd.set_detect_anomaly(True)``
    (very slow; often triggers CUDA launch timeouts on a display GPU) and did
    not actually write perturbations onto parameters (``param = param + eps``
    only rebinds a local name).
    """

    def __init__(
        self,
        model,
        base_optimizer,
        rho_max=0.05,
        rho_min=0.01,
        alpha=0.5,
        total_epochs=100,
    ):
        self.model = model
        self.base_optimizer = base_optimizer
        self.rho_max = rho_max
        self.rho_min = rho_min
        self.alpha = alpha
        self.total_epochs = max(int(total_epochs), 1)

    def update_rho(self, current_epoch):
        """Linear schedule from rho_min → rho_max over total_epochs."""
        t = min(max(int(current_epoch), 0), self.total_epochs)
        return self.rho_min + (self.rho_max - self.rho_min) * (t / self.total_epochs)

    def _grad_norm(self):
        norms = [
            p.grad.detach().norm(p=2)
            for p in self.model.parameters()
            if p.grad is not None
        ]
        if not norms:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def _ascent(self, rho):
        """Perturb weights along the gradient: w ← w + alpha*rho*g/||g||."""
        grad_norm = self._grad_norm()
        scale = (self.alpha * rho) / (grad_norm + 1e-12)
        epsilons = []
        for p in self.model.parameters():
            if p.grad is None:
                epsilons.append(None)
                continue
            e = p.grad * scale
            p.add_(e)
            epsilons.append(e)
        return epsilons

    @torch.no_grad()
    def _restore(self, epsilons):
        for p, e in zip(self.model.parameters(), epsilons):
            if e is not None:
                p.sub_(e)

    def step(self, inputs, labels, criterion, current_epoch):
        """One GSAM step: ascent on first grads, descend with grads at w+eps.

        Returns the first-pass loss value (float) for logging.
        """
        rho = self.update_rho(current_epoch)

        # ----- first forward / backward at w -----
        self.base_optimizer.zero_grad(set_to_none=True)
        outputs = self.model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # ----- ascent: w ← w + eps -----
        epsilons = self._ascent(rho)

        # ----- second forward / backward at w+eps -----
        self.base_optimizer.zero_grad(set_to_none=True)
        outputs_pert = self.model(inputs)
        loss_pert = criterion(outputs_pert, labels)
        loss_pert.backward()

        # ----- restore w, then apply base optimizer with grads from w+eps -----
        self._restore(epsilons)
        self.base_optimizer.step()
        self.base_optimizer.zero_grad(set_to_none=True)

        return float(loss.detach())
