# =============================================================================
# NoOpOptimizer — freezes all model weights during gradient extraction
# =============================================================================
# The model runs forward + backward normally (so gradients are computed),
# but step() does nothing, so weights are never updated.
# =============================================================================
from torch.optim import Optimizer
from mmdet3d.registry import OPTIMIZERS


@OPTIMIZERS.register_module()
class NoOpOptimizer(Optimizer):
    """Optimizer that computes gradients but never updates weights."""

    def __init__(self, params, lr=0.0, **kwargs):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        # No weight updates — gradients remain for extraction
        return loss