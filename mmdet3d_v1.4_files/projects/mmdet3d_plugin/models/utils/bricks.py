"""Stub module for BEVFormer compatibility.

These utilities were part of the original BEVFormer implementation
but are not actually used in the inference pipeline for mmdet3d 1.4.
"""


def run_time(func=None):
    """No-op decorator — original used for timing in mmcv v1.x.

    In mmcv v2.x + mmengine, timing is handled by IterTimerHook,
    so this decorator is kept as a no-op for import compatibility.
    """
    if func is None:
        return lambda f: f
    return func