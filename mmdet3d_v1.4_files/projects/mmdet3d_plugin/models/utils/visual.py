"""Stub module for BEVFormer compatibility.

save_tensor was used for debugging visualization in the original
BEVFormer implementation but is not used in the mmdet3d 1.4 pipeline.
"""


def save_tensor(tensor, filepath, **kwargs):
    """No-op — original saved tensor visualizations for debugging.

    In the refactored mmdet3d 1.4 pipeline, visualization is handled
    by Det3DLocalVisualizer, so this function is kept as a no-op.
    """
    pass