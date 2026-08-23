import os
import traceback
from typing import Union

import torch
import torch.nn.functional as F

from mmengine.hooks import Hook
from mmengine.hooks import Hook as MMHook
from mmengine.runner import Runner
from mmdet3d.registry import HOOKS


class _GradientCapturer:
    """Internal class to capture gradients during backward pass."""

    def __init__(self):
        self.gradient = None
        self._printed_once = False

    def __call__(self, module, grad_input, grad_output):
        if not self._printed_once:
            #print(f"[GradDebug] grad_output entries: {len(grad_output)}", flush=True)
            for i, g in enumerate(grad_output):
                if g is None:
                    print(f"[GradDebug] grad_output[{i}] = None", flush=True)
                # elif torch.is_tensor(g):
                #     print(
                #         f"[GradDebug] grad_output[{i}] shape={tuple(g.shape)} "
                #         f"dtype={g.dtype} device={g.device} "
                #         f"abs_mean={g.abs().mean().item():.4e} abs_max={g.abs().max().item():.4e}",
                #         flush=True,
                #     )
                # else:
                #     print(f"[GradDebug] grad_output[{i}] type={type(g)}", flush=True)
            self._printed_once = True

        valid = [g for g in grad_output if torch.is_tensor(g)]
        if not valid:
            self.gradient = None
            return

        if len(valid) == 1:
            self.gradient = valid[0].detach().clone()
            return

        try:
            self.gradient = torch.cat([g.detach() for g in valid], dim=1).clone()
        except RuntimeError:
            print("[GradDebug] cat failed, fallback to first grad tensor", flush=True)
            self.gradient = valid[0].detach().clone()


def _extract_sample_id(meta, batch_idx):
    lidar_path = meta.get("lidar_path", None) or meta.get("pts_filename", None)

    if lidar_path is None:
        lp = meta.get("lidar_points", {})
        if isinstance(lp, dict):
            lidar_path = lp.get("lidar_path", None)

    if lidar_path is None:
        token = meta.get("token", meta.get("sample_idx", None))
        if token is not None:
            return str(token)
        return f"sample_{batch_idx}"

    base_name = os.path.basename(lidar_path)
    for ext in [".pcd.bin", ".bin", ".npy"]:
        if base_name.endswith(ext):
            base_name = base_name[: -len(ext)]
            break
    return base_name


@HOOKS.register_module()
class FocalFormerGradientHook(Hook):
    def __init__(
        self,
        target_layer: str = "neck",
        save_path: str = "./gradients",
        normalize: bool = True,
        save_interval: int = 100,
        save_activations: bool = False,
    ):
        super().__init__()
        self.target_layer = target_layer
        self.save_path = save_path
        self.normalize = normalize
        self.save_interval = save_interval
        self.save_activations = save_activations

        self._capturer = _GradientCapturer()
        self._handle = None
        self._fwd_handle = None
        self._debug_fwd_handle = None
        self._activation = None
        self._debug_forward_printed = False

        self._module_map = {
            "backbone_block0": "pts_backbone.blocks.0",
            "neck": "pts_neck",
        }

    def _resolve_module_name(self):
        return self._module_map.get(self.target_layer, self.target_layer)

    # def _fwd_hook(self, module, inputs, output):
    #     if isinstance(output, (list, tuple)):
    #         tensors = [o.detach().clone() for o in output if torch.is_tensor(o)]
    #         if len(tensors) == 1:
    #             self._activation = tensors[0]
    #         elif len(tensors) > 1:
    #             try:
    #                 self._activation = torch.cat(tensors, dim=1)
    #             except RuntimeError:
    #                 self._activation = tensors[0]
    #     elif torch.is_tensor(output):
    #         self._activation = output.detach().clone()

    def _debug_fwd_hook(self, module, inputs, output):
        if self._debug_forward_printed:
            return
        outs = output if isinstance(output, (list, tuple)) else [output]
        #for i, o in enumerate(outs):
            # if torch.is_tensor(o):
            #     print(
            #         f"[GradDebug] fwd out[{i}] shape={tuple(o.shape)} "
            #         f"requires_grad={o.requires_grad} grad_fn={type(o.grad_fn).__name__ if o.grad_fn else None}",
            #         flush=True
            #     )
            # else:
            #     print(f"[GradDebug] fwd out[{i}] type={type(o)}", flush=True)
        self._debug_forward_printed = True
        if isinstance(output, (list, tuple)):
            shapes = [tuple(o.shape) if torch.is_tensor(o) else type(o).__name__ for o in output]
            #print(f"[GradDebug] target forward fired, outputs={shapes}", flush=True)
        # elif torch.is_tensor(output):
        #     print(f"[GradDebug] target forward fired, output={tuple(output.shape)}", flush=True)
        # else:
        #     print(f"[GradDebug] target forward fired, output type={type(output)}", flush=True)
        self._debug_forward_printed = True

    def before_train(self, runner: Runner) -> None:
        os.makedirs(self.save_path, exist_ok=True)
        model = runner.model
        module_name = self._resolve_module_name()

        print(f"\n{'=' * 60}", flush=True)
        print("FocalFormerGradientHook", flush=True)
        print(f"  Target: '{self.target_layer}' -> '{module_name}'", flush=True)
        print(f"  Save to: {self.save_path}", flush=True)
        print(f"{'=' * 60}", flush=True)

        target_module = None
        for name, module in model.named_modules():
            if name == module_name:
                target_module = module
                print(f"  ✓ Found module: {name} ({type(module).__name__})", flush=True)
                break

        if target_module is None:
            print(f"  ✗ Module '{module_name}' not found", flush=True)
            raise RuntimeError(f"Module '{module_name}' not found")

        self._handle = target_module.register_full_backward_hook(self._capturer)
        # in before_train(...)
        self._debug_fwd_handle = target_module.register_forward_hook(self._debug_fwd_hook)
        self._fwd_handle = target_module.register_forward_hook(self._fwd_hook)  # ALWAYS register
        print("  ✓ Tensor-grad forward hook registered", flush=True)

        # optional: keep module backward hook for non-list-output layers only
        if self.target_layer != "neck":
            self._handle = target_module.register_full_backward_hook(self._capturer)
            print("  ✓ Backward hook registered", flush=True)

        print("  ✓ Backward hook registered", flush=True)
        print("  ✓ Debug forward hook registered", flush=True)
        if self.save_activations:
            print("  ✓ Activation forward hook registered", flush=True)

        #print(f"[GradDebug] hook file: {__file__}", flush=True)
        # print(
        #     "[GradDebug] after_train_iter overridden? "
        #     f"{self.__class__.after_train_iter is not MMHook.after_train_iter}",
        #     flush=True,
        # )
        # print(f"[GradDebug] bound after_train_iter: {self.after_train_iter.__qualname__}", flush=True)

        m = runner.model.module if hasattr(runner.model, "module") else runner.model
        for n in ["pts_middle_encoder", "pts_backbone", "pts_neck", "imgpts_neck", "pts_bbox_head"]:
            mod = getattr(m, n, None)
            if mod is None:
                continue
            total = sum(1 for _ in mod.parameters())
            trainable = sum(1 for p in mod.parameters() if p.requires_grad)
            #print(f"[GradDebug] {n}: trainable {trainable}/{total}", flush=True)


    def before_train_iter(self, runner, batch_idx, data_batch=None):
        if batch_idx < 3 or batch_idx % self.save_interval == 0:
            print(f"[GradDebug] before_train_iter idx={batch_idx}", flush=True)

    def after_train_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: Union[dict, tuple] = None,
        outputs: Union[dict, tuple] = None,
    ) -> None:
        if batch_idx < 3 or batch_idx % self.save_interval == 0:
            print(f"[GradDebug] ENTER after_train_iter idx={batch_idx}", flush=True)
            print(f"[GradDebug] capturer_has_grad={self._capturer.gradient is not None}", flush=True)

            if isinstance(outputs, dict):
                print(f"[GradDebug] outputs keys={list(outputs.keys())}", flush=True)
                loss = outputs.get("loss", None)
                if torch.is_tensor(loss):
                    print(
                        f"[GradDebug] loss requires_grad={loss.requires_grad}, "
                        f"grad_fn={type(loss.grad_fn).__name__ if loss.grad_fn else None}",
                        flush=True,
                    )

        if self._capturer.gradient is None:
            return

        try:
            grad = self._capturer.gradient.detach().cpu().to(torch.float32)
            if self.normalize:
                grad = F.normalize(grad, p=2.0, dim=1)
        except Exception:
            print("[GradDebug] Exception inside after_train_iter", flush=True)
            traceback.print_exc()
            raise

        data_samples = data_batch.get("data_samples", []) if isinstance(data_batch, dict) else []
        if len(data_samples) == 0:
            print("[GradDebug] no data_samples in batch", flush=True)
            self._capturer.gradient = None
            self._activation = None
            return

        meta = data_samples[0].metainfo
        sample_id = _extract_sample_id(meta, batch_idx)

        out_path = os.path.join(self.save_path, f"{sample_id}_grad.pt")
        torch.save(grad, out_path)
        #print(f"[GradDebug] saved {out_path}", flush=True)

        if self.save_activations and self._activation is not None:
            act = self._activation.detach().cpu().to(torch.float32)
            act_path = os.path.join(self.save_path, f"{sample_id}_act.pt")
            torch.save(act, act_path)
            #print(f"[GradDebug] saved {act_path}", flush=True)

        self._capturer.gradient = None
        self._activation = None

    def _fwd_hook(self, module, inputs, output):
        self._activation = None
        tensors = output if isinstance(output, (list, tuple)) else [output]
        grads = []

        for i, t in enumerate(tensors):
            if torch.is_tensor(t):
                if self._activation is None:
                    self._activation = t.detach().clone()
                else:
                    try:
                        self._activation = torch.cat([self._activation, t.detach().clone()], dim=1)
                    except RuntimeError:
                        pass

                if t.requires_grad:
                    t.register_hook(lambda g, idx=i: self._save_tensor_grad(g, idx))

    def _save_tensor_grad(self, g, idx):
        #print(f"[GradDebug] tensor grad hook fired idx={idx}, shape={tuple(g.shape)}", flush=True)
        g = g.detach().clone()
        if self._capturer.gradient is None:
            self._capturer.gradient = g
        else:
            try:
                self._capturer.gradient = torch.cat([self._capturer.gradient, g], dim=1)
            except RuntimeError:
                pass


    def after_train(self, runner: Runner) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
            self._fwd_handle = None
        if self._debug_fwd_handle is not None:
            self._debug_fwd_handle.remove()
            self._debug_fwd_handle = None

        total = 0
        if os.path.isdir(self.save_path):
            total = len([f for f in os.listdir(self.save_path) if f.endswith("_grad.pt")])
        print(f"\n✓ Gradient extraction complete. {total} gradient files in {self.save_path}", flush=True)























# # =============================================================================
# # Gradient Extraction Hook for FocalFormer3D (NuScenes + Waymo)
# # =============================================================================
# #
# # FocalFormer3D architecture (same for both datasets):
# #   pts_voxel_layer (Voxelization) -> voxels
# #   pts_voxel_encoder (HardSimpleVFE) -> 3D sparse features
# #   pts_middle_encoder (SparseEncoder) -> 3D sparse -> 2D dense
# #   pts_backbone (SECOND) -> 2D feature maps
# #   pts_neck (SECONDFPN) -> multi-scale BEV features
# #   imgpts_neck (FocalEncoder) -> fused BEV features
# #   pts_bbox_head (FocalDecoder) -> detection outputs
# #
# # Feature map sizes differ by dataset due to different voxel/range configs:
# #   NuScenes: range [-54, 54] voxel 0.075 -> 1440 grid -> /8 = 180
# #     backbone_block0: [B, 128, 180, 180]
# #     neck:            [B, 512, 180, 180]
# #   Waymo: range [-75.2, 75.2] voxel 0.1 -> 1504 grid -> /8 = 188
# #     backbone_block0: [B, 128, 188, 188]
# #     neck:            [B, 512, 188, 188]
# #
# # Supported targets:
# #   'backbone_block0' -> pts_backbone.blocks.0
# #   'neck'            -> pts_neck
# #
# # =============================================================================
# import os
# import torch
# import torch.nn.functional as F
# from typing import Sequence, Union

# from mmengine.hooks import Hook
# from mmengine.runner import Runner
# from mmdet3d.registry import HOOKS


# class _GradientCapturer:
#     """Internal class to capture gradients during backward pass."""

#     def __init__(self):
#         self.gradient = None

#     def __call__(
#         self,
#         module: torch.nn.Module,
#         grad_input: Sequence[torch.Tensor],
#         grad_output: Sequence[torch.Tensor],
#     ) -> None:
#         if grad_output[0] is not None:
#             self.gradient = grad_output[0].detach().clone()


# def _extract_sample_id(meta, batch_idx):
#     """Extract a unique sample identifier from metadata.

#     Handles both NuScenes and Waymo path formats:
#       NuScenes: .../n008-2018-...__LIDAR_TOP__1533151603547590.pcd.bin
#       Waymo:    .../training/velodyne/0000000.bin

#     Also handles mmdet3d 1.4 metadata where the path may be under
#     different keys depending on dataset version.
#     """
#     # Try standard keys
#     lidar_path = (
#         meta.get('lidar_path', None)
#         or meta.get('pts_filename', None)
#     )

#     # mmdet3d 1.4 sometimes nests under lidar_points
#     if lidar_path is None:
#         lp = meta.get('lidar_points', {})
#         if isinstance(lp, dict):
#             lidar_path = lp.get('lidar_path', None)

#     # Try sample token (NuScenes) or sample_idx (Waymo)
#     if lidar_path is None:
#         token = meta.get('token', meta.get('sample_idx', None))
#         if token is not None:
#             return str(token)
#         return f'sample_{batch_idx}'

#     base_name = os.path.basename(lidar_path)
#     # Strip common extensions
#     for ext in ['.pcd.bin', '.bin', '.npy']:
#         if base_name.endswith(ext):
#             base_name = base_name[:-len(ext)]
#             break
#     return base_name


# @HOOKS.register_module()
# class FocalFormerGradientHook(Hook):
#     """
#     Hook to capture gradients from FocalFormer3D for adversarial attacks.

#     Extracts dL/df at the specified layer, normalizes, and saves per-sample.
#     Works with both NuScenes and Waymo datasets.

#     Args:
#         target_layer (str): Which layer to hook.
#             'backbone_block0' -> pts_backbone.blocks.0
#             'neck'            -> pts_neck
#         save_path (str): Directory to save gradient .pt files.
#         normalize (bool): Whether to L2-normalize gradients across channels.
#         save_interval (int): Log saving info every N iterations.
#         save_activations (bool): Also save forward activations alongside
#             gradients. Useful for Grad-CAM style analysis.
#     """

#     def __init__(
#         self,
#         target_layer: str = 'neck',
#         save_path: str = './gradients',
#         normalize: bool = True,
#         save_interval: int = 100,
#         save_activations: bool = False,
#     ):
#         super().__init__()
#         self.target_layer = target_layer
#         self.save_path = save_path
#         self.normalize = normalize
#         self.save_interval = save_interval
#         self.save_activations = save_activations
#         self._capturer = _GradientCapturer()
#         self._handle = None
#         self._fwd_handle = None
#         self._activation = None

#         self._module_map = {
#             'backbone_block0': 'pts_backbone.blocks.0',
#             'neck': 'pts_neck',
#         }

#     def _resolve_module_name(self):
#         if self.target_layer in self._module_map:
#             return self._module_map[self.target_layer]
#         return self.target_layer

#     def _fwd_hook(self, module, input, output):
#         """Capture forward activations if requested."""
#         if isinstance(output, (list, tuple)):
#             # SECONDFPN returns a list; concatenate for consistency
#             self._activation = torch.cat(
#                 [o.detach().clone() for o in output], dim=1)
#         elif isinstance(output, torch.Tensor):
#             self._activation = output.detach().clone()

#     def before_train(self, runner: Runner) -> None:
#         os.makedirs(self.save_path, exist_ok=True)
#         model = runner.model
#         module_name = self._resolve_module_name()

#         # Detect dataset from model config
#         dataset = 'unknown'
#         try:
#             test_cfg = model.test_cfg or {}
#             pts_cfg = test_cfg.get('pts', {})
#             dataset = pts_cfg.get('dataset', 'unknown')
#         except (AttributeError, TypeError):
#             pass

#         print(f"\n{'=' * 60}")
#         print(f"FocalFormerGradientHook")
#         print(f"  Dataset:  {dataset}")
#         print(f"  Target:   '{self.target_layer}' -> module '{module_name}'")
#         print(f"  Save to:  {self.save_path}")
#         print(f"  Normalize: {self.normalize}")
#         print(f"  Save activations: {self.save_activations}")
#         print(f"{'=' * 60}")

#         target_module = None
#         for name, module in model.named_modules():
#             if name == module_name:
#                 target_module = module
#                 print(f"  ✓ Found module: {name} ({type(module).__name__})")
#                 break

#         if target_module is None:
#             print(f"\n  ✗ Module '{module_name}' not found!")
#             print(f"\n  Available 'pts' modules:")
#             for name, module in model.named_modules():
#                 if 'pts' in name and len(name.split('.')) <= 3:
#                     print(f"    - {name}: {type(module).__name__}")
#             raise RuntimeError(
#                 f"Module '{module_name}' not found. "
#                 f"Use one of: {list(self._module_map.keys())} "
#                 f"or pass a raw module path.")

#         self._handle = target_module.register_full_backward_hook(self._capturer)
#         print(f"  ✓ Backward hook registered")

#         if self.save_activations:
#             self._fwd_handle = target_module.register_forward_hook(self._fwd_hook)
#             print(f"  ✓ Forward hook registered (activations)")

#         print(f"{'=' * 60}\n")

#     def after_train_iter(
#         self,
#         runner: Runner,
#         batch_idx: int,
#         data_batch: Union[dict, tuple],
#         outputs: Union[dict, tuple],
#     ) -> None:
#         if self._capturer.gradient is None:
#             return

#         # Extract sample identifier from metadata
#         data_samples = data_batch.get('data_samples', [])
#         if len(data_samples) == 0:
#             self._capturer.gradient = None
#             self._activation = None
#             return

#         meta = data_samples[0].metainfo
#         sample_id = _extract_sample_id(meta, batch_idx)

#         # Process gradient
#         grad = self._capturer.gradient.cpu().to(torch.float32)

#         if self.normalize:
#             grad = F.normalize(grad, p=2.0, dim=1)

#         # Save gradient
#         out_path = os.path.join(self.save_path, f'{sample_id}_grad.pt')
#         torch.save(grad, out_path)

#         # Optionally save activations
#         if self.save_activations and self._activation is not None:
#             act = self._activation.cpu().to(torch.float32)
#             act_path = os.path.join(self.save_path, f'{sample_id}_act.pt')
#             torch.save(act, act_path)

#         if batch_idx % self.save_interval == 0:
#             print(f"  [iter {batch_idx}] Saved: {sample_id}_grad.pt "
#                   f"| shape: {list(grad.shape)}")

#         self._capturer.gradient = None
#         self._activation = None

#     def after_train(self, runner: Runner) -> None:
#         if self._handle is not None:
#             self._handle.remove()
#             self._handle = None
#         if self._fwd_handle is not None:
#             self._fwd_handle.remove()
#             self._fwd_handle = None
#         total = len([f for f in os.listdir(self.save_path) if f.endswith('_grad.pt')])
#         print(f"\n✓ Gradient extraction complete. {total} gradient files in {self.save_path}") 





