#!/usr/bin/env python3
# convert_pillarnest_ckpt_to_mmdet14.py
import argparse
from collections import OrderedDict
import os
import sys
import torch


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model", "ema_state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key], key
        if all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt, None
    raise RuntimeError("Could not find a valid state_dict in checkpoint.")


def normalize_key(k: str) -> str:
    # strip wrappers
    for p in ("module.", "model.", "net.", "detector."):
        if k.startswith(p):
            k = k[len(p):]

    # already modern
    if k.startswith("pts_"):
        return k

    # common old names -> mmdet3d v1.4 names
    prefix_map = {
        "voxel_encoder.": "pts_voxel_encoder.",
        "middle_encoder.": "pts_middle_encoder.",
        "backbone.": "pts_backbone.",
        "neck.": "pts_neck.",
        "bbox_head.": "pts_bbox_head.",
        "reader.": "pts_voxel_encoder.",  # some old repos used reader
    }
    for old, new in prefix_map.items():
        if k.startswith(old):
            return new + k[len(old):]

    return k


def maybe_build_target_state_dict(config_path):
    if not config_path:
        return None

    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmdet3d.registry import MODELS

    cfg = Config.fromfile(config_path)
    init_default_scope("mmdet3d")
    model = MODELS.build(cfg.model)
    return model.state_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_ckpt", required=True)
    parser.add_argument("--out_ckpt", required=True)
    parser.add_argument("--config", default=None,
                        help="Optional config to validate key/shape compatibility.")
    parser.add_argument("--drop_shape_mismatch", action="store_true",
                        help="If set with --config, keep only shape-compatible params.")
    args = parser.parse_args()

    ckpt = safe_torch_load(args.in_ckpt)
    src_sd, src_field = extract_state_dict(ckpt)

    converted = OrderedDict()
    collisions = []
    renamed = 0

    for k, v in src_sd.items():
        nk = normalize_key(k)
        if nk != k:
            renamed += 1
        if nk in converted and converted[nk].shape != v.shape:
            collisions.append((k, nk, tuple(v.shape), tuple(converted[nk].shape)))
            continue
        converted[nk] = v

    print(f"Loaded: {args.in_ckpt}")
    print(f"Source keys: {len(src_sd)}")
    print(f"Converted keys: {len(converted)}")
    print(f"Renamed keys: {renamed}")
    if collisions:
        print(f"Collisions (skipped): {len(collisions)}")
        for c in collisions[:10]:
            print("  ", c)

    target_sd = None
    if args.config:
        try:
            target_sd = maybe_build_target_state_dict(args.config)
        except Exception as e:
            print(f"[WARN] Could not build model from config: {e}")
            target_sd = None

    if target_sd is not None:
        compatible = OrderedDict()
        shape_mismatch = []
        unexpected = []

        for k, v in converted.items():
            if k not in target_sd:
                unexpected.append(k)
                continue
            if tuple(v.shape) != tuple(target_sd[k].shape):
                shape_mismatch.append((k, tuple(v.shape), tuple(target_sd[k].shape)))
                continue
            compatible[k] = v

        missing = [k for k in target_sd.keys() if k not in compatible]

        print(f"\nConfig check against: {args.config}")
        print(f"Target model params: {len(target_sd)}")
        print(f"Compatible params: {len(compatible)}")
        print(f"Unexpected keys: {len(unexpected)}")
        print(f"Shape mismatches: {len(shape_mismatch)}")
        print(f"Missing keys: {len(missing)}")

        if shape_mismatch:
            print("First 20 shape mismatches:")
            for k, s1, s2 in shape_mismatch[:20]:
                print(f"  {k}: ckpt{s1} vs model{s2}")

        if args.drop_shape_mismatch:
            converted = compatible
            print("Using only compatible params for output checkpoint.")

    # write output checkpoint
    if isinstance(ckpt, dict):
        out = dict(ckpt)
        if src_field is None:
            out = {"state_dict": converted, "meta": {"converted_from": os.path.basename(args.in_ckpt)}}
        else:
            out[src_field] = converted
            meta = out.get("meta", {})
            if not isinstance(meta, dict):
                meta = {}
            meta["converted_from"] = os.path.basename(args.in_ckpt)
            meta["converted_script"] = "convert_pillarnest_ckpt_to_mmdet14.py"
            out["meta"] = meta
    else:
        out = {"state_dict": converted, "meta": {"converted_from": os.path.basename(args.in_ckpt)}}

    torch.save(out, args.out_ckpt)
    print(f"\nSaved: {args.out_ckpt}")


if __name__ == "__main__":
    main()



