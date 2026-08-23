# =============================================================================
# merge_augs.py — refactored for mmdet3d >= 1.1 / v1.4.x
# =============================================================================
# Changes from old version:
#   - nms_gpu, nms_normal_gpu       → nms_bev, nms_normal_bev
#     from mmdet3d.models.layers
#   - boxes_iou_bev                 → from mmcv.ops import boxes_iou_bev
#   - bbox3d2result, bbox3d_mapping_back, xywhr2xyxyr
#     from mmdet3d.core.bbox        → from mmdet3d.structures
#   - CameraInstance3DBoxes, DepthInstance3DBoxes, LiDARInstance3DBoxes
#     from mmdet3d.core.bbox        → from mmdet3d.structures
#   - box_np_ops: from mmdet3d.core.bbox → from mmdet3d.structures
#   - mmcv.mkdir_or_exist           → mmengine.utils.mkdir_or_exist
#   - import mmcv (general)         → import mmengine where needed
# =============================================================================
import logging
import pickle
import os

logger = logging.getLogger(__name__)
logger.info('[merge_augs] Loading module...')

import torch

# --- NMS functions ---
try:
    from mmdet3d.models.layers import nms_bev, nms_normal_bev
    logger.info('[merge_augs] ✓ Imported nms_bev, nms_normal_bev '
                'from mmdet3d.models.layers')
except ImportError as e:
    logger.error(f'[merge_augs] ✗ Failed to import nms_bev/nms_normal_bev: {e}')
    logger.error('  → These replaced nms_gpu/nms_normal_gpu from '
                 'mmdet3d.ops.iou3d.iou3d_utils')
    raise

# --- IoU BEV (CUDA op, still in mmcv.ops) ---
try:
    from mmcv.ops import boxes_iou_bev
    logger.info('[merge_augs] ✓ Imported boxes_iou_bev from mmcv.ops')
except ImportError as e:
    logger.error(f'[merge_augs] ✗ Failed to import boxes_iou_bev: {e}')
    logger.error('  → Make sure mmcv >= 2.0 is installed with CUDA ops')
    raise

# --- Structures (old: mmdet3d.core.bbox) ---
try:
    from mmdet3d.structures import (
        bbox3d2result,
        bbox3d_mapping_back,
        xywhr2xyxyr,
    )
    logger.info('[merge_augs] ✓ Imported bbox3d2result, bbox3d_mapping_back, '
                'xywhr2xyxyr from mmdet3d.structures')
except ImportError as e:
    logger.error(f'[merge_augs] ✗ Failed to import from mmdet3d.structures: {e}')
    logger.error('  → These moved from mmdet3d.core.bbox to mmdet3d.structures')
    raise

try:
    from mmdet3d.structures import (
        CameraInstance3DBoxes,
        DepthInstance3DBoxes,
        LiDARInstance3DBoxes,
    )
    logger.info('[merge_augs] ✓ Imported 3D box types from mmdet3d.structures')
except ImportError as e:
    logger.error(f'[merge_augs] ✗ Failed to import 3D box types: {e}')
    raise

# --- mkdir_or_exist (old: mmcv → new: mmengine.utils) ---
try:
    from mmengine.utils import mkdir_or_exist
    logger.info('[merge_augs] ✓ Imported mkdir_or_exist from mmengine.utils')
except ImportError as e:
    logger.error(f'[merge_augs] ✗ Failed to import mkdir_or_exist: {e}')
    logger.error('  → Moved from mmcv to mmengine.utils')
    raise


ensemble = False


def merge_aug_bboxes_3d(aug_results, img_metas, test_cfg):
    """Merge augmented detection 3D bboxes and scores.

    Args:
        aug_results (list[dict]): The dict of detection results.
            The dict contains the following keys

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.
        img_metas (list[dict]): Meta information of each sample.
        test_cfg (dict): Test config.

    Returns:
        dict: Bounding boxes results in cpu mode, containing merged results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Merged detection bbox.
            - scores_3d (torch.Tensor): Merged detection scores.
            - labels_3d (torch.Tensor): Merged predicted box labels.
    """
    logger.debug(f'[merge_aug_bboxes_3d] Called with '
                 f'aug_results={"None" if aug_results is None else len(aug_results)} items, '
                 f'img_metas={len(img_metas)} items, '
                 f'ensemble={ensemble}')

    if ensemble:
        import glob
        ensemble_folder = './merge_augs/*'
        aug_bboxes = []
        aug_bboxes_for_nms = []
        aug_scores = []
        aug_labels = []

        ensemble_models = glob.glob(ensemble_folder)
        logger.debug(f'[merge_aug_bboxes_3d] Ensemble mode: '
                     f'found {len(ensemble_models)} models in {ensemble_folder}')

        for ensemble_model in ensemble_models:
            pkl_path = (f'{ensemble_model}/'
                        f'sampleidx_{img_metas[0][0]["sample_idx"]}.pkl')
            logger.debug(f'[merge_aug_bboxes_3d] Loading ensemble pkl: '
                         f'{pkl_path}')
            with open(pkl_path, 'rb') as f:
                temp = pickle.load(f)

            aug_bboxes.append(torch.as_tensor(
                temp['aug_bboxes'], dtype=torch.float32, device='cuda'))
            aug_bboxes_for_nms.append(torch.as_tensor(
                temp['aug_bboxes_for_nms'], dtype=torch.float32, device='cuda'))
            aug_scores.append(torch.as_tensor(
                temp['aug_scores'], dtype=torch.float32, device='cuda'))
            aug_labels.append(torch.as_tensor(
                temp['aug_labels'], dtype=torch.int32, device='cuda'))

        aug_bboxes = torch.cat(aug_bboxes, dim=0)
        aug_bboxes_for_nms = torch.cat(aug_bboxes_for_nms, dim=0)
        aug_scores = torch.cat(aug_scores, dim=0)
        aug_labels = torch.cat(aug_labels, dim=0)

        logger.debug(f'[merge_aug_bboxes_3d] Ensemble merged: '
                     f'bboxes={aug_bboxes.shape}, scores={aug_scores.shape}')

        aug_bboxes = LiDARInstance3DBoxes(
            aug_bboxes, box_dim=aug_bboxes.shape[-1])
    else:
        if 'temp_result_folder' in test_cfg:
            temp_folder = './merge_augs/' + test_cfg.temp_result_folder
        else:
            temp_folder = './merge_augs_initial_results/'

        mkdir_or_exist(temp_folder)

        logger.info('------------------------------------')
        logger.info(f'Save to {temp_folder}')
        logger.info('------------------------------------')

        if aug_results is None:
            pkl_path = (f'{temp_folder}/'
                        f'sampleidx_{img_metas[0][0]["sample_idx"]}.pkl')
            logger.debug(f'[merge_aug_bboxes_3d] aug_results is None, '
                         f'loading from {pkl_path}')
            with open(pkl_path, 'rb') as f:
                temp = pickle.load(f)

            aug_bboxes = torch.as_tensor(
                temp['aug_bboxes'], dtype=torch.float32, device='cuda')
            aug_bboxes_for_nms = torch.as_tensor(
                temp['aug_bboxes_for_nms'], dtype=torch.float32, device='cuda')
            aug_scores = torch.as_tensor(
                temp['aug_scores'], dtype=torch.float32, device='cuda')
            aug_labels = torch.as_tensor(
                temp['aug_labels'], dtype=torch.int32, device='cuda')

            logger.debug(f'[merge_aug_bboxes_3d] Loaded from pkl: '
                         f'bboxes={aug_bboxes.shape}, '
                         f'scores={aug_scores.shape}')

            aug_bboxes = LiDARInstance3DBoxes(
                aug_bboxes, box_dim=aug_bboxes.shape[-1])
        else:
            assert len(aug_results) == len(img_metas), \
                '"aug_results" should have the same length as "img_metas", ' \
                f'got len(aug_results)={len(aug_results)} and ' \
                f'len(img_metas)={len(img_metas)}'

            recovered_bboxes = []
            recovered_scores = []
            recovered_labels = []

            for aug_idx, (bboxes, img_info) in enumerate(
                    zip(aug_results, img_metas)):
                scale_factor = img_info[0]['pcd_scale_factor']
                pcd_horizontal_flip = img_info[0]['pcd_horizontal_flip']
                pcd_vertical_flip = img_info[0]['pcd_vertical_flip']
                logger.debug(
                    f'[merge_aug_bboxes_3d] Aug[{aug_idx}]: '
                    f'scale={scale_factor}, '
                    f'h_flip={pcd_horizontal_flip}, '
                    f'v_flip={pcd_vertical_flip}, '
                    f'num_boxes={len(bboxes["scores_3d"])}')
                recovered_scores.append(bboxes['scores_3d'])
                recovered_labels.append(bboxes['labels_3d'])
                bboxes = bbox3d_mapping_back(
                    bboxes['boxes_3d'], scale_factor,
                    pcd_horizontal_flip, pcd_vertical_flip)
                recovered_bboxes.append(bboxes)

            aug_bboxes = recovered_bboxes[0].cat(recovered_bboxes)
            aug_bboxes_for_nms = xywhr2xyxyr(aug_bboxes.bev)
            aug_scores = torch.cat(recovered_scores, dim=0)
            aug_labels = torch.cat(recovered_labels, dim=0)

            logger.debug(f'[merge_aug_bboxes_3d] Recovered & merged: '
                         f'bboxes={aug_bboxes.tensor.shape}, '
                         f'scores={aug_scores.shape}, '
                         f'labels={aug_labels.shape}')

            if True:
                temp = dict()
                temp['aug_bboxes'] = aug_bboxes.tensor.cpu().numpy()
                temp['aug_bboxes_for_nms'] = aug_bboxes_for_nms.cpu().numpy()
                temp['aug_scores'] = aug_scores.cpu().numpy()
                temp['aug_labels'] = aug_labels.cpu().numpy()
                pkl_path = (f'{temp_folder}/'
                            f'sampleidx_{img_metas[0][0]["sample_idx"]}.pkl')
                logger.debug(f'[merge_aug_bboxes_3d] Saving to {pkl_path}')
                with open(pkl_path, 'wb') as f:
                    pickle.dump(temp, f)

    test_cfg = test_cfg.copy()
    # Extra NMS settings
    test_cfg['nms_type'] = 'rotate'
    test_cfg['use_rotate_nms'] = True
    test_cfg['max_num'] = 500
    test_cfg['nms_thr'] = 0.1
    test_cfg['score_threshold'] = 0.05

    logger.debug(f'[merge_aug_bboxes_3d] NMS config: '
                 f'use_rotate_nms={test_cfg["use_rotate_nms"]}, '
                 f'nms_thr={test_cfg["nms_thr"]}, '
                 f'max_num={test_cfg["max_num"]}, '
                 f'score_threshold={test_cfg["score_threshold"]}')

    # Select NMS function
    # OLD: nms_gpu / nms_normal_gpu
    # NEW: nms_bev / nms_normal_bev
    if test_cfg.use_rotate_nms:
        nms_func = nms_bev
        logger.debug('[merge_aug_bboxes_3d] Using nms_bev (rotate NMS)')
    else:
        nms_func = nms_normal_bev
        logger.debug('[merge_aug_bboxes_3d] Using nms_normal_bev')

    merged_bboxes = []
    merged_scores = []
    merged_labels = []

    # Apply multi-class NMS when merging bboxes
    if len(aug_labels) == 0:
        logger.debug('[merge_aug_bboxes_3d] No labels — returning empty result')
        return bbox3d2result(aug_bboxes, aug_scores, aug_labels)

    num_classes = torch.max(aug_labels).item() + 1
    logger.debug(f'[merge_aug_bboxes_3d] Applying per-class NMS over '
                 f'{num_classes} classes, {len(aug_labels)} total boxes')

    for class_id in range(num_classes):
        class_inds = (aug_labels == class_id)
        bboxes_i = aug_bboxes[class_inds]
        bboxes_nms_i = aug_bboxes_for_nms[class_inds, :]
        scores_i = aug_scores[class_inds]
        labels_i = aug_labels[class_inds]

        if len(bboxes_nms_i) == 0:
            logger.debug(f'[merge_aug_bboxes_3d] class_id={class_id}: '
                         f'0 boxes, skipping')
            continue

        logger.debug(f'[merge_aug_bboxes_3d] class_id={class_id}: '
                     f'{len(bboxes_nms_i)} boxes before NMS')

        selected = nms_func(bboxes_nms_i, scores_i, test_cfg.nms_thr)

        logger.debug(f'[merge_aug_bboxes_3d] class_id={class_id}: '
                     f'{len(selected)} boxes after NMS')

        if True:  # voting
            vote_iou_thresh = 0.65
            use_voting_scores = False
            logger.debug(f'[merge_aug_bboxes_3d] Voting: '
                         f'vote_iou_thresh={vote_iou_thresh}, '
                         f'use_voting_scores={use_voting_scores}')

            selected_bboxes = bboxes_i[selected, :]
            selected_scores = scores_i[selected]
            selected_labels = labels_i[selected]

            iou = boxes_iou_bev(
                xywhr2xyxyr(selected_bboxes.bev), bboxes_nms_i)
            logger.debug(f'[merge_aug_bboxes_3d] Voting IoU matrix: '
                         f'{iou.shape}, '
                         f'min={iou.min():.4f}, max={iou.max():.4f}')

            iou[iou < vote_iou_thresh] = 0.

            voted_bboxes = ((iou[:, :, None] *
                             bboxes_i.tensor[None]).sum(dim=1) /
                            (iou[:, :, None].sum(dim=1) + 1e-6))
            voted_bboxes[:, 6] = torch.atan2(
                (iou * torch.sin(
                    bboxes_i.tensor[None, :, 6])).sum(dim=1) /
                (iou.sum(dim=1) + 1e-6),
                (iou * torch.cos(
                    bboxes_i.tensor[None, :, 6])).sum(dim=1) /
                (iou.sum(dim=1) + 1e-6))

            voted_bboxes = LiDARInstance3DBoxes(
                voted_bboxes, box_dim=voted_bboxes.shape[-1])

            selected_bboxes = voted_bboxes
            if use_voting_scores:
                voted_scores = ((iou * scores_i[None]).sum(dim=1) /
                                iou.sum(dim=1))
                selected_scores = voted_scores

            logger.debug(f'[merge_aug_bboxes_3d] class_id={class_id}: '
                         f'{len(selected_bboxes)} boxes after voting')

        merged_bboxes.append(selected_bboxes)
        merged_scores.append(selected_scores)
        merged_labels.append(selected_labels)

    if len(merged_bboxes) == 0:
        logger.warning('[merge_aug_bboxes_3d] ⚠ No boxes survived NMS+voting')
        return bbox3d2result(aug_bboxes[:0], aug_scores[:0], aug_labels[:0])

    merged_bboxes = merged_bboxes[0].cat(merged_bboxes)
    merged_scores = torch.cat(merged_scores, dim=0)
    merged_labels = torch.cat(merged_labels, dim=0)

    _, order = merged_scores.sort(0, descending=True)
    num = min(test_cfg.max_num, len(aug_bboxes))
    order = order[:num]

    merged_bboxes = merged_bboxes[order]
    merged_scores = merged_scores[order]
    merged_labels = merged_labels[order]

    logger.debug(f'[merge_aug_bboxes_3d] Final result: '
                 f'{len(merged_bboxes)} boxes '
                 f'(max_num cap={test_cfg.max_num}), '
                 f'score range=[{merged_scores.min():.4f}, '
                 f'{merged_scores.max():.4f}]')

    return bbox3d2result(merged_bboxes, merged_scores, merged_labels)


logger.info('[merge_augs] ✓ Module fully loaded')
