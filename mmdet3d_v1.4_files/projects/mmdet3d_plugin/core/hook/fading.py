# =============================================================================
# fading_hook.py — refactored for mmdet3d >= 1.1 / v1.4.x (mmengine)
# =============================================================================
# Changes from old version:
#   - HOOKS from mmcv.runner.hooks   → HOOKS from mmengine.registry
#   - Hook  from mmcv.runner.hooks   → Hook  from mmengine.hooks
#   - runner.data_loader             → runner.train_dataloader
#   - runner.epoch                   → runner.epoch  (unchanged)
#   - dataset.dataset.pipeline.transforms → dataset.pipeline.transforms
#     (mmengine wraps differently; may need .dataset depending on sampler)
#
# NOTE: In mmdet3d v1.4, the recommended approach for fading ObjectSample
# is to use DisableObjectSampleHook (built-in). This custom hook is provided
# for cases where you need custom fading logic beyond what the built-in
# hook supports.
# =============================================================================
import logging

logger = logging.getLogger(__name__)
logger.info('[fading_hook] Loading module...')

# --- Registry import ---
try:
    from mmengine.registry import HOOKS
    logger.info('[fading_hook] ✓ Imported HOOKS from mmengine.registry')
except ImportError as e:
    logger.error(f'[fading_hook] ✗ Failed to import HOOKS '
                 f'from mmengine.registry: {e}')
    raise

# --- Base Hook import ---
try:
    from mmengine.hooks import Hook
    logger.info('[fading_hook] ✓ Imported Hook from mmengine.hooks')
except ImportError as e:
    logger.error(f'[fading_hook] ✗ Failed to import Hook '
                 f'from mmengine.hooks: {e}')
    logger.error('  → Make sure mmengine >= 0.7.0 is installed')
    raise


def _find_dataset(dataloader):
    """Walk through wrapper layers to find the actual dataset object.

    mmengine may wrap the dataset in ConcatDataset, RepeatDataset,
    CBGSDataset, etc. Each of these stores the inner dataset in
    .dataset — we drill down until we find one with .pipeline.
    """
    dataset = dataloader.dataset
    depth = 0
    while hasattr(dataset, 'dataset') and not hasattr(dataset, 'pipeline'):
        logger.debug(f'[Fading._find_dataset] Unwrapping layer {depth}: '
                     f'{type(dataset).__name__}')
        dataset = dataset.dataset
        depth += 1
        if depth > 10:
            logger.error('[Fading._find_dataset] ✗ Exceeded 10 wrapper '
                         'layers — likely infinite loop')
            break
    logger.debug(f'[Fading._find_dataset] Found dataset: '
                 f'{type(dataset).__name__} (depth={depth})')
    return dataset


@HOOKS.register_module()
class Fading(Hook):
    """Remove ObjectSample transform from the training pipeline after
    a specified epoch (fade_epoch).

    This prevents ground-truth sampling augmentation from being applied
    in later training epochs, which can improve final performance.

    Args:
        fade_epoch (int): Epoch at which to remove ObjectSample.
            Default: 100000 (effectively never).
    """

    def __init__(self, fade_epoch=100000):
        super().__init__()
        self.fade_epoch = fade_epoch
        self._removed = False
        logger.info(f'[Fading] ✓ Initialized with fade_epoch={fade_epoch}')

    def before_train_epoch(self, runner):
        current_epoch = runner.epoch
        logger.debug(f'[Fading.before_train_epoch] epoch={current_epoch}, '
                     f'fade_epoch={self.fade_epoch}, '
                     f'already_removed={self._removed}')

        if self._removed:
            return

        if current_epoch >= self.fade_epoch:
            logger.info(f'[Fading] epoch {current_epoch} >= '
                        f'fade_epoch {self.fade_epoch}, '
                        f'removing ObjectSample...')

            # --- Locate the dataset ---
            try:
                dataloader = runner.train_dataloader
                logger.debug(f'[Fading] train_dataloader type: '
                             f'{type(dataloader).__name__}')
            except AttributeError:
                logger.error('[Fading] ✗ runner has no train_dataloader — '
                             'is this an mmengine Runner?')
                logger.error(f'  → runner type: {type(runner).__name__}')
                logger.error(f'  → runner attrs: '
                             f'{[a for a in dir(runner) if "data" in a.lower()]}')
                return

            dataset = _find_dataset(dataloader)

            if not hasattr(dataset, 'pipeline'):
                logger.error(f'[Fading] ✗ dataset {type(dataset).__name__} '
                             f'has no .pipeline attribute')
                logger.error(f'  → dataset attrs: '
                             f'{[a for a in dir(dataset) if not a.startswith("_")]}')
                return

            # --- List current transforms for debugging ---
            transforms = dataset.pipeline.transforms
            logger.debug(f'[Fading] Current pipeline has '
                         f'{len(transforms)} transforms:')
            for idx, t in enumerate(transforms):
                logger.debug(f'  [{idx}] {type(t).__name__}')

            # --- Find and remove ObjectSample ---
            found = False
            for i, transform in enumerate(transforms):
                tname = type(transform).__name__
                if tname == 'ObjectSample':
                    transforms.pop(i)
                    self._removed = True
                    found = True
                    logger.info(f'[Fading] ✓ Removed ObjectSample at '
                                f'index {i} (epoch={current_epoch})')
                    break

            if not found:
                logger.warning('[Fading] ⚠ ObjectSample not found in '
                               'pipeline — nothing to remove')
                logger.warning('  Remaining transforms: '
                               f'{[type(t).__name__ for t in transforms]}')
                # Mark as removed so we don't search every epoch
                self._removed = True

            # --- Log final pipeline ---
            logger.debug(f'[Fading] Pipeline after removal '
                         f'({len(transforms)} transforms):')
            for idx, t in enumerate(transforms):
                logger.debug(f'  [{idx}] {type(t).__name__}')


logger.info('[fading_hook] ✓ Registered Fading to HOOKS')
logger.info('[fading_hook] ✓ Module fully loaded')