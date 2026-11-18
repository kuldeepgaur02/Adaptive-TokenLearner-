"""CPU replacement for mmdet3d's GPU rotated-box IoU, used by the KITTI metric.

Why this exists
---------------
``mmdet3d/evaluation/functional/kitti_utils/rotate_iou.py`` compiles a CUDA
kernel with numba at import time. In this environment that fails:

    numba.cuda.cudadrv.error.NvvmError: Failed to compile
    IR version 1.6 incompatible with current version 2.0
    NVVM_ERROR_IR_VERSION_MISMATCH

numba 0.56.4 emits NVVM IR 1.6 (CUDA 11), but the libnvvm it finds is CUDA 12
and expects IR 2.0. The proper fix is to upgrade numba; until that is approved,
this module substitutes a numpy + shapely implementation of the one function
that needs the GPU.

Everything else in the KITTI metric is untouched: difficulty levels, the
ignore rules, the 11- and 40-point AP computation all remain mmdet3d's own
code. Only the pairwise box-overlap primitive is replaced, and it reproduces
the CUDA kernel's conventions exactly (see ``_corners``).

Because the real module raises on import, a stub is placed in ``sys.modules``
BEFORE anything tries to import it. ``from .rotate_iou import
rotate_iou_gpu_eval`` then finds this stub and never executes the real file.

How it is loaded
----------------
Add to the config's custom_imports:

    'projects.myfusion.cpu_rotate_iou',

It must come before anything that triggers the KITTI metric.

Speed
-----
A rotated-polygon intersection is ~50 us in shapely, and the evaluator asks for
the full cross product of a partition's detections and ground truth, which runs
to tens of millions of pairs. Almost none of them touch, so pairs are first
filtered by their axis-aligned bounding boxes, which is vectorised and cheap.
Only the survivors reach shapely. In practice this keeps the whole validation
set to under a couple of minutes.
"""

from __future__ import annotations

import logging
import sys
import types

import numpy as np

logger = logging.getLogger(__name__)

_TARGET = "mmdet3d.evaluation.functional.kitti_utils.rotate_iou"
_PARENT = "mmdet3d.evaluation.functional.kitti_utils"

# Rows are processed in blocks so the [N, K] boolean overlap mask never gets
# large, however big a partition the evaluator hands us.
_BLOCK = 2048

try:
    from shapely.geometry import Polygon
    _HAS_SHAPELY = True
except ImportError:  # pragma: no cover
    _HAS_SHAPELY = False
    Polygon = None


def _corners(boxes: np.ndarray) -> np.ndarray:
    """Corner coordinates of rotated boxes.

    Args:
        boxes: [N, 5] as (center_x, center_y, x_size, y_size, angle)
    Returns:
        [N, 4, 2]

    This mirrors ``rbbox_to_corners`` in the CUDA kernel exactly, including the
    corner ORDER and the sign of the rotation (positive angle = clockwise):

        corner_x = ( cos * local_x + sin * local_y) + center_x
        corner_y = (-sin * local_x + cos * local_y) + center_y

    Getting the sign wrong does not raise; it silently produces IoUs that are
    plausible but wrong for every non-axis-aligned box, so it is worth being
    literal here.
    """
    cx, cy = boxes[:, 0], boxes[:, 1]
    x_size, y_size = boxes[:, 2], boxes[:, 3]
    angle = boxes[:, 4]

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    local_x = np.stack([-x_size / 2, -x_size / 2, x_size / 2, x_size / 2], axis=1)
    local_y = np.stack([-y_size / 2, y_size / 2, y_size / 2, -y_size / 2], axis=1)

    corner_x = cos_a[:, None] * local_x + sin_a[:, None] * local_y + cx[:, None]
    corner_y = -sin_a[:, None] * local_x + cos_a[:, None] * local_y + cy[:, None]

    return np.stack([corner_x, corner_y], axis=2)


def _polygons(corners: np.ndarray):
    """One shapely polygon per box. A degenerate box becomes None."""
    out = []
    for quad in corners:
        try:
            poly = Polygon(quad)
            if not poly.is_valid:
                # A zero-size or self-intersecting annotation does occur in
                # KITTI; buffer(0) repairs it, and a repaired empty polygon
                # simply contributes no intersection.
                poly = poly.buffer(0)
            out.append(poly if poly.area > 0 else None)
        except Exception:
            out.append(None)
    return out


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    """Pairwise overlap between two sets of rotated BEV boxes, on the CPU.

    Drop-in replacement for the GPU version. Same signature, same return shape
    and dtype; ``device_id`` is accepted and ignored.

    Args:
        boxes:       [N, 5] (center_x, center_y, x_size, y_size, angle)
        query_boxes: [K, 5]
        criterion:  -1 -> intersection / union           (the usual IoU)
                     0 -> intersection / area(boxes)
                     1 -> intersection / area(query_boxes)
                     anything else -> raw intersection area. d3_box_overlap
                     passes 2 to get the BEV intersection, which it then
                     multiplies by the height overlap.
    Returns:
        [N, K] in the dtype of ``boxes``
    """
    if not _HAS_SHAPELY:
        raise RuntimeError(
            "cpu_rotate_iou needs shapely, which is not importable. Either "
            "install it or get numba upgraded so the original GPU kernel works."
        )

    in_dtype = boxes.dtype
    boxes = np.ascontiguousarray(boxes, dtype=np.float64)
    query_boxes = np.ascontiguousarray(query_boxes, dtype=np.float64)

    num_boxes, num_queries = boxes.shape[0], query_boxes.shape[0]
    result = np.zeros((num_boxes, num_queries), dtype=np.float32)
    if num_boxes == 0 or num_queries == 0:
        return result.astype(in_dtype)

    box_corners = _corners(boxes)
    query_corners = _corners(query_boxes)

    # Axis-aligned bounds, used only to throw away pairs that cannot touch.
    b_x0, b_x1 = box_corners[..., 0].min(1), box_corners[..., 0].max(1)
    b_y0, b_y1 = box_corners[..., 1].min(1), box_corners[..., 1].max(1)
    q_x0, q_x1 = query_corners[..., 0].min(1), query_corners[..., 0].max(1)
    q_y0, q_y1 = query_corners[..., 1].min(1), query_corners[..., 1].max(1)

    box_polys = _polygons(box_corners)
    query_polys = _polygons(query_corners)

    # The kernel uses the nominal area (x_size * y_size), not the polygon's,
    # so the two implementations agree even on degenerate boxes.
    box_area = boxes[:, 2] * boxes[:, 3]
    query_area = query_boxes[:, 2] * query_boxes[:, 3]

    tested = 0
    for start in range(0, num_boxes, _BLOCK):
        stop = min(start + _BLOCK, num_boxes)

        candidate = (
            (b_x0[start:stop, None] < q_x1[None, :])
            & (b_x1[start:stop, None] > q_x0[None, :])
            & (b_y0[start:stop, None] < q_y1[None, :])
            & (b_y1[start:stop, None] > q_y0[None, :])
        )
        rows, cols = np.nonzero(candidate)
        tested += rows.size

        for row, col in zip(rows, cols):
            index = start + int(row)
            poly_b = box_polys[index]
            poly_q = query_polys[int(col)]
            if poly_b is None or poly_q is None:
                continue

            try:
                inter = poly_b.intersection(poly_q).area
            except Exception:
                # Shapely very occasionally fails on near-degenerate geometry.
                # Treating it as no overlap is the safe direction: it can only
                # lower a score, never invent a detection.
                continue
            if inter <= 0.0:
                continue

            if criterion == -1:
                denom = box_area[index] + query_area[int(col)] - inter
                value = inter / denom if denom > 0 else 0.0
            elif criterion == 0:
                denom = box_area[index]
                value = inter / denom if denom > 0 else 0.0
            elif criterion == 1:
                denom = query_area[int(col)]
                value = inter / denom if denom > 0 else 0.0
            else:
                value = inter

            result[index, int(col)] = value

    logger.debug(
        "cpu_rotate_iou: %dx%d pairs, %d survived the bounding-box filter (%.2f%%)",
        num_boxes, num_queries, tested,
        100.0 * tested / max(num_boxes * num_queries, 1),
    )
    return result.astype(in_dtype)


def _install() -> None:
    """Put this implementation in place of the module that will not import."""
    existing = sys.modules.get(_TARGET)
    if existing is not None and getattr(existing, "_is_cpu_fallback", False):
        return                                  # already installed

    stub = types.ModuleType(_TARGET)
    stub.__file__ = __file__
    stub._is_cpu_fallback = True
    stub.rotate_iou_gpu_eval = rotate_iou_gpu_eval
    sys.modules[_TARGET] = stub

    # If the parent package is already imported, `from .rotate_iou import ...`
    # may read the attribute off it rather than going through sys.modules.
    parent = sys.modules.get(_PARENT)
    if parent is not None:
        parent.rotate_iou = stub

    logger.info(
        "cpu_rotate_iou installed: KITTI evaluation will use the CPU "
        "rotated-IoU implementation instead of the numba CUDA kernel."
    )


_install()