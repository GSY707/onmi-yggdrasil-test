"""Test-process compatibility for the CPU torchvision wheel in this workspace.

The installed torchvision build tries to register its fake NMS kernel during
import, but the matching native operator is not present.  Transformers only
needs the import for optional image helpers in these tests, so declare the
operator signature before collection.  This does not affect V2-A runtime
semantics or model weights.
"""

import torch


try:
    torch.ops.torchvision.nms
except (AttributeError, RuntimeError):
    _TORCHVISION_LIBRARY = torch.library.Library("torchvision", "DEF")
    _TORCHVISION_LIBRARY.define(
        "nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"
    )
