# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from ._utils import _C
# from maskrcnn_benchmark import _C

from torch.cuda.amp import *

# Only valid with fp32 inputs - give AMP the hint
def nms(*args, **kwargs):
    with autocast():
        return _C.nms(*args, **kwargs)

# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
