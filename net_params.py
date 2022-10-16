# -*- coding: utf-8 -*-
'''
@Time    : 2022/9/25 10:35
@Author  : KingRyu
@Email   : 1050087553@qq.com
@Project : Sprog_CBAM_ResUnet
@File    : net_params.py
@Language: Python3
'''

from torch import nn
from encoder_cell import EncoderBasicBlock, FirstLayer
from decoder_cell import DecoderBasicBlock

encoder_params = [
    [
        nn.MaxPool2d(2, 2, 0),
        nn.MaxPool2d(2, 2, 0),
    ],

    [
        FirstLayer(42),
        EncoderBasicBlock(42, 64),
        EncoderBasicBlock(64, 64),
    ]
]

decoder_params = [
    [
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.Upsample(scale_factor=2, mode='bilinear'),
    ],

    [
        DecoderBasicBlock(256, 64),
        DecoderBasicBlock(128, 64),
        nn.Conv2d(64, 5, 1, 1, 0)
    ]
]