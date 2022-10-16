# -*- coding: utf-8 -*-
'''
@Time    : 2022/9/25 10:27
@Author  : KingRyu
@Email   : 1050087553@qq.com
@Project : Sprog_CBAM_ResUnet
@File    : net.py
@Language: Python3
'''

from torch import nn

class Net(nn.Module):

    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs):
        short_cuts = self.encoder(inputs)
        outputs = self.decoder(short_cuts)
        return outputs