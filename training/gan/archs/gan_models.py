# archs/gan_models.py

import torch
import torch.nn as nn
from archs.darkir import DarkIR  # 导入 DarkIR 网络

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()

        # 通过 config 配置来初始化 DarkIR 网络
        self.darkir = DarkIR(
            img_channel=config['img_channels'],       # 输入通道数，RGB图像为 3
            width=config['width'],                    # 特征图的宽度
            middle_blk_num_enc=config['middle_blk_num_enc'], # 编码器中间块数量
            middle_blk_num_dec=config['middle_blk_num_dec'], # 解码器中间块数量
            enc_blk_nums=config['enc_blk_nums'],      # 编码器各阶段块数
            dec_blk_nums=config['dec_blk_nums'],      # 解码器各阶段块数
            dilations=config['dilations'],            # 扩张卷积因子
            extra_depth_wise=config['extra_depth_wise'] # 是否使用额外的深度卷积
        )

    def forward(self, x):
        # 通过 DarkIR 网络生成图像
        return self.darkir(x)  # 直接返回 DarkIR 的输出


class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # 64xH/2xW/2
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),

            # 128xH/4xW/4
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=False),

            # 256xH/8xW/8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=False),

            # 512xH/16xW/16
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=False),

            # 输出一张“真假概率图”（PatchGAN 思想）
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)