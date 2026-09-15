# 模型结构 (Model Architecture)

## 概述
本研究的核心是比较两种去模糊模型：**原始DarkIR模型**（仅使用生成器）与**DarkIR-GAN模型**（在DarkIR生成器基础上增加PatchGAN判别器，进行对抗训练）。通过对比两者在PSNR、SSIM、MUSIQ、LPIPS、NIQE五项指标上的性能，评估引入对抗监督对去模糊效果的影响。

## 生成器：DarkIR
DarkIR是一种专为低光图像恢复设计的编码器-解码器网络。与当前基于Transformer模型的趋势不同，DarkIR提出了新的注意力机制，以增强高效卷积神经网络（CNN）的感受野。与之前的方法相比，在参数和MAC操作方面减少了计算成本。DarkIR在流行的LOLBlur、LOLv2和Real-LOLBlur数据集上取得了新的最先进结果，并能够在现实世界的夜间和暗光图像上实现泛化。

本工作中，DarkIR作为生成器，其具体配置为：特征宽度64，编码块数[2,2,4]，解码块数[2,2,2]，中间块各1个，使用膨胀卷积[1,2,3]进一步扩大感受野。网络通过跳跃连接融合多尺度特征，最终通过残差连接输出去模糊图像。

## 判别器：PatchGAN（仅用于DarkIR-GAN）
在DarkIR-GAN模型中，我们引入了一个PatchGAN判别器。该判别器的核心思想是对图像局部区域（patch）进行真假判别，输出一个空间真假概率图而非单个标量。这种局部感知机制能够更精细地监督生成图像的纹理细节，提升去模糊结果的视觉质量。

PatchGAN判别器由4个卷积层组成：
1. 卷积层（4×4, stride=2）将3通道输入映射为64通道特征；
2. 卷积层（4×4, stride=2）输出128通道，后接批量归一化（BatchNorm）和LeakyReLU（α=0.2）；
3. 卷积层（4×4, stride=2）输出256通道，同样进行归一化和激活；
4. 卷积层（4×4, stride=1）输出512通道；
5. 最终卷积层（4×4, stride=1）输出1通道特征图，每个位置对应一个局部区域的判别得分。

## 损失函数
DarkIR-GAN模型的训练涉及多种损失函数，其总损失由以下各项加权构成：

### 原始DarkIR损失函数
原始DarkIR模型仅使用像素损失、感知损失、边缘损失和增强损失，不含对抗损失。其总损失为：
\[ \mathcal{L}_{\text{DarkIR}} = \lambda_{\text{pixel}} \mathcal{L}_{\text{pixel}} + \lambda_{\text{perceptual}} \mathcal{L}_{\text{perceptual}} + \lambda_{\text{edge}} \mathcal{L}_{\text{edge}} + \lambda_{\text{enhance}} \mathcal{L}_{\text{enhance}} \]

### 1. 像素损失（Pixel Loss）
采用L1范数衡量生成图像与真实清晰图像之间的像素级差异：
\[
\mathcal{L}_{\text{pixel}} = \| G(I_{\text{blur}}) - I_{\text{sharp}} \|_1
\]
其中 \( G \) 为生成器，\( I_{\text{blur}} \) 和 \( I_{\text{sharp}} \) 分别为模糊输入和清晰目标图像。权重 \( \lambda_{\text{pixel}} = 1.0 \)。

### 2. 感知损失（Perceptual Loss）
使用LPIPS（Learned Perceptual Image Patch Similarity）度量生成图像与目标图像在深度特征空间中的差异：
\[
\mathcal{L}_{\text{perceptual}} = \text{LPIPS}(G(I_{\text{blur}}), I_{\text{sharp}})
\]
权重 \( \lambda_{\text{perceptual}} = 1.0 \)。

### 3. 边缘损失（Edge Loss）
通过拉普拉斯算子提取图像边缘，计算边缘图的L2损失：
\[
\mathcal{L}_{\text{edge}} = \| \nabla G(I_{\text{blur}}) - \nabla I_{\text{sharp}} \|_2
\]
权重 \( \lambda_{\text{edge}} = 0.1 \)。

### 4. 增强损失（Enhance Loss）
在编码器中间层进行下采样，计算下采样图像的L1损失，旨在增强低光区域的细节：
\[
\mathcal{L}_{\text{enhance}} = \| \text{downsample}(G(I_{\text{blur}})) - \text{downsample}(I_{\text{sharp}}) \|_1
\]
权重 \( \lambda_{\text{enhance}} = 1.0 \)。

### 5. 对抗损失（Adversarial Loss）
采用均方误差（MSE）作为对抗损失。判别器 \( D \) 的目标是区分真实清晰图像与生成图像，而生成器的目标是欺骗判别器。

**判别器损失**（含标签平滑）：
\[
\mathcal{L}_D = \frac{1}{2} \left[ \| D(I_{\text{sharp}}) - 0.9 \|^2 + \| D(G(I_{\text{blur}})) - 0 \|^2 \right]
\]

**生成器对抗损失**：
\[
\mathcal{L}_{\text{adv}} = \| D(G(I_{\text{blur}})) - 1 \|^2
\]
对抗损失权重 \( \lambda_{\text{adv}} = 0.01 \)。

### 总损失
DarkIR-GAN生成器的总损失为：
\[
\mathcal{L}_G = \lambda_{\text{pixel}} \mathcal{L}_{\text{pixel}} + \lambda_{\text{perceptual}} \mathcal{L}_{\text{perceptual}} + \lambda_{\text{edge}} \mathcal{L}_{\text{edge}} + \lambda_{\text{enhance}} \mathcal{L}_{\text{enhance}} + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}}
\]

原始DarkIR模型仅使用前四项损失（不含对抗损失），即 \( \mathcal{L}_{\text{adv}} = 0 \)。

## 模型对比
- **原始DarkIR模型**：仅包含DarkIR生成器，通过像素级损失（如L1）和感知损失进行训练。
- **DarkIR-GAN模型**：在DarkIR生成器基础上增加PatchGAN判别器，训练时引入对抗损失（adversarial loss），使生成器在对抗中学习产生更逼真的纹理。

两种模型的生成器结构完全一致，区别仅在于是否引入判别器及对抗训练。这种设计使得对比实验能够直接反映对抗监督对去模糊性能的贡献。