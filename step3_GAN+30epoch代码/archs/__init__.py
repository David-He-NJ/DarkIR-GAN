# archs/__init__.py

import torch
import torch.nn as nn
from .gan_models import Generator, Discriminator  # 导入 GAN 模型
from torch.optim.lr_scheduler import CosineAnnealingLR

class GANModel(nn.Module):
    """
    定义 GAN 模型，由生成器和判别器组成
    """
    def __init__(self, generator, discriminator):
        super(GANModel, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x):
        # 生成器生成图像
        generated_image = self.generator(x)
        # 判别器判别生成图像
        disc_output = self.discriminator(generated_image)
        return generated_image, disc_output


def create_model(config, rank):
    """
    创建 GAN 模型，包括生成器（DarkIR）和判别器。
    """
    network_config = config['network']
    
    # 创建 DarkIR 生成器
    generator = Generator(config['network'])  # 使用配置来初始化 DarkIR 生成器
    
    # 创建判别器
    discriminator = Discriminator(input_channels=3)  # 假设输入是 RGB 图像（3 通道）
    
    # 创建 GAN 模型
    model = GANModel(generator, discriminator)
    
    # 计算模型的参数数量和计算量
    macs, params = count_params(model)  
    
    return model, macs, params


from ptflops import get_model_complexity_info

def count_params(model):
    """
    计算模型的参数数量和 MACs（乘加操作次数）
    """
    # 计算模型的参数数量
    total_params = sum(p.numel() for p in model.parameters())
    
    # 计算 MACs 和参数量
    # 假设输入尺寸是 (3, 256, 256)，你可以根据需要修改输入尺寸
    macs, _ = get_model_complexity_info(model, (3, 256, 256), as_strings=False, print_per_layer_stat=False)
    
    return macs, total_params


def resume_model(model, optimizer, scheduler, resume_path, rank, resume_epoch):
    """
    从指定路径恢复模型和优化器
    """
    print(f"Resuming model from: {resume_path}")
    checkpoint = torch.load(resume_path, map_location='cpu')
    epoch_start = checkpoint['epoch'] if 'epoch' in checkpoint else 0
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"Resumed model at epoch {epoch_start}")
    return model, optimizer, scheduler, epoch_start


def save_checkpoint(model, optimizer, scheduler, metrics_eval, metrics_train, paths, rank=0):
    """
    保存当前的模型状态、优化器和调度器
    """
    state_dict = {
        'epoch': metrics_train['epoch'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics_eval': metrics_eval,
        'metrics_train': metrics_train,
    }
    
    torch.save(state_dict, paths['new'])
    print(f"Checkpoint saved at {paths['new']}")
    
    if metrics_eval['valid_psnr'] > metrics_train.get('best_psnr', 0):
        # Save best model if PSNR improves
        torch.save(state_dict, paths['best'])
        print(f"Best model saved at {paths['best']}")

def create_optim_scheduler(config, model):
    """
    创建优化器和学习率调度器
    """
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                               lr=config['lr_initial'],
                               weight_decay=config['weight_decay'],
                               betas=(config['betas'][0], config['betas'][1]))
    
    if config['lr_scheme'] == 'CosineAnnealing':
        scheduler = CosineAnnealingLR(optim, T_max=config['epochs'], eta_min=config['eta_min'])
    else:
        raise NotImplementedError('The specified learning rate scheduler is not implemented.')
    
    return optim, scheduler