# train_gan_multi.py
# GAN 版本训练脚本 - 与 Baseline 版本保持一致的记录和可视化逻辑

import os
import csv
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torch.utils.tensorboard import SummaryWriter
import pyiqa
import torchvision.utils as vutils

from archs import create_model, create_optim_scheduler, resume_model, save_checkpoint
from data import create_dataloaders
from losses import create_loss, calculate_loss


# ---------------------------
# 读取配置文件
# ---------------------------
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# ---------------------------
# 8个目标场景ID
# ---------------------------
TARGET_IDS = ['0017-0013', '0182-0037', '0045-0014', '0127-0025', 
              '0060-0035', '0153-0013', '0193-0037', '0082-0010']


# ---------------------------
# 路径识别逻辑：提取场景ID
# ---------------------------
def extract_scene_id(path):
    """
    从路径中提取场景ID
    例如: .../0012/0010.png -> 0012-0010
    """
    folder = path.split('/')[-2]
    filename = path.split('/')[-1].split('.')[0]
    current_id = f"{folder}-{filename}"
    return current_id


# ---------------------------
# 保存可视化输出图
# ---------------------------
def save_comparison_image(inputs, targets, fake_images, save_path, epoch, current_id):
    """
    只保存模型输出图（Output）
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 归一化到[0, 1]
    fake_images = torch.clamp(fake_images, 0, 1)
    
    # 只保存输出图像
    vutils.save_image(fake_images, save_path, nrow=1, normalize=False)
    print(f"📸 Saved output image: {save_path}")


# ---------------------------
# 安全指标计算函数（避免奇异矩阵）
# ---------------------------
def safe_iqa_metrics(fake_images, target_images, device,
                     psnr_model=None, ssim_model=None, lpips_model=None,
                     musiq_model=None, nrqm_model=None, niqe_model=None):
    """
    安全地计算 IQA 指标，避免奇异矩阵问题
    """
    results = {'psnr': 0, 'ssim': 0, 'lpips': 0, 'musiq': 0, 'nrqm': 0, 'niqe': 0}

    fake_images = torch.clamp(fake_images, 0, 1)
    target_images = torch.clamp(target_images, 0, 1)

    try:
        if psnr_model:
            results['psnr'] = psnr_model(fake_images, target_images).item()
    except Exception:
        pass

    try:
        if ssim_model:
            results['ssim'] = ssim_model(fake_images, target_images).item()
    except Exception:
        pass

    try:
        if lpips_model:
            results['lpips'] = lpips_model(fake_images, target_images).item()
    except Exception:
        pass

    try:
        if musiq_model:
            results['musiq'] = musiq_model(fake_images).item()
    except Exception:
        pass

    try:
        if nrqm_model:
            results['nrqm'] = nrqm_model(fake_images).item()
    except Exception:
        pass

    try:
        if niqe_model:
            results['niqe'] = niqe_model(fake_images).item()
    except Exception:
        pass

    return results


# ---------------------------
# 训练一个 epoch（GAN 模式 - 与Baseline一致）
# ---------------------------
def train_one_epoch_gan(
    generator,
    discriminator,
    dataloader,
    criterion,
    optimizer_G,
    optimizer_D,
    device,
    lambda_adv=0.005
):
    """
    每个 epoch 的训练：只进行训练和计算 loss，不计算指标
    """
    generator.train()
    discriminator.train()

    total_loss_G = total_loss_D = 0.0
    total_loss_G_adv = 0.0
    total_loss_pixel = 0.0
    total_loss_perceptual = 0.0
    total_loss_edge = 0.0
    total_loss_enhance = 0.0

    # 创建对抗损失函数
    adversarial_criterion = nn.MSELoss()

    for i, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        # ---------------------------
        # 1. 训练判别器
        # ---------------------------
        optimizer_D.zero_grad()
        
        # 判别器对真实图像的输出
        outputs_real = discriminator(targets)
        # 标签平滑：真图标签设为 0.9，防止判别器过于自信
        real_labels = torch.ones_like(outputs_real, device=device) * 0.9
        loss_D_real = adversarial_criterion(outputs_real, real_labels)
        
        # 生成假图像
        fake_images = generator(inputs)
        
        # 判别器对假图像的输出（使用detach避免更新生成器）
        outputs_fake = discriminator(fake_images.detach())
        loss_D_fake = adversarial_criterion(outputs_fake, torch.zeros_like(outputs_fake, device=device))
        
        # 判别器总损失
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        optimizer_D.step()

        # ---------------------------
        # 2. 训练生成器
        # ---------------------------
        optimizer_G.zero_grad()
        
        # 计算DarkIR复合损失
        # Pixel loss
        loss_pixel = criterion['pixel_loss'](fake_images, targets)
        
        # 感知损失 (LPIPS)
        loss_perceptual = 0.0
        if 'perceptual_loss' in criterion:
            loss_perceptual = criterion['perceptual_loss'](fake_images, targets)
        
        # 边缘损失
        loss_edge = 0.0
        if 'edge_loss' in criterion:
            loss_edge = criterion['edge_loss'](fake_images, targets)
        
        # 增强损失
        loss_enhance = 0.0
        if 'enhance_loss' in criterion:
            loss_enhance = criterion['enhance_loss'](inputs, targets)
        
        # 生成器对抗损失
        outputs_fake_for_G = discriminator(fake_images)
        loss_G_adversarial = adversarial_criterion(outputs_fake_for_G, torch.ones_like(outputs_fake_for_G, device=device))
        
        # 修改后的生成器总损失
        # 策略：大幅降低 Edge，大幅提升 Perceptual，适度增加 GAN
        loss_G_total = (1.0 * loss_pixel +             # Pixel 保持 1.0 守底线
                       (1.0 * loss_perceptual if not isinstance(loss_perceptual, float) else 0) + # ★ 关键！从 0.03 提到 1.0，强力压制 LPIPS
                       (0.1 * loss_edge if not isinstance(loss_edge, float) else 0) +       # ★ 关键！从 30.0 砍到 0.1。彻底解放 GAN 生成纹理的能力
                       (1.0 * loss_enhance if not isinstance(loss_enhance, float) else 0) +   # Enhance 保持
                       (0.01 * loss_G_adversarial))    # ★ 关键！GAN 权重给到 0.01，让判别器说话算数
        
        loss_G_total.backward()
        optimizer_G.step()

        # 累积损失
        total_loss_G += loss_G_total.item()
        total_loss_D += loss_D.item()
        total_loss_G_adv += loss_G_adversarial.item()
        total_loss_pixel += loss_pixel.item()
        total_loss_perceptual += loss_perceptual.item() if not isinstance(loss_perceptual, float) else 0
        total_loss_edge += loss_edge.item() if not isinstance(loss_edge, float) else 0
        total_loss_enhance += loss_enhance.item() if not isinstance(loss_enhance, float) else 0

    # epoch 结束时打印平均损失
    avg_loss_G = total_loss_G / len(dataloader)
    avg_loss_D = total_loss_D / len(dataloader)
    avg_loss_pixel = total_loss_pixel / len(dataloader)
    avg_loss_perceptual = total_loss_perceptual / len(dataloader)
    avg_loss_edge = total_loss_edge / len(dataloader)
    avg_loss_enhance = total_loss_enhance / len(dataloader)
    
    print(
        f"Epoch Training Loss: G={avg_loss_G:.4f}, D={avg_loss_D:.4f}, "
        f"Pixel: {avg_loss_pixel:.4f}, "
        f"Perc: {avg_loss_perceptual:.4f}, "
        f"Edge: {avg_loss_edge:.4f}, "
        f"Enhance: {avg_loss_enhance:.4f}, "
        f"Adv: {total_loss_G_adv / len(dataloader):.4f}"
    )

    return (avg_loss_G, avg_loss_D, avg_loss_pixel, avg_loss_perceptual, avg_loss_edge, avg_loss_enhance)


# ---------------------------
# 快速验证阶段（每个场景选15张）
# ---------------------------
def validate_quick_gan(generator, discriminator, dataloader, criterion, device,
                       psnr_model, ssim_model, lpips_model,
                       musiq_model, nrqm_model, niqe_model,
                       epoch, desc="Quick Val", lambda_adv=0.005):
    """
    快速验证：计算指标但不保存可视化结果
    用于每个epoch后的快速评估
    """
    generator.eval()
    discriminator.eval()

    total_loss_G = total_loss_D = 0.0
    total_loss_pixel = 0.0
    total_loss_perceptual = 0.0
    total_loss_edge = 0.0
    total_loss_enhance = 0.0
    psnr_sum = ssim_sum = lpips_sum = musiq_sum = nrqm_sum = niqe_sum = 0.0
    count = 0
    
    # 创建对抗损失函数
    adversarial_criterion = nn.MSELoss()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=desc, leave=False)):
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                inputs, targets, _ = batch
            else:
                inputs, targets = batch
            
            inputs, targets = inputs.to(device), targets.to(device)
            fake_images = generator(inputs)
            fake_images = torch.clamp(fake_images, 0.0, 1.0)
            targets = torch.clamp(targets, 0.0, 1.0)

            # 判别器损失计算（仅用于记录）
            outputs_real = discriminator(targets)
            # 标签平滑：真图标签设为 0.9
            real_labels = torch.ones_like(outputs_real, device=device) * 0.9
            loss_D_real = adversarial_criterion(outputs_real, real_labels)
            outputs_fake = discriminator(fake_images.detach())
            loss_D_fake = adversarial_criterion(outputs_fake, torch.zeros_like(outputs_fake, device=device))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            total_loss_D += loss_D.item()

            # 计算DarkIR复合损失
            loss_pixel = criterion['pixel_loss'](fake_images, targets)
            
            loss_perceptual = 0.0
            if 'perceptual_loss' in criterion:
                loss_perceptual = criterion['perceptual_loss'](fake_images, targets)
            
            loss_edge = 0.0
            if 'edge_loss' in criterion:
                loss_edge = criterion['edge_loss'](fake_images, targets)
            
            loss_enhance = 0.0
            if 'enhance_loss' in criterion:
                loss_enhance = criterion['enhance_loss'](inputs, targets)
            
            # 修改后的快速验证损失（与训练一致）
            loss_G = (1.0 * loss_pixel +
                     (1.0 * loss_perceptual if not isinstance(loss_perceptual, float) else 0) +
                     (0.1 * loss_edge if not isinstance(loss_edge, float) else 0) +
                     (1.0 * loss_enhance if not isinstance(loss_enhance, float) else 0))
            
            total_loss_G += loss_G.item()
            total_loss_pixel += loss_pixel.item()
            total_loss_perceptual += loss_perceptual.item() if not isinstance(loss_perceptual, float) else 0
            total_loss_edge += loss_edge.item() if not isinstance(loss_edge, float) else 0
            total_loss_enhance += loss_enhance.item() if not isinstance(loss_enhance, float) else 0

            metrics = safe_iqa_metrics(
                fake_images, targets, device,
                psnr_model=psnr_model, ssim_model=ssim_model, lpips_model=lpips_model,
                musiq_model=musiq_model, nrqm_model=nrqm_model, niqe_model=niqe_model
            )
            psnr_sum += metrics['psnr']
            ssim_sum += metrics['ssim']
            lpips_sum += metrics['lpips']
            musiq_sum += metrics['musiq']
            nrqm_sum += metrics['nrqm']
            niqe_sum += metrics['niqe']
            count += 1

    avg_loss_G = total_loss_G / len(dataloader) if len(dataloader) > 0 else 0
    avg_loss_D = total_loss_D / len(dataloader) if len(dataloader) > 0 else 0
    avg_loss_pixel = total_loss_pixel / len(dataloader) if len(dataloader) > 0 else 0
    avg_loss_perceptual = total_loss_perceptual / len(dataloader) if len(dataloader) > 0 else 0
    avg_loss_edge = total_loss_edge / len(dataloader) if len(dataloader) > 0 else 0
    avg_loss_enhance = total_loss_enhance / len(dataloader) if len(dataloader) > 0 else 0
    avg_psnr = psnr_sum / count if count > 0 else 0
    avg_ssim = ssim_sum / count if count > 0 else 0
    avg_lpips = lpips_sum / count if count > 0 else 0
    avg_musiq = musiq_sum / count if count > 0 else 0
    avg_nrqm = nrqm_sum / count if count > 0 else 0
    avg_niqe = niqe_sum / count if count > 0 else 0

    print(
        f"{desc} → Loss_G: {avg_loss_G:.4f}, Loss_D: {avg_loss_D:.4f}, "
        f"PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}, "
        f"MUSIQ: {avg_musiq:.4f}, NRQM: {avg_nrqm:.4f}, NIQE: {avg_niqe:.4f}"
    )

    return (avg_loss_G, avg_loss_D, avg_loss_pixel, avg_loss_perceptual, avg_loss_edge, avg_loss_enhance,
            avg_psnr, avg_ssim, avg_lpips, avg_musiq, avg_nrqm, avg_niqe)


# ---------------------------
# 完整验证阶段（GAN 模式，带可视化）
# ---------------------------
def validate_gan(generator, discriminator, dataloader, criterion, device,
                 psnr_model, ssim_model, lpips_model,
                 musiq_model, nrqm_model, niqe_model,
                 epoch, vis_save_dir, lambda_adv=0.005):
    """
    验证阶段，计算指标并保存特定场景的可视化结果
    """
    generator.eval()
    discriminator.eval()

    total_loss_G = total_loss_D = 0.0
    total_loss_pixel = 0.0
    total_loss_perceptual = 0.0
    total_loss_edge = 0.0
    total_loss_enhance = 0.0
    psnr_sum = ssim_sum = lpips_sum = musiq_sum = nrqm_sum = niqe_sum = 0.0
    count = 0
    
    # 创建对抗损失函数
    adversarial_criterion = nn.MSELoss()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating", leave=False)):
            # 检查batch是否包含路径信息
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                inputs, targets, paths = batch
            else:
                inputs, targets = batch
                paths = None
            
            inputs, targets = inputs.to(device), targets.to(device)

            fake_images = generator(inputs)
            fake_images = torch.clamp(fake_images, 0.0, 1.0)
            targets = torch.clamp(targets, 0.0, 1.0)

            # 判别器损失计算（仅用于记录）
            outputs_real = discriminator(targets)
            # 标签平滑：真图标签设为 0.9
            real_labels = torch.ones_like(outputs_real, device=device) * 0.9
            loss_D_real = adversarial_criterion(outputs_real, real_labels)
            outputs_fake = discriminator(fake_images.detach())
            loss_D_fake = adversarial_criterion(outputs_fake, torch.zeros_like(outputs_fake, device=device))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            total_loss_D += loss_D.item()

            # 计算DarkIR复合损失
            loss_pixel = criterion['pixel_loss'](fake_images, targets)
            
            loss_perceptual = 0.0
            if 'perceptual_loss' in criterion:
                loss_perceptual = criterion['perceptual_loss'](fake_images, targets)
            
            loss_edge = 0.0
            if 'edge_loss' in criterion:
                loss_edge = criterion['edge_loss'](fake_images, targets)
            
            loss_enhance = 0.0
            if 'enhance_loss' in criterion:
                loss_enhance = criterion['enhance_loss'](inputs, targets)
            
            # 修改后的验证损失（与训练一致）
            loss_G = (1.0 * loss_pixel +
                     (1.0 * loss_perceptual if not isinstance(loss_perceptual, float) else 0) +
                     (0.1 * loss_edge if not isinstance(loss_edge, float) else 0) +
                     (1.0 * loss_enhance if not isinstance(loss_enhance, float) else 0))
            
            total_loss_G += loss_G.item()
            total_loss_pixel += loss_pixel.item()
            total_loss_perceptual += loss_perceptual.item() if not isinstance(loss_perceptual, float) else 0
            total_loss_edge += loss_edge.item() if not isinstance(loss_edge, float) else 0
            total_loss_enhance += loss_enhance.item() if not isinstance(loss_enhance, float) else 0

            metrics = safe_iqa_metrics(
                fake_images, targets, device,
                psnr_model=psnr_model, ssim_model=ssim_model, lpips_model=lpips_model,
                musiq_model=musiq_model, nrqm_model=nrqm_model, niqe_model=niqe_model
            )
            psnr_sum += metrics['psnr']
            ssim_sum += metrics['ssim']
            lpips_sum += metrics['lpips']
            musiq_sum += metrics['musiq']
            nrqm_sum += metrics['nrqm']
            niqe_sum += metrics['niqe']
            count += 1

            # ---------------------------
            # 特定场景可视化保存
            # ---------------------------
            if paths is not None:
                for j, path in enumerate(paths):
                    if j >= inputs.size(0):
                        break
                    current_id = extract_scene_id(path)
                    if current_id in TARGET_IDS:
                        save_path = os.path.join(vis_save_dir, f"epoch_{epoch+1}_{current_id}_output.png")
                        save_comparison_image(
                            inputs[j:j+1],
                            targets[j:j+1],
                            fake_images[j:j+1],
                            save_path,
                            epoch+1,
                            current_id
                        )

    avg_loss_G = total_loss_G / len(dataloader)
    avg_loss_D = total_loss_D / len(dataloader)
    avg_loss_pixel = total_loss_pixel / len(dataloader)
    avg_loss_perceptual = total_loss_perceptual / len(dataloader)
    avg_loss_edge = total_loss_edge / len(dataloader)
    avg_loss_enhance = total_loss_enhance / len(dataloader)
    avg_psnr = psnr_sum / count
    avg_ssim = ssim_sum / count
    avg_lpips = lpips_sum / count
    avg_musiq = musiq_sum / count
    avg_nrqm = nrqm_sum / count
    avg_niqe = niqe_sum / count

    print(
        f"Validation → Loss_G: {avg_loss_G:.4f}, Loss_D: {avg_loss_D:.4f}, "
        f"PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}, "
        f"MUSIQ: {avg_musiq:.4f}, NRQM: {avg_nrqm:.4f}, NIQE: {avg_niqe:.4f}"
    )

    return (avg_loss_G, avg_loss_D, avg_loss_pixel, avg_loss_perceptual, avg_loss_edge, avg_loss_enhance,
            avg_psnr, avg_ssim, avg_lpips, avg_musiq, avg_nrqm, avg_niqe)


# ---------------------------
# 初始化CSV日志
# ---------------------------
def init_csv_logger(log_path):
    """初始化CSV日志文件"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'val_loss_G', 'val_loss_D', 'val_loss_pixel', 'val_loss_perceptual',
                         'val_loss_edge', 'val_loss_enhance',
                         'val_psnr', 'val_ssim', 'val_lpips', 'val_musiq', 'val_nrqm', 'val_niqe'])


# ---------------------------
# 追加CSV日志
# ---------------------------
def append_csv_log(log_path, epoch, val_loss_G, val_loss_D, val_loss_pixel, val_loss_perceptual,
                   val_loss_edge, val_loss_enhance, val_psnr, val_ssim, val_lpips,
                   val_musiq, val_nrqm, val_niqe):
    """追加一行训练日志到CSV，包含所有验证指标"""
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, val_loss_G, val_loss_D, val_loss_pixel, val_loss_perceptual,
                        val_loss_edge, val_loss_enhance, val_psnr, val_ssim, val_lpips,
                        val_musiq, val_nrqm, val_niqe])


# ---------------------------
# 保存检查点（双重最佳策略）
# ---------------------------
def save_checkpoint_gan(generator, discriminator, optimizer_G, optimizer_D,
                        scheduler_G, scheduler_D, epoch, val_psnr, val_lpips,
                        save_dir, best_psnr, best_lpips, is_periodic=False):
    """
    GAN双重最佳权重保存策略（与Baseline一致）
    - gan_best_psnr.pt: PSNR最高时的模型
    - gan_best_lpips.pt: LPIPS最低时的模型
    - gan_epoch_{epoch}.pt: 每5个Epoch定期备份
    """
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'scheduler_G_state_dict': scheduler_G.state_dict(),
        'scheduler_D_state_dict': scheduler_D.state_dict(),
        'val_psnr': val_psnr,
        'val_lpips': val_lpips,
    }
    
    # 定期备份（每5个epoch）
    if is_periodic:
        periodic_path = os.path.join(save_dir, f"gan_epoch_{epoch}.pt")
        torch.save(checkpoint, periodic_path)
        print(f"💾 Periodic checkpoint saved: {periodic_path}")
    
    # 保存最佳PSNR模型
    if val_psnr > best_psnr:
        best_psnr_path = os.path.join(save_dir, "gan_best_psnr.pt")
        torch.save(checkpoint, best_psnr_path)
        print(f"🏆 New best PSNR: {val_psnr:.2f}! Saved to {best_psnr_path}")
        best_psnr = val_psnr
    
    # 保存最佳LPIPS模型
    if val_lpips < best_lpips:
        best_lpips_path = os.path.join(save_dir, "gan_best_lpips.pt")
        torch.save(checkpoint, best_lpips_path)
        print(f"🏆 New best LPIPS: {val_lpips:.4f}! Saved to {best_lpips_path}")
        best_lpips = val_lpips
    
    return best_psnr, best_lpips


# ---------------------------
# 主函数
# ---------------------------
def main():
    # 加载配置文件
    opt = load_config("options/config.yaml")
    
    rank = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------
    # 1. 设置核心超参数（与Baseline一致）
    # ---------------------------
    TOTAL_EPOCHS = 30  # 训练epoch为30
    LEARNING_RATE = 1e-6  # 微调学习率（与Baseline一致）
    
    # 覆盖配置中的参数
    opt['train']['epochs'] = TOTAL_EPOCHS
    opt['train']['lr_initial'] = LEARNING_RATE
    
    print(f"🎯 GAN Training Configuration:")
    print(f"   - Total Epochs: {TOTAL_EPOCHS}")
    print(f"   - Learning Rate: {LEARNING_RATE}")
    print(f"   - Adversarial Weight: 0.005")

    # 创建模型
    model, macs, params = create_model(opt, rank)
    generator = model.generator.to(device)
    discriminator = model.discriminator.to(device)

    # ========== 加载 Baseline 预训练权重作为生成器起点 ==========
    # 使用与 DarkIR 训练函数相同的 Baseline 权重
    checkpoint_path = 'experiments/checkpoints/baseline_best_lpips.pt'
    start_epoch = 0
    
    if os.path.exists(checkpoint_path):
        print(f"\n📦 Loading Baseline checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 检查checkpoint格式并提取state_dict（与Baseline一致的逻辑）
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                # Baseline格式的checkpoint
                generator_dict = checkpoint['model_state_dict']
                # 注意：GAN训练始终从epoch 0开始，不使用Baseline的epoch
                # start_epoch = checkpoint.get('epoch', 0)
                print(f"   📂 Loaded from Baseline checkpoint with 'model_state_dict' key")
            elif 'state_dict' in checkpoint:
                generator_dict = checkpoint['state_dict']
                print(f"   📂 Loaded from checkpoint with 'state_dict' key")
            else:
                generator_dict = checkpoint
                print(f"   📂 Loaded checkpoint directly")
        else:
            generator_dict = checkpoint
            print(f"   📂 Loaded checkpoint directly")
        
        generator_model_dict = generator.state_dict()
        
        # 检查是否需要添加 'darkir.' 前缀（Generator包装了DarkIR）
        # Baseline checkpoint的键名如 'intro.weight'，而Generator的键名如 'darkir.intro.weight'
        sample_key = list(generator_dict.keys())[0]
        if 'darkir.' not in sample_key and 'darkir.' in list(generator_model_dict.keys())[0]:
            print(f"   🔄 添加 'darkir.' 前缀到权重键名（{len(generator_dict)}层）...")
            generator_dict = {f"darkir.{k}": v for k, v in generator_dict.items()}
        
        # 检查生成器匹配的层数和形状（与Baseline一致）
        matched_layers_g = 0
        shape_mismatches_g = []
        for k in generator_dict:
            if k in generator_model_dict:
                if generator_dict[k].shape == generator_model_dict[k].shape:
                    matched_layers_g += 1
                else:
                    shape_mismatches_g.append(f"{k}: checkpoint {generator_dict[k].shape} vs model {generator_model_dict[k].shape}")
        
        generator.load_state_dict(generator_dict, strict=True)
        print(f"✅ Loaded generator weights from Baseline checkpoint.")
        print(f"   📊 匹配层数: {matched_layers_g}/{len(generator_dict)} (checkpoint) | 模型总层数: {len(generator_model_dict)}")
        if shape_mismatches_g:
            print(f"   ⚠️ 形状不匹配的层: {len(shape_mismatches_g)}")
            for m in shape_mismatches_g[:5]:
                print(f"      - {m}")
        
        # 验证权重是否正确加载（与Baseline一致）
        print(f"   🔍 验证权重加载...")
        test_param = generator.state_dict()['darkir.intro.weight']
        ckpt_param = generator_dict['darkir.intro.weight']
        if torch.allclose(test_param.cpu(), ckpt_param.cpu()):
            print(f"   ✔️ 权重验证通过 (darkir.intro.weight 匹配)")
        else:
            print(f"   ❌ 权重验证失败！darkir.intro.weight 不匹配")
        
        # 注意：Baseline权重不包含判别器，判别器将从头训练
        print(f"   💡 判别器将从头训练（Baseline权重不包含判别器）")
    else:
        print(f"⚠️ Checkpoint not found at {checkpoint_path}, training from scratch.")
        print(f"   💡 Hint: Set resume_path in config.yaml to pretrained model path")
    # =============================================================

    # 加载IQA指标模型（与Baseline一致，保留6项指标）
    print("📏 Loading IQA metrics...")
    lpips_model = pyiqa.create_metric('lpips', device=device, pretrained_model_path='/root/autodl-tmp/lpips.pth')
    musiq_model = pyiqa.create_metric('musiq', device=device, pretrained_model_path='/root/autodl-tmp/musiq.pth')
    niqe_model  = pyiqa.create_metric('niqe', device=device, pretrained_model_path='/root/autodl-tmp/niqe.mat')
    nrqm_model  = pyiqa.create_metric('nrqm', device=device, pretrained_model_path='/root/autodl-tmp/nrqm.mat')
    print("✅ IQA metrics loaded (PSNR, SSIM, LPIPS, MUSIQ, NRQM, NIQE).")

    # 创建数据加载器
    # 训练集：每个场景选择15张图片
    train_loader, train_sampler = create_dataloaders(rank, 1, opt['datasets'], mode='train', return_path=False, samples_per_scene=15)
    # 快速验证集：每个场景选15张，不返回路径
    val_loader_quick, _ = create_dataloaders(rank, 1, opt['datasets'], mode='val', return_path=False, samples_per_scene=15)
    # 完整验证集：所有图片都跑，返回路径用于可视化（每3个epoch使用）
    val_loader_full, _ = create_dataloaders(rank, 1, opt['datasets'], mode='val', return_path=True, samples_per_scene=-1)

    # 创建损失函数
    criterion = create_loss(opt['train']['loss'], rank=0)
    
    # 重新定义优化器：G和D使用不同的学习率
    LR_G = 1e-5  # 生成器学习率
    LR_D = 1e-4  # 判别器学习率
    ETA_MIN = 1e-6  # 最小学习率（最后5个epoch达到）
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR_G, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR_D, betas=(0.5, 0.999))
    
    # 使用 Cosine Annealing 调度器
    # T_max = TOTAL_EPOCHS，确保在最后一个epoch降到eta_min
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=TOTAL_EPOCHS, eta_min=ETA_MIN)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=TOTAL_EPOCHS, eta_min=ETA_MIN)
    
    print(f"✅ Optimizers created:")
    print(f"   Generator: Adam, lr={LR_G}, betas=(0.5, 0.999)")
    print(f"   Discriminator: Adam, lr={LR_D}, betas=(0.5, 0.999)")
    print(f"✅ Schedulers: CosineAnnealingLR, T_max={TOTAL_EPOCHS}, eta_min={ETA_MIN}")
    print(f"   （最后5个epoch LR将回到1e-6级别）")

    # ---------------------------
    # 2. 创建实验目录和日志系统（与Baseline一致）
    # ---------------------------
    experiment_dir = 'experiments'
    vis_save_dir = os.path.join(experiment_dir, 'GAN_vis')
    save_dir = os.path.join(experiment_dir, 'checkpoints')
    log_path = os.path.join(experiment_dir, 'gan_training_log.csv')
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(vis_save_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化CSV日志
    init_csv_logger(log_path)
    print(f"📝 CSV logger initialized: {log_path}")
    
    # 初始化TensorBoard
    tb_writer = SummaryWriter(log_dir=os.path.join(experiment_dir, 'tensorboard_gan'))
    print(f"📊 TensorBoard initialized: {os.path.join(experiment_dir, 'tensorboard_gan')}")

    # 初始化最佳指标记录
    best_psnr = 0.0
    best_lpips = float('inf')
    
    # 评估加载的模型
    print("\n📝 Evaluating loaded model (quick validation)...")
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lambda_adv = opt['train']['loss'].get('adversarial_weight', 0.005)
    validate_quick_gan(generator, discriminator, val_loader_quick, criterion, device,
                       psnr_metric, ssim_metric, lpips_model,
                       musiq_model, nrqm_model, niqe_model,
                       start_epoch, desc="Init Val (Loaded)", lambda_adv=lambda_adv)

    print(f"\n🎯 Target scenes for visualization: {TARGET_IDS}")

    # 创建指标计算模型
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # ---------------------------
    # 3. 训练循环（与Baseline一致）
    # ---------------------------
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch [{epoch+1}/{TOTAL_EPOCHS}]")
        print(f"{'='*60}")
        
        # 在每个epoch开始时，如果sampler有set_epoch方法，则调用它进行重新采样
        if hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)
            print(f"🔄 Resampled training data: {len(train_sampler)} samples for this epoch")
        
        lambda_adv = opt['train']['loss'].get('adversarial_weight', 0.005)
        
        # 训练一个epoch
        (train_loss_G, train_loss_D, train_loss_pixel,
         train_loss_perceptual, train_loss_edge, train_loss_enhance) = train_one_epoch_gan(
            generator, discriminator, train_loader, criterion,
            optimizer_G, optimizer_D, device,
            lambda_adv=lambda_adv
        )

        scheduler_G.step()
        scheduler_D.step()

        # ---------------------------
        # 每个epoch都进行快速验证
        # ---------------------------
        print(f"\n📝 Running quick validation after epoch {epoch+1}...")
        (quick_val_loss_G, quick_val_loss_D, quick_val_loss_pixel, quick_val_loss_perceptual,
         quick_val_loss_edge, quick_val_loss_enhance,
         quick_val_psnr, quick_val_ssim, quick_val_lpips,
         quick_val_musiq, quick_val_nrqm, quick_val_niqe) = validate_quick_gan(
            generator, discriminator, val_loader_quick, criterion, device,
            psnr_metric, ssim_metric, lpips_model,
            musiq_model, nrqm_model, niqe_model,
            epoch, desc=f"Epoch {epoch+1} Quick Val", lambda_adv=lambda_adv
        )

        # 记录训练损失到TensorBoard（每个epoch都记录）
        tb_writer.add_scalar('Loss/train_G', train_loss_G, epoch+1)
        tb_writer.add_scalar('Loss/train_D', train_loss_D, epoch+1)
        tb_writer.add_scalar('Loss/train_pixel', train_loss_pixel, epoch+1)
        tb_writer.add_scalar('Loss/train_perceptual', train_loss_perceptual, epoch+1)
        tb_writer.add_scalar('Loss/train_edge', train_loss_edge, epoch+1)
        tb_writer.add_scalar('Loss/train_enhance', train_loss_enhance, epoch+1)
        
        # 记录快速验证结果
        tb_writer.add_scalar('Loss/val_quick_G', quick_val_loss_G, epoch+1)
        tb_writer.add_scalar('Loss/val_quick_D', quick_val_loss_D, epoch+1)
        tb_writer.add_scalar('Metrics/PSNR_quick', quick_val_psnr, epoch+1)
        tb_writer.add_scalar('Metrics/SSIM_quick', quick_val_ssim, epoch+1)
        tb_writer.add_scalar('Metrics/LPIPS_quick', quick_val_lpips, epoch+1)

        # ---------------------------
        # 每3个epoch进行完整验证（带可视化）
        # ---------------------------
        if (epoch + 1) % 3 == 0 or epoch == TOTAL_EPOCHS - 1:
            print(f"\n🎨 Running full validation with visualization...")
            (val_loss_G, val_loss_D, val_loss_pixel, val_loss_perceptual, val_loss_edge, val_loss_enhance,
             val_psnr, val_ssim, val_lpips, val_musiq, val_nrqm, val_niqe) = validate_gan(
                generator, discriminator, val_loader_full, criterion,
                device, psnr_metric, ssim_metric, lpips_model,
                musiq_model, nrqm_model, niqe_model,
                epoch, vis_save_dir, lambda_adv
            )

            print(
                f"\n📈 Summary Epoch [{epoch+1}/{TOTAL_EPOCHS}]:\n"
                f"   Train → Loss_G: {train_loss_G:.4f}, Loss_D: {train_loss_D:.4f} (Pixel: {train_loss_pixel:.4f}, "
                f"Perc: {train_loss_perceptual:.4f}, Edge: {train_loss_edge:.4f}, Enhance: {train_loss_enhance:.4f})\n"
                f"   Quick Val → Loss_G: {quick_val_loss_G:.4f}, Loss_D: {quick_val_loss_D:.4f}, PSNR: {quick_val_psnr:.2f}\n"
                f"   Full Val  → Loss_G: {val_loss_G:.4f}, Loss_D: {val_loss_D:.4f}\n"
                f"   Metrics → PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}, LPIPS: {val_lpips:.4f}, "
                f"MUSIQ: {val_musiq:.4f}, NRQM: {val_nrqm:.4f}, NIQE: {val_niqe:.4f}"
            )

            # 4. 记录日志到CSV（保存完整验证指标）
            append_csv_log(log_path, epoch+1, val_loss_G, val_loss_D,
                          val_loss_pixel, val_loss_perceptual, val_loss_edge, val_loss_enhance,
                          val_psnr, val_ssim, val_lpips, val_musiq, val_nrqm, val_niqe)
            
            # 5. 记录完整验证结果到TensorBoard
            tb_writer.add_scalar('Loss/val_full_G', val_loss_G, epoch+1)
            tb_writer.add_scalar('Loss/val_full_D', val_loss_D, epoch+1)
            tb_writer.add_scalar('Loss/val_full_pixel', val_loss_pixel, epoch+1)
            tb_writer.add_scalar('Loss/val_full_perceptual', val_loss_perceptual, epoch+1)
            tb_writer.add_scalar('Loss/val_full_edge', val_loss_edge, epoch+1)
            tb_writer.add_scalar('Loss/val_full_enhance', val_loss_enhance, epoch+1)
            tb_writer.add_scalar('Metrics/PSNR_full', val_psnr, epoch+1)
            tb_writer.add_scalar('Metrics/SSIM_full', val_ssim, epoch+1)
            tb_writer.add_scalar('Metrics/LPIPS_full', val_lpips, epoch+1)
            tb_writer.add_scalar('Metrics/MUSIQ_full', val_musiq, epoch+1)
            tb_writer.add_scalar('Metrics/NRQM_full', val_nrqm, epoch+1)
            tb_writer.add_scalar('Metrics/NIQE_full', val_niqe, epoch+1)

            # 6. 双重最佳权重保存策略（每5个epoch定期备份）
            is_periodic = ((epoch + 1) % 5 == 0)
            best_psnr, best_lpips = save_checkpoint_gan(
                generator, discriminator, optimizer_G, optimizer_D,
                scheduler_G, scheduler_D, epoch+1, val_psnr, val_lpips,
                save_dir, best_psnr, best_lpips, is_periodic
            )
        else:
            # 非完整验证epoch，只显示快速验证结果
            print(
                f"\n📈 Summary Epoch [{epoch+1}/{TOTAL_EPOCHS}]:\n"
                f"   Train → Loss_G: {train_loss_G:.4f}, Loss_D: {train_loss_D:.4f} (Pixel: {train_loss_pixel:.4f}, "
                f"Perc: {train_loss_perceptual:.4f}, Edge: {train_loss_edge:.4f}, Enhance: {train_loss_enhance:.4f})\n"
                f"   Quick Val → Loss_G: {quick_val_loss_G:.4f}, Loss_D: {quick_val_loss_D:.4f}, PSNR: {quick_val_psnr:.2f}, "
                f"SSIM: {quick_val_ssim:.4f}, LPIPS: {quick_val_lpips:.4f}\n"
                f"   Full Val  → 跳过 (每3个epoch完整验证)"
            )

        torch.cuda.empty_cache()

    tb_writer.close()
    print("\n" + "="*60)
    print("✅ GAN Training Completed Successfully!")
    print(f"   Best PSNR: {best_psnr:.2f}")
    print(f"   Best LPIPS: {best_lpips:.4f}")
    print(f"   Checkpoints saved to: {save_dir}")
    print(f"   Visualizations saved to: {vis_save_dir}")
    print(f"   Logs saved to: {log_path}")
    print("="*60)


if __name__ == "__main__":
    main()
