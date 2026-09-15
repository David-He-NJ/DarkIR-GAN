# train_baseline.py
# Baseline 版本训练脚本 - 与 GAN 版本保持一致的记录和可视化逻辑

import os
import csv
import yaml
import torch
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
def save_comparison_image(inputs, targets, outputs, save_path, epoch, current_id):
    """
    只保存模型输出图（Output）
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 归一化到[0, 1]
    outputs = torch.clamp(outputs, 0, 1)

    # 只保存输出图像
    vutils.save_image(outputs, save_path, nrow=1, normalize=False)
    print(f"📸 Saved output image: {save_path}")


# ---------------------------
# 安全指标计算函数（避免奇异矩阵）
# ---------------------------
def safe_iqa_metrics(outputs, target_images, device,
                     psnr_model=None, ssim_model=None, lpips_model=None,
                     musiq_model=None, nrqm_model=None, niqe_model=None):
    """
    安全地计算 IQA 指标，避免奇异矩阵问题
    """
    results = {'psnr': 0, 'ssim': 0, 'lpips': 0, 'musiq': 0, 'nrqm': 0, 'niqe': 0}

    outputs = torch.clamp(outputs, 0, 1)
    target_images = torch.clamp(target_images, 0, 1)

    try:
        if psnr_model:
            results['psnr'] = psnr_model(outputs, target_images).item()
    except Exception:
        pass

    try:
        if ssim_model:
            results['ssim'] = ssim_model(outputs, target_images).item()
    except Exception:
        pass

    try:
        if lpips_model:
            results['lpips'] = lpips_model(outputs, target_images).item()
    except Exception:
        pass

    try:
        if musiq_model:
            results['musiq'] = musiq_model(outputs).item()
    except Exception:
        pass

    try:
        if nrqm_model:
            results['nrqm'] = nrqm_model(outputs).item()
    except Exception:
        pass

    try:
        if niqe_model:
            results['niqe'] = niqe_model(outputs).item()
    except Exception:
        pass

    return results


# ---------------------------
# 训练一个 epoch（Baseline 模式）
# ---------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device,
                    block_size=100):
    """
    每个 epoch 的训练：只进行训练和计算 loss，不计算指标
    """
    model.train()
    total_loss = 0.0
    total_loss_pixel = 0.0
    total_loss_perceptual = 0.0
    total_loss_edge = 0.0
    total_loss_enhance = 0.0

    for i, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = torch.clamp(outputs, 0.0, 1.0)
        targets = torch.clamp(targets, 0.0, 1.0)

        # 计算DarkIR复合损失
        # Pixel loss
        loss_pixel = criterion['pixel_loss'](outputs, targets)

        # Perceptual loss (LPIPS)
        loss_perceptual = 0.0
        if 'perceptual_loss' in criterion:
            loss_perceptual = criterion['perceptual_loss'](outputs, targets)

        # Edge loss
        loss_edge = 0.0
        if 'edge_loss' in criterion:
            loss_edge = criterion['edge_loss'](outputs, targets)

        # Enhance loss
        loss_enhance = 0.0
        if 'enhance_loss' in criterion:
            loss_enhance = criterion['enhance_loss'](outputs, targets)

        # 总损失 - 权重从配置文件读取
        loss = (loss_pixel +
                (0.1 * loss_perceptual if not isinstance(loss_perceptual, float) else 0) +
                (5.0 * loss_edge if not isinstance(loss_edge, float) else 0) +
                (0.5 * loss_enhance if not isinstance(loss_enhance, float) else 0))

        loss.backward()
        optimizer.step()

        # 累积损失
        total_loss += loss.item()
        total_loss_pixel += loss_pixel.item()
        total_loss_perceptual += loss_perceptual.item() if not isinstance(loss_perceptual, float) else 0
        total_loss_edge += loss_edge.item() if not isinstance(loss_edge, float) else 0
        total_loss_enhance += loss_enhance.item() if not isinstance(loss_enhance, float) else 0

    # epoch 结束时打印平均损失
    avg_loss = total_loss / len(dataloader)
    avg_loss_pixel = total_loss_pixel / len(dataloader)
    avg_loss_perceptual = total_loss_perceptual / len(dataloader)
    avg_loss_edge = total_loss_edge / len(dataloader)
    avg_loss_enhance = total_loss_enhance / len(dataloader)

    print(
        f"Epoch Training Loss: {avg_loss:.4f}, "
        f"Pixel: {avg_loss_pixel:.4f}, "
        f"Perc: {avg_loss_perceptual:.4f}, "
        f"Edge: {avg_loss_edge:.4f}, "
        f"Enhance: {avg_loss_enhance:.4f}"
    )

    return (avg_loss, avg_loss_pixel, avg_loss_perceptual, avg_loss_edge, avg_loss_enhance)


# ---------------------------
# 快速验证阶段（每个场景选15张）
# ---------------------------
def validate_quick(model, dataloader, criterion, device,
                   psnr_model, ssim_model, lpips_model,
                   musiq_model, nrqm_model, niqe_model,
                   epoch, desc="Quick Val"):
    """
    快速验证：计算指标但不保存可视化结果
    用于每个epoch后的快速评估
    """
    model.eval()

    # 调试：检查模型状态
    print(f"🔍 Debug - Model training mode: {model.training}")
    print(f"🔍 Debug - Model device: {next(model.parameters()).device}")

    # 检查权重是否与加载时一致
    intro_weight = model.state_dict()['intro.weight']
    print(f"🔍 Debug - intro.weight mean: {intro_weight.mean():.6f}")

    total_loss = 0.0
    total_loss_pixel = 0.0
    total_loss_perceptual = 0.0
    total_loss_edge = 0.0
    total_loss_enhance = 0.0
    psnr_sum = ssim_sum = lpips_sum = musiq_sum = nrqm_sum = niqe_sum = 0.0
    count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=desc, leave=False)):
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                inputs, targets, _ = batch
            else:
                inputs, targets = batch

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # 调试：打印第一个batch的数值范围
            if batch_idx == 0:
                print(f"🔍 Debug - inputs range: [{inputs.min():.4f}, {inputs.max():.4f}], mean={inputs.mean():.4f}")
                print(f"🔍 Debug - outputs range (before clamp): [{outputs.min():.4f}, {outputs.max():.4f}], mean={outputs.mean():.4f}")

            outputs = torch.clamp(outputs, 0.0, 1.0)
            targets = torch.clamp(targets, 0.0, 1.0)

            if batch_idx == 0:
                print(f"🔍 Debug - outputs range (after clamp): [{outputs.min():.4f}, {outputs.max():.4f}]")
                print(f"🔍 Debug - targets range: [{targets.min():.4f}, {targets.max():.4f}], mean={targets.mean():.4f}")

            # 计算损失
            loss_pixel = criterion['pixel_loss'](outputs, targets)

            loss_perceptual = 0.0
            if 'perceptual_loss' in criterion:
                loss_perceptual = criterion['perceptual_loss'](outputs, targets)

            loss_edge = 0.0
            if 'edge_loss' in criterion:
                loss_edge = criterion['edge_loss'](outputs, targets)

            loss_enhance = 0.0
            if 'enhance_loss' in criterion:
                loss_enhance = criterion['enhance_loss'](outputs, targets)

            loss = (loss_pixel +
                    (0.1 * loss_perceptual if not isinstance(loss_perceptual, float) else 0) +
                    (5.0 * loss_edge if not isinstance(loss_edge, float) else 0) +
                    (0.5 * loss_enhance if not isinstance(loss_enhance, float) else 0))

            total_loss += loss.item()
            total_loss_pixel += loss_pixel.item()
            total_loss_perceptual += loss_perceptual.item() if not isinstance(loss_perceptual, float) else 0
            total_loss_edge += loss_edge.item() if not isinstance(loss_edge, float) else 0
            total_loss_enhance += loss_enhance.item() if not isinstance(loss_enhance, float) else 0

            metrics = safe_iqa_metrics(
                outputs, targets, device,
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

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
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
        f"{desc} → Loss: {avg_loss:.4f}, "
        f"PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}, "
        f"MUSIQ: {avg_musiq:.4f}, NRQM: {avg_nrqm:.4f}, NIQE: {avg_niqe:.4f}"
    )

    return (avg_loss, avg_loss_pixel, avg_loss_perceptual, avg_loss_edge, avg_loss_enhance,
            avg_psnr, avg_ssim, avg_lpips, avg_musiq, avg_nrqm, avg_niqe)


# ---------------------------
# 完整验证阶段（Baseline 模式，带可视化）
# ---------------------------
def validate(model, dataloader, criterion, device,
             psnr_model, ssim_model, lpips_model,
             musiq_model, nrqm_model, niqe_model,
             epoch, vis_save_dir):
    """
    验证阶段，计算指标并保存特定场景的可视化结果
    """
    model.eval()
    total_loss = 0.0
    total_loss_pixel = 0.0
    total_loss_perceptual = 0.0
    total_loss_edge = 0.0
    total_loss_enhance = 0.0
    psnr_sum = ssim_sum = lpips_sum = musiq_sum = nrqm_sum = niqe_sum = 0.0
    count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating", leave=False)):
            # 检查batch是否包含路径信息
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                inputs, targets, paths = batch
            else:
                inputs, targets = batch
                paths = None

            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            outputs = torch.clamp(outputs, 0.0, 1.0)
            targets = torch.clamp(targets, 0.0, 1.0)

            # 计算DarkIR复合损失
            # Pixel loss
            loss_pixel = criterion['pixel_loss'](outputs, targets)

            # Perceptual loss (LPIPS)
            loss_perceptual = 0.0
            if 'perceptual_loss' in criterion:
                loss_perceptual = criterion['perceptual_loss'](outputs, targets)

            # Edge loss
            loss_edge = 0.0
            if 'edge_loss' in criterion:
                loss_edge = criterion['edge_loss'](outputs, targets)

            # Enhance loss
            loss_enhance = 0.0
            if 'enhance_loss' in criterion:
                loss_enhance = criterion['enhance_loss'](outputs, targets)

            # 总损失 - 权重从配置文件读取
            loss = (loss_pixel +
                    (0.1 * loss_perceptual if not isinstance(loss_perceptual, float) else 0) +
                    (5.0 * loss_edge if not isinstance(loss_edge, float) else 0) +
                    (0.5 * loss_enhance if not isinstance(loss_enhance, float) else 0))

            total_loss += loss.item()
            total_loss_pixel += loss_pixel.item()
            total_loss_perceptual += loss_perceptual.item() if not isinstance(loss_perceptual, float) else 0
            total_loss_edge += loss_edge.item() if not isinstance(loss_edge, float) else 0
            total_loss_enhance += loss_enhance.item() if not isinstance(loss_enhance, float) else 0

            metrics = safe_iqa_metrics(
                outputs, targets, device,
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
                            outputs[j:j+1],
                            save_path,
                            epoch+1,
                            current_id
                        )

    avg_loss = total_loss / len(dataloader)
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
        f"Validation → Loss: {avg_loss:.4f}, "
        f"PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}, "
        f"MUSIQ: {avg_musiq:.4f}, NRQM: {avg_nrqm:.4f}, NIQE: {avg_niqe:.4f}"
    )

    return (avg_loss, avg_loss_pixel, avg_loss_perceptual, avg_loss_edge, avg_loss_enhance,
            avg_psnr, avg_ssim, avg_lpips, avg_musiq, avg_nrqm, avg_niqe)


# ---------------------------
# 初始化CSV日志
# ---------------------------
def init_csv_logger(log_path):
    """初始化CSV日志文件"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'val_loss', 'val_loss_pixel', 'val_loss_perceptual',
                         'val_loss_edge', 'val_loss_enhance',
                         'val_psnr', 'val_ssim', 'val_lpips', 'val_musiq', 'val_nrqm', 'val_niqe'])


# ---------------------------
# 追加CSV日志
# ---------------------------
def append_csv_log(log_path, epoch, val_loss, val_loss_pixel, val_loss_perceptual,
                   val_loss_edge, val_loss_enhance, val_psnr, val_ssim, val_lpips,
                   val_musiq, val_nrqm, val_niqe):
    """追加一行训练日志到CSV，包含所有验证指标"""
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, val_loss, val_loss_pixel, val_loss_perceptual,
                        val_loss_edge, val_loss_enhance, val_psnr, val_ssim, val_lpips,
                        val_musiq, val_nrqm, val_niqe])


# ---------------------------
# 保存检查点（双重最佳策略）
# ---------------------------
def save_checkpoint_baseline(model, optimizer, scheduler, epoch, val_psnr, val_lpips,
                             save_dir, best_psnr, best_lpips, is_periodic=False):
    """
    Baseline 双重最佳权重保存策略
    - baseline_best_psnr.pt: PSNR最高时的模型
    - baseline_best_lpips.pt: LPIPS最低时的模型
    - baseline_epoch_{epoch}.pt: 每5个Epoch定期备份
    """
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_psnr': val_psnr,
        'val_lpips': val_lpips,
    }

    # 定期备份（每5个epoch）
    if is_periodic:
        periodic_path = os.path.join(save_dir, f"baseline_epoch_{epoch}.pt")
        torch.save(checkpoint, periodic_path)
        print(f"💾 Periodic checkpoint saved: {periodic_path}")

    # 保存最佳PSNR模型
    if val_psnr > best_psnr:
        best_psnr_path = os.path.join(save_dir, "baseline_best_psnr.pt")
        torch.save(checkpoint, best_psnr_path)
        print(f"🏆 New best PSNR: {val_psnr:.2f}! Saved to {best_psnr_path}")
        best_psnr = val_psnr

    # 保存最佳LPIPS模型
    if val_lpips < best_lpips:
        best_lpips_path = os.path.join(save_dir, "baseline_best_lpips.pt")
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
    # 1. 设置核心超参数
    # ---------------------------
    TOTAL_EPOCHS = 30  # 训练epoch为30
    LEARNING_RATE = 1e-4  # 初始学习率，使用动态余弦退火
    ETA_MIN = 1e-6  # 最小学习率

    # 覆盖配置中的参数
    opt['train']['epochs'] = TOTAL_EPOCHS
    opt['train']['lr_initial'] = LEARNING_RATE
    opt['train']['eta_min'] = ETA_MIN

    print(f"🎯 Baseline Training Configuration:")
    print(f"   - Total Epochs: {TOTAL_EPOCHS}")
    print(f"   - Learning Rate: {LEARNING_RATE}")
    print(f"   - No Adversarial Loss (Baseline Mode)")

    # 创建模型
    model, macs, params = create_model(opt['network'], rank)
    model.to(device)

    # 加载IQA指标模型
    print("📏 Loading IQA metrics...")
    lpips_model = pyiqa.create_metric('lpips', device=device, pretrained_model_path='../../weights/iqa/lpips.pth')
    musiq_model = pyiqa.create_metric('musiq', device=device, pretrained_model_path='../../weights/iqa/musiq.pth')
    niqe_model  = pyiqa.create_metric('niqe', device=device, pretrained_model_path='../../weights/iqa/niqe.mat')
    nrqm_model  = pyiqa.create_metric('nrqm', device=device, pretrained_model_path='../../weights/iqa/nrqm.mat')
    print("✅ IQA metrics loaded (PSNR, SSIM, LPIPS, MUSIQ, NRQM, NIQE).")

    # 创建数据加载器
    # 训练集：每个场景选择15张图片
    train_loader, train_sampler = create_dataloaders(rank, 1, opt['datasets'], mode='train', return_path=False, samples_per_scene=15)
    # 快速验证集：每个场景选15张，不返回路径
    val_loader_quick, _ = create_dataloaders(rank, 1, opt['datasets'], mode='val', return_path=False, samples_per_scene=15)
    # 完整验证集：所有图片都跑，返回路径用于可视化（每3个epoch使用）
    val_loader_full, _ = create_dataloaders(rank, 1, opt['datasets'], mode='val', return_path=True, samples_per_scene=-1)

    # 创建损失函数和优化器
    criterion = create_loss(opt['train']['loss'], rank=0)
    optimizer, scheduler = create_optim_scheduler(opt['train'], model)

    # 不再强制覆盖优化器学习率，让调度器管理
    print(f"✅ Optimizer learning rate: {LEARNING_RATE}, scheduler: CosineAnnealing to {ETA_MIN}")

    # ========== 加载检查点并继续训练 ==========
    checkpoint_path = opt['path'].get('resume_path', '')
    start_epoch = 0
    loaded_epoch = 0
    loaded_best_psnr = 0.0
    loaded_best_lpips = float('inf')

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n📦 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 检查checkpoint格式并提取state_dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                pretrained_dict = checkpoint['model_state_dict']
                loaded_epoch = checkpoint.get('epoch', 0)
                loaded_best_psnr = checkpoint.get('val_psnr', 0.0)
                loaded_best_lpips = checkpoint.get('val_lpips', float('inf'))
                print(f"   📂 Loaded from checkpoint with 'model_state_dict' key (previous epoch: {loaded_epoch})")

                # 加载优化器状态（如果存在）
                if 'optimizer_state_dict' in checkpoint:
                    try:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        print(f"   📂 Loaded optimizer state")
                    except Exception as e:
                        print(f"   ⚠️ Failed to load optimizer state: {e}")

                # 加载调度器状态（如果存在）
                if 'scheduler_state_dict' in checkpoint:
                    try:
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                        print(f"   📂 Loaded scheduler state")
                    except Exception as e:
                        print(f"   ⚠️ Failed to load scheduler state: {e}")

            elif 'state_dict' in checkpoint:
                pretrained_dict = checkpoint['state_dict']
                print(f"   📂 Loaded from checkpoint with 'state_dict' key")
            else:
                pretrained_dict = checkpoint
                print(f"   📂 Loaded checkpoint directly")
        else:
            pretrained_dict = checkpoint

        model_dict = model.state_dict()

        # 检查匹配的层数和形状
        matched_layers = 0
        shape_mismatches = []
        for k in pretrained_dict:
            if k in model_dict:
                if pretrained_dict[k].shape == model_dict[k].shape:
                    matched_layers += 1
                else:
                    shape_mismatches.append(f"{k}: checkpoint {pretrained_dict[k].shape} vs model {model_dict[k].shape}")

        model.load_state_dict(pretrained_dict, strict=True)
        print(f"✅ Loaded checkpoint from {checkpoint_path}.")
        print(f"   📊 匹配层数: {matched_layers}/{len(pretrained_dict)} (checkpoint) | 模型总层数: {len(model_dict)}")
        if shape_mismatches:
            print(f"   ⚠️ 形状不匹配的层: {len(shape_mismatches)}")
            for m in shape_mismatches[:5]:
                print(f"      - {m}")

        # 验证权重是否正确加载
        print(f"   🔍 验证权重加载...")
        test_param = model.state_dict()['intro.weight']
        ckpt_param = pretrained_dict['intro.weight']
        if torch.allclose(test_param.cpu(), ckpt_param.cpu()):
            print(f"   ✔️ 权重验证通过 (intro.weight 匹配)")
        else:
            print(f"   ❌ 权重验证失败！intro.weight 不匹配")

        # 重置 start_epoch 为 0，重新开始训练 30 个 epoch
        print(f"\n🔄 继续训练: 从 epoch {loaded_epoch} 的检查点加载")
        print(f"   检查点指标: PSNR={loaded_best_psnr:.2f}, LPIPS={loaded_best_lpips:.4f}")
        print(f"   将再训练 {TOTAL_EPOCHS} 个 epoch (epoch 1 to {TOTAL_EPOCHS})")
        start_epoch = 0  # 重置为 0，再训练 30 个 epoch
    else:
        print(f"⚠️ Checkpoint not found at {checkpoint_path}, training from scratch.")
        print(f"   💡 Hint: Set resume_path in config.yaml to the checkpoint path (e.g., './experiments/checkpoints/baseline_best_psnr.pt')")
    # =============================================================

    # ---------------------------
    # 2. 创建实验目录和日志系统
    # ---------------------------
    experiment_dir = 'experiments'
    vis_save_dir = os.path.join(experiment_dir, 'Baseline_vis')
    save_dir = os.path.join(experiment_dir, 'checkpoints')
    log_path = os.path.join(experiment_dir, 'baseline_training_log.csv')

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(vis_save_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # 初始化CSV日志
    init_csv_logger(log_path)
    print(f"📝 CSV logger initialized: {log_path}")

    # 初始化TensorBoard
    tb_writer = SummaryWriter(log_dir=os.path.join(experiment_dir, 'tensorboard_baseline'))
    print(f"📊 TensorBoard initialized: {os.path.join(experiment_dir, 'tensorboard_baseline')}")

    # 初始化最佳指标记录（使用从检查点加载的值，如果没有则使用默认值）
    best_psnr = loaded_best_psnr if loaded_best_psnr > 0 else 0.0
    best_lpips = loaded_best_lpips if loaded_best_lpips < float('inf') else float('inf')
    print(f"🎯 Initial best metrics: PSNR={best_psnr:.2f}, LPIPS={best_lpips:.4f}")

    # 评估加载的模型
    print("\n📝 Evaluating loaded model (quick validation)...")
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    validate_quick(model, val_loader_quick, criterion, device,
                   psnr_metric, ssim_metric, lpips_model,
                   musiq_model, nrqm_model, niqe_model,
                   start_epoch, desc="Init Val (Loaded)")

    print(f"\n🎯 Target scenes for visualization: {TARGET_IDS}")

    # 创建指标计算模型
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # ---------------------------
    # 3. 训练循环
    # ---------------------------
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch [{epoch+1}/{TOTAL_EPOCHS}]")
        print(f"{'='*60}")

        # 在每个epoch开始时，如果sampler有set_epoch方法，则调用它进行重新采样
        if hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)
            print(f"🔄 Resampled training data: {len(train_sampler)} samples for this epoch")

        # 训练一个epoch
        (train_loss, train_loss_pixel, train_loss_perceptual,
         train_loss_edge, train_loss_enhance) = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        scheduler.step()

        # ---------------------------
        # 每个epoch都进行快速验证
        # ---------------------------
        print(f"\n📝 Running quick validation after epoch {epoch+1}...")
        (quick_val_loss, quick_val_loss_pixel, quick_val_loss_perceptual,
         quick_val_loss_edge, quick_val_loss_enhance,
         quick_val_psnr, quick_val_ssim, quick_val_lpips,
         quick_val_musiq, quick_val_nrqm, quick_val_niqe) = validate_quick(
            model, val_loader_quick, criterion, device,
            psnr_metric, ssim_metric, lpips_model,
            musiq_model, nrqm_model, niqe_model,
            epoch, desc=f"Epoch {epoch+1} Quick Val"
        )

        # 记录训练损失到TensorBoard（每个epoch都记录）
        tb_writer.add_scalar('Loss/train_total', train_loss, epoch+1)
        tb_writer.add_scalar('Loss/train_pixel', train_loss_pixel, epoch+1)
        tb_writer.add_scalar('Loss/train_perceptual', train_loss_perceptual, epoch+1)
        tb_writer.add_scalar('Loss/train_edge', train_loss_edge, epoch+1)
        tb_writer.add_scalar('Loss/train_enhance', train_loss_enhance, epoch+1)

        # 记录快速验证结果
        tb_writer.add_scalar('Loss/val_quick_total', quick_val_loss, epoch+1)
        tb_writer.add_scalar('Metrics/PSNR_quick', quick_val_psnr, epoch+1)
        tb_writer.add_scalar('Metrics/SSIM_quick', quick_val_ssim, epoch+1)
        tb_writer.add_scalar('Metrics/LPIPS_quick', quick_val_lpips, epoch+1)

        # ---------------------------
        # 每3个epoch进行完整验证（带可视化）
        # ---------------------------
        if (epoch + 1) % 3 == 0 or epoch == TOTAL_EPOCHS - 1:
            print(f"\n🎨 Running full validation with visualization...")
            (val_loss, val_loss_pixel, val_loss_perceptual, val_loss_edge, val_loss_enhance,
             val_psnr, val_ssim, val_lpips, val_musiq, val_nrqm, val_niqe) = validate(
                model, val_loader_full, criterion, device,
                psnr_metric, ssim_metric, lpips_model,
                musiq_model, nrqm_model, niqe_model,
                epoch, vis_save_dir
            )

            print(
                f"\n📈 Summary Epoch [{epoch+1}/{TOTAL_EPOCHS}]:\n"
                f"   Train → Loss: {train_loss:.4f} (Pixel: {train_loss_pixel:.4f}, "
                f"Perc: {train_loss_perceptual:.4f}, Edge: {train_loss_edge:.4f}, Enhance: {train_loss_enhance:.4f})\n"
                f"   Quick Val → Loss: {quick_val_loss:.4f}, PSNR: {quick_val_psnr:.2f}\n"
                f"   Full Val  → Loss: {val_loss:.4f}\n"
                f"   Metrics → PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}, LPIPS: {val_lpips:.4f}, "
                f"MUSIQ: {val_musiq:.4f}, NRQM: {val_nrqm:.4f}, NIQE: {val_niqe:.4f}"
            )

            # 4. 记录日志到CSV（保存完整验证指标）
            append_csv_log(log_path, epoch+1, val_loss, val_loss_pixel, val_loss_perceptual,
                          val_loss_edge, val_loss_enhance, val_psnr, val_ssim, val_lpips,
                          val_musiq, val_nrqm, val_niqe)

            # 5. 记录完整验证结果到TensorBoard
            tb_writer.add_scalar('Loss/val_full_total', val_loss, epoch+1)
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

            # 6. 双重最佳权重保存策略
            is_periodic = True
            best_psnr, best_lpips = save_checkpoint_baseline(
                model, optimizer, scheduler, epoch+1, val_psnr, val_lpips,
                save_dir, best_psnr, best_lpips, is_periodic
            )
        else:
            # 非完整验证epoch，只显示快速验证结果
            print(
                f"\n📈 Summary Epoch [{epoch+1}/{TOTAL_EPOCHS}]:\n"
                f"   Train → Loss: {train_loss:.4f} (Pixel: {train_loss_pixel:.4f}, "
                f"Perc: {train_loss_perceptual:.4f}, Edge: {train_loss_edge:.4f}, Enhance: {train_loss_enhance:.4f})\n"
                f"   Quick Val → Loss: {quick_val_loss:.4f}, PSNR: {quick_val_psnr:.2f}, "
                f"SSIM: {quick_val_ssim:.4f}, LPIPS: {quick_val_lpips:.4f}\n"
                f"   Full Val  → 跳过 (每3个epoch完整验证)"
            )

        torch.cuda.empty_cache()

    tb_writer.close()
    print("\n" + "="*60)
    print("✅ Baseline Training Completed Successfully!")
    print(f"   Best PSNR: {best_psnr:.2f}")
    print(f"   Best LPIPS: {best_lpips:.4f}")
    print(f"   Checkpoints saved to: {save_dir}")
    print(f"   Visualizations saved to: {vis_save_dir}")
    print(f"   Logs saved to: {log_path}")
    print("="*60)


if __name__ == "__main__":
    main()
