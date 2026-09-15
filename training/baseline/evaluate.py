# evaluate.py
# 加载best_model.pt，在完整验证集上评估并打印指标

import os
import yaml
import torch
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import pyiqa

from archs import create_model
from data import create_dataloaders
from losses import create_loss


def load_config(path):
    """读取配置文件"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


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


def validate_full(model, dataloader, criterion, device,
                  psnr_model, ssim_model, lpips_model,
                  musiq_model, nrqm_model, niqe_model):
    """
    完整验证，计算所有指标
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
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                inputs, targets, _ = batch
            else:
                inputs, targets = batch

            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            outputs = torch.clamp(outputs, 0.0, 1.0)
            targets = torch.clamp(targets, 0.0, 1.0)

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

    return (avg_loss, avg_loss_pixel, avg_loss_perceptual, avg_loss_edge, avg_loss_enhance,
            avg_psnr, avg_ssim, avg_lpips, avg_musiq, avg_nrqm, avg_niqe)


def main():
    # 加载配置文件
    opt = load_config("options/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("评估 best_model.pt")
    print("=" * 60)

    # 创建模型
    model, macs, params = create_model(opt['network'], 0)
    model.to(device)
    print(f"✅ Model created: MACs={macs}, Params={params}")

    # 加载 best_model.pt
    checkpoint_path = "experiments/checkpoints/best_model.pt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    print(f"\n📦 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 检查checkpoint格式并提取state_dict
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', 0)
            print(f"   📂 Loaded from checkpoint with 'model_state_dict' key (epoch: {epoch})")
        elif 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
            print(f"   📂 Loaded from checkpoint with 'state_dict' key")
        else:
            pretrained_dict = checkpoint
            print(f"   📂 Loaded checkpoint directly")
    else:
        pretrained_dict = checkpoint
        print(f"   📂 Loaded checkpoint directly")

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

    # 加载IQA指标模型
    print("\n📏 Loading IQA metrics...")
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_model = pyiqa.create_metric('lpips', device=device, pretrained_model_path='../../weights/iqa/lpips.pth')
    musiq_model = pyiqa.create_metric('musiq', device=device, pretrained_model_path='../../weights/iqa/musiq.pth')
    niqe_model = pyiqa.create_metric('niqe', device=device, pretrained_model_path='../../weights/iqa/niqe.mat')
    nrqm_model = pyiqa.create_metric('nrqm', device=device, pretrained_model_path='../../weights/iqa/nrqm.mat')
    print("✅ IQA metrics loaded (PSNR, SSIM, LPIPS, MUSIQ, NRQM, NIQE)")

    # 创建数据加载器 - 完整验证集，所有样本
    print("\n📂 Loading validation dataset (all samples)...")
    val_loader, _ = create_dataloaders(0, 1, opt['datasets'], mode='val', return_path=False, samples_per_scene=-1)
    print(f"✅ Validation dataset loaded: {len(val_loader)} batches")

    # 创建损失函数
    criterion = create_loss(opt['train']['loss'], rank=0)

    # 运行验证
    print("\n" + "=" * 60)
    print("Running Full Validation")
    print("=" * 60)

    (val_loss, val_loss_pixel, val_loss_perceptual, val_loss_edge, val_loss_enhance,
     val_psnr, val_ssim, val_lpips, val_musiq, val_nrqm, val_niqe) = validate_full(
        model, val_loader, criterion, device,
        psnr_metric, ssim_metric, lpips_model,
        musiq_model, nrqm_model, niqe_model
    )

    # 打印结果
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)
    print(f"Loss:        {val_loss:.4f}")
    print(f"  - Pixel:     {val_loss_pixel:.4f}")
    print(f"  - Perceptual:{val_loss_perceptual:.4f}")
    print(f"  - Edge:      {val_loss_edge:.4f}")
    print(f"  - Enhance:   {val_loss_enhance:.4f}")
    print(f"\nMetrics:")
    print(f"  - PSNR:      {val_psnr:.2f}")
    print(f"  - SSIM:      {val_ssim:.4f}")
    print(f"  - LPIPS:     {val_lpips:.4f}")
    print(f"  - MUSIQ:     {val_musiq:.4f}")
    print(f"  - NRQM:      {val_nrqm:.4f}")
    print(f"  - NIQE:      {val_niqe:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
