#DarkIR单独可运行（每1000次输出一次量化结果）


import os
import torch
import yaml
from tqdm import tqdm

from archs import create_model, create_optim_scheduler, resume_model, save_checkpoint
from data import create_dataloaders
from losses import create_loss, calculate_loss

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import pyiqa


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def train_one_epoch_partial(model, dataloader, criterion, optimizer, device,
                            musiq_model, nrqm_model, niqe_model, lpips_model,
                            block_size=100, long_block_size=1000):
    """
    block_size: 每次计算指标的图片数量（100）
    long_block_size: 一个大的循环周期，用于划分训练块（1000）
    """
    model.train()
    total_loss = 0.0
    global_step = 0

    # 指标初始化
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # 用于累积每个block的指标
    block_loss = 0.0
    running_psnr = running_ssim = running_musiq = running_nrqm = running_niqe = running_lpips = 0.0
    block_count = 0

    for i, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = torch.clamp(outputs, 0.0, 1.0)
        targets = torch.clamp(targets, 0.0, 1.0)

        loss = calculate_loss(criterion, outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        global_step += 1

        # 每1000张图片中，前100张既训练又计算指标，后900张只训练不计算指标
        if (global_step % long_block_size) <= block_size and (global_step % long_block_size) != 0:
            # 累积指标（只针对前100张）
            block_loss += loss.item()

            psnr_val = psnr_metric(outputs, targets)
            ssim_val = ssim_metric(outputs, targets)
            musiq_val = musiq_model(outputs)
            nrqm_val = nrqm_model(outputs)
            niqe_val = niqe_model(outputs)
            lpips_val = lpips_model(outputs, targets)

            running_psnr += psnr_val.item()
            running_ssim += ssim_val.item()
            running_musiq += musiq_val.item()
            running_nrqm += nrqm_val.item()
            running_niqe += niqe_val.item()
            running_lpips += lpips_val.item()
            block_count += 1

        # 每1000张图片打印一次指标（基于前100张的计算结果）
        if (global_step % long_block_size) == 0 and global_step > 0:
            print(f"Step [{global_step}/{len(dataloader)}] → "
                  f"Loss(avg over {block_count} imgs): {block_loss/block_count:.4f}, "
                  f"PSNR: {running_psnr/block_count:.2f}, "
                  f"SSIM: {running_ssim/block_count:.4f}, "
                  f"MUSIQ: {running_musiq/block_count:.4f}, "
                  f"NRQM: {running_nrqm/block_count:.4f}, "
                  f"NIQE: {running_niqe/block_count:.4f}, "
                  f"LPIPS: {running_lpips/block_count:.4f}")

            # 重置指标累积
            block_loss = running_psnr = running_ssim = running_musiq = running_nrqm = running_niqe = running_lpips = 0.0
            block_count = 0

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device,
             musiq_model, nrqm_model, niqe_model, lpips_model):
    model.eval()
    total_loss = 0.0
    psnr_sum = ssim_sum = musiq_sum = nrqm_sum = niqe_sum = lpips_sum = 0.0
    count = 0

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = torch.clamp(outputs, 0.0, 1.0)
            targets = torch.clamp(targets, 0.0, 1.0)

            loss = calculate_loss(criterion, outputs, targets)
            total_loss += loss.item()

            psnr_sum += psnr_metric(outputs, targets).item()
            ssim_sum += ssim_metric(outputs, targets).item()
            musiq_sum += musiq_model(outputs).item()
            nrqm_sum += nrqm_model(outputs).item()
            niqe_sum += niqe_model(outputs).item()
            lpips_sum += lpips_model(outputs, targets).item()
            count += 1

    avg_loss = total_loss / len(dataloader)
    avg_psnr = psnr_sum / count
    avg_ssim = ssim_sum / count
    avg_musiq = musiq_sum / count
    avg_nrqm = nrqm_sum / count
    avg_niqe = niqe_sum / count
    avg_lpips = lpips_sum / count

    print(f"Validation → Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.2f}, "
          f"SSIM: {avg_ssim:.4f}, MUSIQ: {avg_musiq:.4f}, "
          f"NRQM: {avg_nrqm:.4f}, NIQE: {avg_niqe:.4f}, LPIPS: {avg_lpips:.4f}")

    return avg_loss, avg_psnr, avg_ssim, avg_musiq, avg_nrqm, avg_niqe, avg_lpips


def main():
    opt = load_config("options/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = 0
    
    # 确保保存目录存在
    os.makedirs(opt['path']['save_dir'], exist_ok=True)
    print(f"📁 保存目录已确认: {opt['path']['save_dir']}")

    print("📏 Loading IQA metrics (offline)...")
    musiq_model = pyiqa.create_metric('musiq', device=device, pretrained_model_path='/root/autodl-tmp/musiq.pth')
    lpips_model = pyiqa.create_metric('lpips', device=device, pretrained_model_path='/root/autodl-tmp/lpips.pth')
    niqe_model  = pyiqa.create_metric('niqe', device=device, pretrained_model_path='/root/autodl-tmp/niqe.mat') 
    nrqm_model  = pyiqa.create_metric('nrqm', device=device, pretrained_model_path='/root/autodl-tmp/nrqm.mat')
    print("✅ IQA metrics loaded successfully.")

    # 创建模型与数据
    model, macs, params = create_model(opt['network'], rank)
    model.to(device)

    world_size = 1
    train_loader, _ = create_dataloaders(rank, world_size, opt['datasets'], mode='train')
    val_loader, _ = create_dataloaders(rank, world_size, opt['datasets'], mode='val')

    criterion = create_loss(opt['train']['loss'], rank=0)
    optimizer, scheduler = create_optim_scheduler(opt['train'], model)

    path_model = opt['path']['resume_path']
    start_epoch = 0
    best_psnr = 0.0

    if path_model is not None and os.path.exists(path_model):
        print(f"\n📦 Loading model from: {path_model}")
        ckpt = torch.load(path_model, map_location='cpu')
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model, optimizer, scheduler, start_epoch = resume_model(
                model, optimizer, scheduler, path_model, rank, opt['train']['resume']
            )
        else:
            model.load_state_dict(ckpt, strict=False)
        print(f"✅ Model resumed from epoch {start_epoch}.")
    else:
        print("\n⚠️ No valid resume path found. Training from scratch.")

    # 主训练循环（去除初始验证，只在每个epoch结束后验证）
    for epoch in range(start_epoch, opt['train']['epochs']):
        print(f"\nEpoch [{epoch+1}/{opt['train']['epochs']}]")
        train_loss = train_one_epoch_partial(
                    model, train_loader, criterion, optimizer, device,
                    musiq_model, nrqm_model, niqe_model, lpips_model,
                    block_size=100, long_block_size=1000
                )

        scheduler.step()

        # epoch结束后用验证集算指标
        val_loss, val_psnr, val_ssim, val_musiq, val_nrqm, val_niqe, val_lpips = validate(
            model, val_loader, criterion, device, musiq_model, nrqm_model, niqe_model, lpips_model
        )

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}, MUSIQ: {val_musiq:.4f}, "
              f"NRQM: {val_nrqm:.4f}, NIQE: {val_niqe:.4f}, LPIPS: {val_lpips:.4f}")

        metrics_train = {'epoch': epoch, 'train_loss': train_loss, 'best_psnr': best_psnr}
        metrics_eval = {'valid_psnr': val_psnr}
        paths = {
            'new': os.path.join(opt['path']['save_dir'], f"epoch_{epoch+1}.pt"),
            'best': os.path.join(opt['path']['save_dir'], "best_model.pt"),
        }
        best_psnr = save_checkpoint(model, optimizer, scheduler, metrics_eval, metrics_train, paths, rank=rank)

    print("\n✅ Training Completed.")


if __name__ == "__main__":
    main()
