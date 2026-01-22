import sys
import os
import torch
import torchvision
import numpy as np
import pydicom
import glob
from torch.utils.data import DataLoader, random_split, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
# --- 关键改动：引入 CSVLogger ---
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning import Trainer
import torchvision.transforms.functional as F
from torchvision import transforms

# --- 路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.core.reconstruction import SRTrainer


# ... (CTInMemoryDataset 类保持不变，请保留之前的代码) ...
class CTInMemoryDataset(Dataset):
    def __init__(self, data_dir, phase='train', crop_size=128, scale_factor=4):
        super().__init__()
        self.phase = phase
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        dcm_files = sorted(glob.glob(os.path.join(data_dir, "**/*.dcm"), recursive=True))
        if len(dcm_files) == 0: raise ValueError(f"No .dcm files found in {data_dir}")
        print(f"[{phase}] 正在将 {len(dcm_files)} 张 DICOM 预加载到内存 (极速模式)...")
        self.data_cache = []
        for dcm_path in dcm_files:
            try:
                ds = pydicom.dcmread(dcm_path)
                try:
                    ds.decompress()
                except:
                    pass
                if hasattr(ds, 'pixel_array'):
                    slope = getattr(ds, 'RescaleSlope', 1.0)
                    intercept = getattr(ds, 'RescaleIntercept', 0.0)
                    img_hu = ds.pixel_array.astype(np.float32) * slope + intercept
                    img_norm = np.clip(img_hu, -1000, 2000)
                    img_norm = (img_norm + 1000) / 3000.0
                    tensor = torch.from_numpy(img_norm).unsqueeze(0).float()
                    self.data_cache.append(tensor)
            except Exception:
                pass
        print(f"[{phase}] 预加载完成！有效图片: {len(self.data_cache)} 张")

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, index):
        hr_tensor = self.data_cache[index]
        if hr_tensor.shape[1] < self.crop_size or hr_tensor.shape[2] < self.crop_size:
            return self.__getitem__(np.random.randint(0, len(self.data_cache)))
        if self.phase == 'train':
            i, j, h, w = transforms.RandomCrop.get_params(hr_tensor, output_size=(self.crop_size, self.crop_size))
            hr_tensor = F.crop(hr_tensor, i, j, h, w)
            if torch.rand(1) < 0.5: hr_tensor = F.hflip(hr_tensor)
            if torch.rand(1) < 0.5: hr_tensor = F.vflip(hr_tensor)
        lr_h, lr_w = hr_tensor.shape[1] // self.scale_factor, hr_tensor.shape[2] // self.scale_factor
        lr_tensor = torch.nn.functional.interpolate(hr_tensor.unsqueeze(0), size=(lr_h, lr_w), mode='bicubic',
                                                    align_corners=False).squeeze(0)
        return {"lr": lr_tensor, "hr": hr_tensor}


# ... (ImageLogger 类保持不变，请保留之前的代码) ...
class ImageLogger(Callback):
    def __init__(self, num_samples=4):
        super().__init__()
        self.num_samples = num_samples

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:
            try:
                lr, hr = batch["lr"][:self.num_samples], batch["hr"][:self.num_samples]
                with torch.no_grad():
                    sr = pl_module(lr)
                lr_upscaled = torch.nn.functional.interpolate(lr, size=hr.shape[2:], mode='nearest')
                grid_lr = torchvision.utils.make_grid(lr_upscaled, nrow=4, normalize=True)
                grid_sr = torchvision.utils.make_grid(sr, nrow=4, normalize=True)
                grid_hr = torchvision.utils.make_grid(hr, nrow=4, normalize=True)
                trainer.logger[0].experiment.add_image('1_Input_LowRes', grid_lr, trainer.global_step)
                trainer.logger[0].experiment.add_image('2_Output_SuperRes', grid_sr, trainer.global_step)
                trainer.logger[0].experiment.add_image('3_GroundTruth_HighRes', grid_hr, trainer.global_step)
            except Exception:
                pass


def main():
    DATA_DIR = os.path.join(project_root, "data", "raw")
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    MAX_EPOCHS = 300
    LR = 1e-4
    CROP_SIZE = 128
    SCALE_FACTOR = 4

    pl.seed_everything(42)

    print(f"🚀 初始化极速数据加载器...")
    full_dataset = CTInMemoryDataset(DATA_DIR, phase='train', crop_size=CROP_SIZE, scale_factor=SCALE_FACTOR)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                            persistent_workers=True if NUM_WORKERS > 0 else False)

    model = SRTrainer(lr=LR)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(project_root, "checkpoints", "sr_model"),
        filename='srgan-{epoch:02d}-{val_psnr:.2f}',  # 名字里带上 PSNR
        save_top_k=3,
        monitor='val_psnr',  # 监控 PSNR
        mode='max'  # PSNR 越大越好
    )

    # --- 日志记录器设置 ---
    tb_logger = TensorBoardLogger(os.path.join(project_root, "logs"), name="sr_tb_logs")
    # CSV Logger: 保存 excel 数据的地方
    csv_logger = CSVLogger(os.path.join(project_root, "logs"), name="sr_csv_logs")

    img_logger = ImageLogger(num_samples=4)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        # 同时使用两个 logger
        logger=[tb_logger, csv_logger],
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step'), img_logger],
        log_every_n_steps=5,
        check_val_every_n_epoch=1,
        precision="16-mixed"
    )

    print("🚀 开始训练 (极速版 + CSV记录)...")
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()