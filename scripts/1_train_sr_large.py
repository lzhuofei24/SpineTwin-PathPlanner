import sys
import os
import torch
import torchvision
import numpy as np
import pydicom
import glob
from datetime import datetime  # <--- 新增时间模块
from torch.utils.data import DataLoader, random_split, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning import Trainer
import torchvision.transforms.functional as F
from torchvision import transforms

# --- 路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.core.reconstruction import SRTrainer


# ==========================================
# Dataset (保持不变)
# ==========================================
class CTLargeDataset(Dataset):
    def __init__(self, data_dir, phase='train', crop_size=128, scale_factor=4):
        super().__init__()
        self.phase = phase
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        print(f"[{phase}] 正在扫描路径: {data_dir}")
        self.dcm_files = sorted(glob.glob(os.path.join(data_dir, "**/*.dcm"), recursive=True))
        if len(self.dcm_files) == 0:
            raise ValueError(f"错误: 在路径 {data_dir} 下没有找到任何 .dcm 文件！")
        print(f"[{phase}] 扫描完成！共发现 {len(self.dcm_files)} 张 DICOM 切片。")
        print(f"[{phase}] 模式: 硬盘直读 (Disk I/O) - 内存占用将严格限制。")

    def __len__(self):
        return len(self.dcm_files)

    def __getitem__(self, index):
        dcm_path = self.dcm_files[index]
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
                hr_tensor = torch.from_numpy(img_norm).unsqueeze(0).float()
            else:
                return self.__getitem__(np.random.randint(0, len(self.dcm_files)))

            if hr_tensor.shape[1] < self.crop_size or hr_tensor.shape[2] < self.crop_size:
                return self.__getitem__(np.random.randint(0, len(self.dcm_files)))

            if self.phase == 'train':
                i, j, h, w = transforms.RandomCrop.get_params(hr_tensor, output_size=(self.crop_size, self.crop_size))
                hr_tensor = F.crop(hr_tensor, i, j, h, w)
                if torch.rand(1) < 0.5: hr_tensor = F.hflip(hr_tensor)
                if torch.rand(1) < 0.5: hr_tensor = F.vflip(hr_tensor)

            lr_h, lr_w = hr_tensor.shape[1] // self.scale_factor, hr_tensor.shape[2] // self.scale_factor
            lr_tensor = torch.nn.functional.interpolate(hr_tensor.unsqueeze(0), size=(lr_h, lr_w), mode='bicubic',
                                                        align_corners=False).squeeze(0)
            return {"lr": lr_tensor, "hr": hr_tensor}
        except Exception:
            return self.__getitem__(np.random.randint(0, len(self.dcm_files)))


# ==========================================
# ImageLogger (保持不变)
# ==========================================
class ImageLogger(Callback):
    def __init__(self, num_samples=4):
        super().__init__()
        self.num_samples = num_samples

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:
            try:
                lr = batch["lr"][:self.num_samples]
                hr = batch["hr"][:self.num_samples]
                with torch.no_grad():
                    sr = pl_module(lr)
                lr_upscaled = torch.nn.functional.interpolate(lr, size=hr.shape[2:], mode='nearest')
                grid_lr = torchvision.utils.make_grid(lr_upscaled, nrow=4, normalize=True)
                grid_sr = torchvision.utils.make_grid(sr, nrow=4, normalize=True)
                grid_hr = torchvision.utils.make_grid(hr, nrow=4, normalize=True)
                # 注意：当 logger 是列表时，需要遍历或者指定一个
                for logger in trainer.loggers:
                    if isinstance(logger, TensorBoardLogger):
                        logger.experiment.add_image('1_Input_LowRes', grid_lr, trainer.global_step)
                        logger.experiment.add_image('2_Output_SuperRes', grid_sr, trainer.global_step)
                        logger.experiment.add_image('3_GroundTruth_HighRes', grid_hr, trainer.global_step)
            except Exception:
                pass


def get_latest_checkpoint(checkpoints_root):
    """
    辅助函数：扫描所有时间戳文件夹，找到最近一次修改的 last.ckpt
    """
    if not os.path.exists(checkpoints_root):
        return None

    # 获取所有子文件夹 (即所有的时间戳文件夹)
    all_runs = [os.path.join(checkpoints_root, d) for d in os.listdir(checkpoints_root) if
                os.path.isdir(os.path.join(checkpoints_root, d))]
    if not all_runs:
        return None

    # 按修改时间排序，最近的在最后
    all_runs.sort(key=os.path.getmtime)

    # 从最新的文件夹开始往前找 last.ckpt
    for run_dir in reversed(all_runs):
        ckpt_path = os.path.join(run_dir, "last.ckpt")
        if os.path.exists(ckpt_path):
            return ckpt_path

    return None


def main():
    # ================= 配置区域 =================
    CUSTOM_DATA_PATH = r"D:\database\CIP数据集\CT影像 dcm格式"
    # CUSTOM_DATA_PATH = r"D:\project\SpineTwin-PathPlanner\data"

    BATCH_SIZE = 128
    NUM_WORKERS = 6
    MAX_EPOCHS = 20
    LR = 1e-4
    CROP_SIZE = 128
    SCALE_FACTOR = 4
    # ===========================================

    pl.seed_everything(42)

    # --- 1. 生成当前时间戳 (Run ID) ---
    # 格式: 2026-01-22_17-05
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # 定义本次运行的专属目录
    current_ckpt_dir = os.path.join(project_root, "checkpoints", run_id)
    current_log_dir = os.path.join(project_root, "logs", run_id)

    if not os.path.exists(CUSTOM_DATA_PATH):
        print(f"❌ 错误: 找不到路径: {CUSTOM_DATA_PATH}")
        return

    print(f"🚀 初始化大数据集加载器...")
    full_dataset = CTLargeDataset(CUSTOM_DATA_PATH, phase='train', crop_size=CROP_SIZE, scale_factor=SCALE_FACTOR)

    val_count = 2000
    if len(full_dataset) < val_count: val_count = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_count
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_count])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                            persistent_workers=True)

    model = SRTrainer(lr=LR)

    # --- 2. 配置回调与日志 (使用时间戳路径) ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=current_ckpt_dir,  # 模型存到 checkpoints/时间/ 下
        filename='srgan-{epoch:02d}-{val_psnr:.2f}',
        save_top_k=-1,  # 保存所有
        every_n_epochs=1,
        monitor='val_psnr',
        mode='max',
        save_last=True
    )

    # 日志存到 logs/时间/ 下
    # name="" 和 version="" 是为了不让它再自动创建 version_0 子文件夹，直接用我们指定的时间文件夹
    tb_logger = TensorBoardLogger(save_dir=current_log_dir, name="", version="")
    csv_logger = CSVLogger(save_dir=current_log_dir, name="", version="")

    img_logger = ImageLogger(num_samples=4)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        logger=[tb_logger, csv_logger],
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step'), img_logger],
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
        precision="16-mixed",
    )

    # --- 3. 智能续训逻辑 ---
    # 扫描 checkpoints 根目录，寻找最近一次运行的 last.ckpt
    checkpoints_root = os.path.join(project_root, "checkpoints")
    latest_ckpt = get_latest_checkpoint(checkpoints_root)

    print("-" * 50)
    print(f"🕒 本次运行 ID: {run_id}")
    print(f"📂 模型保存路径: {current_ckpt_dir}")
    print(f"📝 日志保存路径: {current_log_dir}")

    if latest_ckpt:
        print(f"♻️ 发现历史存档: {latest_ckpt}")
        print(f"♻️ 系统将加载该权重，并继续训练 (Logs将写入新的时间文件夹)")
    else:
        print("✨ 未发现历史存档，开始全新训练...")
    print("-" * 50)

    # 开始训练
    trainer.fit(model, train_loader, val_loader, ckpt_path=latest_ckpt)


if __name__ == "__main__":
    main()