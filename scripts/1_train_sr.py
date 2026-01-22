import sys
import os
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

# --- 路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.datasets.ct_dataset import CTSuperResDataset
from src.core.reconstruction import SRTrainer


# ==========================================
# 改进 1: 新增图片可视化回调 (ImageLogger)
# ==========================================
class ImageLogger(Callback):
    def __init__(self, num_samples=4):
        super().__init__()
        self.num_samples = num_samples

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # 只在第一个 batch 记录图片，避免日志太大
        if batch_idx == 0:
            lr = batch["lr"][:self.num_samples]
            hr = batch["hr"][:self.num_samples]

            # 这里的 pl_module 就是你的 model，调用它生成图片
            # 必须使用 no_grad 避免梯度计算
            with torch.no_grad():
                sr = pl_module(lr)

            # 拼接图片：左边是低清，中间是生成，右边是高清
            # 为了显示清楚，把 LR 插值放大到和 HR 一样大
            lr_upscaled = torch.nn.functional.interpolate(lr, size=hr.shape[2:], mode='nearest')

            # 拼接成网格 (Grid)
            grid_lr = torchvision.utils.make_grid(lr_upscaled, nrow=4, normalize=True)
            grid_sr = torchvision.utils.make_grid(sr, nrow=4, normalize=True)
            grid_hr = torchvision.utils.make_grid(hr, nrow=4, normalize=True)

            # 记录到 TensorBoard
            trainer.logger.experiment.add_image('1_Input_LowRes', grid_lr, trainer.global_step)
            trainer.logger.experiment.add_image('2_Output_SuperRes', grid_sr, trainer.global_step)
            trainer.logger.experiment.add_image('3_GroundTruth_HighRes', grid_hr, trainer.global_step)


def main():
    # ================= 配置区域 =================
    DATA_DIR = os.path.join(project_root, "data", "raw")
    BATCH_SIZE = 8
    # 既然你有 RTX 4060，可以尝试开 2-4 个 workers 加速数据读取
    NUM_WORKERS = 4
    MAX_EPOCHS = 200
    LR = 1e-4
    SCALE_FACTOR = 4
    CROP_SIZE = 128
    # ===========================================

    pl.seed_everything(42)

    # 1. 准备数据
    print(f"正在加载数据: {DATA_DIR} ...")
    full_dataset = CTSuperResDataset(
        data_dir=DATA_DIR,
        phase='train',
        crop_size=CROP_SIZE,
        scale_factor=SCALE_FACTOR
    )

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True  # GPU训练建议开启
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # 2. 初始化模型
    model = SRTrainer(lr=LR)

    # ==========================================
    # 改进 2: 修改 Checkpoint 监控指标
    # ==========================================
    # 监控 'val_loss' (MSE误差)，越小代表越清晰
    # 不要监控 'g_loss'，那个是骗判别器的能力，不代表清晰度
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(project_root, "checkpoints", "sr_model"),
        filename='srgan-{epoch:02d}-{val_loss:.5f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # 实例化图片记录器
    img_logger = ImageLogger(num_samples=4)

    logger = TensorBoardLogger(os.path.join(project_root, "logs"), name="sr_experiment")

    # 开启 TensorCore 加速
    torch.set_float32_matmul_precision('medium')

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        logger=logger,
        # 加入 img_logger
        callbacks=[checkpoint_callback, lr_monitor, img_logger],
        log_every_n_steps=5,
        check_val_every_n_epoch=1,  # 每1轮都要跑验证集，为了生成图片看
    )

    print("🚀 开始训练 SRGAN 模型 (增强版)...")
    print("请务必使用 'tensorboard --logdir logs' 查看 'IMAGES' 标签页，肉眼观察效果！")

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()