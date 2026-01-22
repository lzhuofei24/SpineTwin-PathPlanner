import sys
import os
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# --- 关键：将项目根目录添加到搜索路径，确保能 import src ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.datasets.ct_dataset import CTSuperResDataset
from src.core.reconstruction import SRTrainer


def main():
    # ================= 配置区域 =================
    DATA_DIR = os.path.join(project_root, "data", "raw")  # 数据路径
    BATCH_SIZE = 8  # 如果显存不够(如<4G)，改为 4 或 2
    NUM_WORKERS = 0  # Windows下建议先设为0，调试没问题后再改为 2 或 4
    MAX_EPOCHS = 100  # 训练轮数
    LR = 1e-4  # 学习率
    SCALE_FACTOR = 4  # 超分倍数
    CROP_SIZE = 128  # 训练切片大小 (HR)
    # ===========================================

    # 1. 设置随机种子，保证可复现
    pl.seed_everything(42)

    # 2. 准备数据
    print(f"正在加载数据: {DATA_DIR} ...")
    full_dataset = CTSuperResDataset(
        data_dir=DATA_DIR,
        phase='train',
        crop_size=CROP_SIZE,
        scale_factor=SCALE_FACTOR
    )

    # 划分训练集和验证集 (9:1)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"数据集划分完成 -> 训练集: {len(train_dataset)} 张, 验证集: {len(val_dataset)} 张")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # 3. 初始化模型
    model = SRTrainer(lr=LR)

    # 4. 配置回调函数 (保存模型和监控学习率)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(project_root, "checkpoints", "sr_model"),
        filename='srgan-{epoch:02d}-{g_loss:.4f}',
        save_top_k=3,  # 保存最好的3个模型
        monitor='g_loss',  # 根据生成器损失判断好坏
        mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # 5. 配置训练器
    logger = TensorBoardLogger(os.path.join(project_root, "logs"), name="sr_experiment")

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=5,  # 每5步记录一次日志
        # limit_train_batches=0.1 # 调试时取消注释，只跑10%的数据快速测试流程
    )

    # 6. 开始训练
    print("🚀 开始训练 SRGAN 模型...")
    print(f"日志将保存在: {logger.log_dir}")
    print("可以使用 tensorboard --logdir logs 查看训练曲线")

    trainer.fit(model, train_loader, val_loader)


# 由于 windows 下多进程的限制，必须加这个保护
if __name__ == "__main__":
    from pytorch_lightning import Trainer  # 延迟导入避免某些循环引用

    main()