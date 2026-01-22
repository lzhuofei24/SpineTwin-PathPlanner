import sys
import os
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

# --- 关键修改 1: 获取项目根目录的绝对路径 ---
# __file__ 是当前脚本的路径 (scripts/train_sr_demo.py)
# dirname -> scripts
# dirname(dirname) -> SpineTwin-PathPlanner (根目录)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.datasets.ct_dataset import CTSuperResDataset
from src.core.reconstruction import SRTrainer


def main():
    # --- 关键修改 2: 使用 os.path.join 拼接绝对路径 ---
    # 这样无论你在哪里运行脚本，都能精准找到 data/raw
    data_path = os.path.join(project_root, "data", "raw")

    # 打印一下路径确认
    print(f"正在尝试加载数据路径: {data_path}")

    # 1. 准备数据
    try:
        dataset = CTSuperResDataset(data_path, phase='train', crop_size=64, scale_factor=4)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)  # Windows下调试建议 num_workers=0
    except ValueError as e:
        print(f"错误: {e}")
        print("请检查 D:\\project\\SpineTwin-PathPlanner\\data\\raw 文件夹下是否有 .dcm 文件")
        return

    # 2. 模型配置
    model = SRTrainer()

    # 3. 训练器配置 (使用GPU)
    # 显式指定 accelerator='gpu'，利用你的 RTX 4060
    trainer = Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        default_root_dir=os.path.join(project_root, "logs", "sr_demo")
    )

    # 4. 开始训练
    print("🚀 开始训练超分辨率模型 (Demo)...")
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()