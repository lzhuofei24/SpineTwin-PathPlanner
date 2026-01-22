import matplotlib.pyplot as plt
from src.datasets.ct_dataset import CTSuperResDataset

# 指向你的 DICOM 文件夹路径
data_path = "data/raw"  # 确保这里面有 .dcm 文件

try:
    ds = CTSuperResDataset(data_path, phase='train', crop_size=128)
    sample = ds[0]

    print("LR Shape:", sample['lr'].shape)
    print("HR Shape:", sample['hr'].shape)
    print("来源文件:", sample['path'])

    # 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(sample['lr'].squeeze(), cmap='gray')
    axes[0].set_title("Input (Low Resolution)")
    axes[1].imshow(sample['hr'].squeeze(), cmap='gray')
    axes[1].set_title("Ground Truth (High Resolution)")
    plt.show()

except Exception as e:
    print(f"测试失败: {e}")
    print("请检查路径下是否包含 .dcm 文件")