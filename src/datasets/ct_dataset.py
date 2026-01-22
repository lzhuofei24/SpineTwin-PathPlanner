import os
import glob
import numpy as np
import torch
import pydicom
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision import transforms  # <--- 关键修正：导入 transforms


class CTSuperResDataset(Dataset):
    def __init__(self, data_dir, phase='train', crop_size=64, scale_factor=4):
        """
        Args:
            data_dir: 包含 .dcm 文件的根目录 (支持子文件夹)
            phase: 'train' 或 'val'
            crop_size: 训练时的裁剪块大小 (HR图像大小)
            scale_factor: 缩放因子 (如4倍超分)
        """
        super().__init__()
        self.phase = phase
        self.crop_size = crop_size
        self.scale_factor = scale_factor

        # 1. 递归查找所有 .dcm 文件
        self.dcm_files = sorted(glob.glob(os.path.join(data_dir, "**/*.dcm"), recursive=True))

        if len(self.dcm_files) == 0:
            raise ValueError(f"在 {data_dir} 下未找到任何 .dcm 文件，请检查路径。")

        print(f"[{phase}] 正在索引 DICOM 文件... 共找到 {len(self.dcm_files)} 张切片")

    def __len__(self):
        return len(self.dcm_files)

    def __getitem__(self, index):
        dcm_path = self.dcm_files[index]

        try:
            # 1. 读取 DICOM
            ds = pydicom.dcmread(dcm_path)

            # 处理压缩格式 (如果不加 try-except，缺少依赖库时会报错)
            try:
                ds.decompress()
            except Exception as e:
                # 如果解压失败，通常是因为缺少 pylibjpeg 等库
                raise RuntimeError(f"解压DICOM失败，请检查是否安装了 pylibjpeg 和 gdcm: {e}")

            # 2. 获取像素数据并转换为 HU
            # 使用 pydicom 原生方法处理 RescaleSlope/Intercept
            if not hasattr(ds, 'pixel_array'):
                raise ValueError("该DICOM文件不包含像素数据")

            img_hu = ds.pixel_array.astype(np.float32) * getattr(ds, 'RescaleSlope', 1.0) + getattr(ds,
                                                                                                    'RescaleIntercept',
                                                                                                    0.0)

            # 3. 归一化 (Normalization) [-1000, 2000] -> [0, 1]
            min_hu, max_hu = -1000.0, 2000.0
            img_norm = np.clip(img_hu, min_hu, max_hu)
            img_norm = (img_norm - min_hu) / (max_hu - min_hu)

            # 转为 Tensor [1, H, W]
            hr_tensor = torch.from_numpy(img_norm).unsqueeze(0).float()

            # 4. 尺寸检查 (如果图片本身比裁剪框还小，就跳过这张图，取下一张)
            if hr_tensor.shape[1] < self.crop_size or hr_tensor.shape[2] < self.crop_size:
                return self.__getitem__((index + 1) % len(self.dcm_files))

            # 5. 训练时数据增强
            if self.phase == 'train':
                # --- 修正点: 使用 transforms.RandomCrop ---
                i, j, h, w = transforms.RandomCrop.get_params(
                    hr_tensor, output_size=(self.crop_size, self.crop_size)
                )
                hr_tensor = F.crop(hr_tensor, i, j, h, w)

                if torch.rand(1) < 0.5:
                    hr_tensor = F.hflip(hr_tensor)
                if torch.rand(1) < 0.5:
                    hr_tensor = F.vflip(hr_tensor)

            # 6. 物理退化仿真 (生成 LR)
            lr_h = hr_tensor.shape[1] // self.scale_factor
            lr_w = hr_tensor.shape[2] // self.scale_factor

            lr_tensor = torch.nn.functional.interpolate(
                hr_tensor.unsqueeze(0),
                size=(lr_h, lr_w),
                mode='bicubic',
                align_corners=False
            ).squeeze(0)

            return {"lr": lr_tensor, "hr": hr_tensor, "path": dcm_path}

        except Exception as e:
            # 打印具体错误路径，方便定位坏文件
            print(f"Error reading {dcm_path}: {e}")
            # 遇到坏文件，尝试读取下一张，避免训练中断
            # 注意：如果所有文件都坏了，这里会递归直到报错 (Python有最大递归深度限制)
            if index < len(self.dcm_files) - 1:
                return self.__getitem__(index + 1)
            else:
                raise e