# SpineTwin-PathPlanner

基于多模态与数字孪生的**脊柱穿刺路径规划**研究项目。通过 CT 超分辨率、骨密度建模、跨模态配准与融合规划等模块，构建从术前 CT 到术中 X 光/DRR 的端到端路径规划流程。

---

## 项目概述

本仓库实现“多模态 + 数字孪生”思路下的脊柱穿刺路径规划管线，主要包含四类能力：

| 阶段 | 功能 | 说明 |
|------|------|------|
| **CT 超分辨率** | 低分辨率 CT → 高分辨率 CT | 基于 SRGAN，提升 CT 切片空间分辨率，便于后续配准与规划 |
| **骨密度建模** | CT → 骨密度/力学先验 | 使用 CycleGAN 类模型做骨密度相关建模（接口已预留） |
| **跨模态配准** | CT/DRR ↔ X 光 | 双流配准网络（TwoStreamRegNet），对齐术前 CT 与术中 X 光 |
| **路径规划** | 2D/3D 穿刺路径生成 | 融合规划网络（PathPlanner），在配准后的空间中进行路径规划 |

当前**已完整实现并可用于训练**的是 **CT 超分辨率（SRGAN）** 流程；其余模块已在代码与配置中预留接口，待进一步实现。

---

## 项目结构

```
SpineTwin-PathPlanner/
├── configs/                 # 各阶段 YAML 配置
│   ├── sr_config.yaml       # CT 超分：scale_factor、batch_size、lr、epochs
│   ├── density_config.yaml  # 骨密度：CycleGAN 的 lambda_cycle / lambda_identity
│   ├── reg_config.yaml      # 配准：图像尺寸、TwoStreamNet 相关
│   └── plan_config.yaml     # 路径规划：input_channels、hidden_dim
├── scripts/                 # 训练与演示入口
│   ├── 1_train_sr.py        # 超分训练（内存预加载，适合中小规模数据）
│   ├── 1_train_sr_large.py  # 超分训练（按需读盘，大数据集 + 续训 + 时间戳日志）
│   ├── train_sr_demo.py     # 超分小 demo，快速验证流程
│   ├── 2_train_density.py   # 骨密度训练（占位）
│   ├── 3_train_reg.py       # 配准训练（占位）
│   └── 4_train_plan.py      # 路径规划训练（占位）
├── src/
│   ├── core/                # 训练与推理逻辑
│   │   ├── reconstruction.py   # SRTrainer：SRGAN 的 PyTorch Lightning 训练
│   │   ├── density_estimation.py  # 骨密度训练逻辑（占位）
│   │   ├── registration.py      # 配准训练逻辑（占位）
│   │   └── planning.py          # 路径规划推理逻辑（占位）
│   ├── models/              # 网络结构
│   │   ├── srgan.py         # SRGenerator + SRDiscriminator（CT 单通道）
│   │   ├── cyclegan.py       # DensityCycleGAN（占位）
│   │   ├── pose_net.py       # TwoStreamRegNet（占位）
│   │   └── fusion_planner.py # PathPlanner（占位）
│   ├── datasets/
│   │   ├── ct_dataset.py     # CTSuperResDataset：DICOM → lr/hr 对
│   │   ├── drr_generator.py  # DRR 模拟 X 光（占位）
│   │   └── trajectory_data.py
│   └── utils/
│       ├── geometry.py      # 3D 几何变换（占位）
│       ├── metrics.py       # PSNR/SSIM/TRE/Dice 等（占位）
│       └── visualization.py # 可视化（占位）
├── test_dicom.py            # 测试 DICOM 加载与 lr/hr 可视化
└── README.md
```

训练产生的**日志**与**检查点**默认落在项目下的 `logs/`、`checkpoints/`，已在 `.gitignore` 中忽略；医疗影像数据目录（如 `data/raw/`）亦被忽略。

---

## 模块与脚本对应关系

| 能力 | 模型定义 | 训练/推理逻辑 | 配置 | 训练脚本 |
|------|----------|----------------|------|----------|
| CT 超分辨率 | `src/models/srgan.py` | `src/core/reconstruction.py`（SRTrainer） | `configs/sr_config.yaml` | `1_train_sr.py` / `1_train_sr_large.py` |
| 骨密度建模 | `src/models/cyclegan.py` | `src/core/density_estimation.py` | `configs/density_config.yaml` | `2_train_density.py` |
| 跨模态配准 | `src/models/pose_net.py` | `src/core/registration.py` | `configs/reg_config.yaml` | `3_train_reg.py` |
| 路径规划 | `src/models/fusion_planner.py` | `src/core/planning.py` | `configs/plan_config.yaml` | `4_train_plan.py` |

---

## 环境与依赖

建议使用 Python 3.8+，并安装：

- `torch`、`torchvision`
- `pytorch-lightning`
- `pydicom`（若使用压缩 DICOM，可配合 `pylibjpeg`、`gdcm`）
- `numpy`、`matplotlib`
- `PyYAML`（若从 YAML 读配置）

无 `requirements.txt` 时，可根据上述列表自行创建或使用 conda/pip 安装。

---

## 数据准备

### CT 超分辨率（当前已用）

- 将 **DICOM 格式的 CT 切片**（`.dcm`）放在某一目录下，支持多级子目录。
- **默认路径**：项目根目录下的 `data/raw/`。
- **大数据脚本**：在 `1_train_sr_large.py` 中可修改 `CUSTOM_DATA_PATH`，指向你的 DICOM 根目录。

数据会按 HU 窗位做线性归一化（如 `[-1000, 2000] HU → [0, 1]`），并在训练时由程序生成 4× 下采样的低分辨率图像作为输入。

---

## 使用方法

### 1. 检查 DICOM 与数据加载

确保 `data/raw`（或你指定的目录）下有 `.dcm` 文件，再运行：

```bash
python test_dicom.py
```

可查看单样本的 `lr` / `hr` shape 及来源路径，并弹出可视化对比图。

### 2. CT 超分辨率训练

- **小数据 / 快速验证**（数据先载入内存）  
  使用 `scripts/1_train_sr.py`，数据路径在脚本内写死为 `project_root/data/raw`，并会划分 90% train / 10% val。
- **大数据 / 长期训练**（按需从盘读取 + 续训）  
  使用 `scripts/1_train_sr_large.py`：
  - 在脚本内设置 `CUSTOM_DATA_PATH` 为你的 DICOM 根目录；
  - 每次运行会生成时间戳型 `run_id`，对应 `checkpoints/<run_id>/`、`logs/<run_id>/`；
  - 若在 `checkpoints/` 下存在某次运行的 `last.ckpt`，会自动从最近一次 `last.ckpt` 续训。

在项目根目录下执行示例：

```bash
# 小数据、默认 data/raw
python scripts/1_train_sr.py

# 大数据、自定义路径与续训
python scripts/1_train_sr_large.py
```

- **最小 demo**：  
  `python scripts/train_sr_demo.py` 使用 `CTSuperResDataset` 与默认 `data/raw`，适合快速跑通流程。

### 3. 超分配置说明（`configs/sr_config.yaml`）

| 项 | 含义 |
|----|------|
| `model.scale_factor` | 超分倍数（如 4 表示 4×） |
| `train.batch_size` / `train.lr` / `train.epochs` | 批大小、学习率、最大 epoch |

脚本内部分参数（如 `CROP_SIZE`、`NUM_WORKERS`）会覆盖或补充 YAML，完整行为以脚本为准。

### 4. 其他模块（骨密度、配准、规划）

对应模型与 core 逻辑多为占位，配置已在 `configs/` 中写好。实现后可直接在 `2_train_density.py`、`3_train_reg.py`、`4_train_plan.py` 中调用相应 core 与 dataset。

---

## 超分模型与训练简述

- **生成器**（`SRGenerator`）：基于 SRResNet 的残差结构 + PixelShuffle 上采样，针对 CT 单通道，输出与输入已做归一化到 [0,1]。
- **判别器**（`SRDiscriminator`）：卷积二分类真/假高分辨率 CT 切片。
- **训练**：在 `SRTrainer` 中交替更新 G/D，损失包含像素 MSE 与对抗损失；验证阶段计算并记录 `val_psnr`，checkpoint 可按 `val_psnr` 保存 top-k。

---

## 输出与日志

- **检查点**  
  - `1_train_sr.py`：`checkpoints/sr_model/`，按 `val_psnr` 存 top-3。  
  - `1_train_sr_large.py`：`checkpoints/<run_id>/`，含 `last.ckpt` 及按 epoch 保存的权重。
- **日志**  
  - TensorBoard + CSV 写入 `logs/`（或 `logs/<run_id>/`），便于画曲线与离线分析。

---

## 注意事项

1. **数据与隐私**：请勿将真实患者 DICOM 提交到版本库；`data/`、`*.dcm`、`logs/`、`checkpoints/` 等已列入 `.gitignore`。
2. **路径**：所有脚本均通过 `sys.path.append(project_root)` 等方式假定在项目根目录或其子目录运行，若从其他路径运行需自行保证可导入 `src`。
3. **占位模块**：`cyclegan`、`pose_net`、`fusion_planner` 以及对应 core/datasets/utils 中的占位，仅表示架构预留，实际算法需在后续开发中补齐。

---

## 引用与参考

- CT 超分部分对应文献中的 SRGAN 设计（如报告 [42]），生成器采用 SRResNet 风格结构，针对医学 CT 单通道做了适配。

---

## 许可证与免责声明

本项目仅供研究使用。任何临床应用或商业使用前，需自行完成合规审批与验证；作者不对使用本项目带来的直接或间接后果负责。
