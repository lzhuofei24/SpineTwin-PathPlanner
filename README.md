# SpineTwin-PathPlanner

基于多模态和数字孪生的脊柱穿刺路径规划研究

## 模块对应关系
- `src/models/srgan.py`: CT超分辨率 (报告 3.2)
- `src/models/cyclegan.py`: 骨密度建模 (报告 3.3)
- `src/models/pose_net.py`: 跨模态配准 (报告 3.4)
- `src/models/fusion_planner.py`: 路径规划 (报告 3.5)