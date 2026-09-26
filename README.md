# FishMorph Local

面向**多鱼种扩展**的鱼类形态参数智能测量框架。当前已实现并验证的 species profile 为**蓝子鱼（Siganus）**：
蓝子鱼形态参数智能测量与标注系统 V1.0（由 `SiganusMorph Local` 稳定 V1 完整迁移而来）。

> 当前状态：仅 Siganus V1 已实现并验证。其他鱼种（罗非鱼、鲤鱼、石斑鱼等）尚未实现，
> 相关扩展规划见 `docs/MULTISPECIES_TODO.md`。

## 功能（Siganus V1）

- 图像导入与标准化单鱼侧位照片处理
- 校准板识别（ArUco / ChArUco）与透视校正、比例尺换算
- AI 预标注：YOLO-pose + Heatmap U-Net 关键点预测（16 关键点体系）
- 形态参数自动计算（体长、全长、体高、尾柄深等）
- 质量控制（QC）与人工拖拽复核
- 单鱼测量流程与批量测量流程
- 结果导出：CSV / Excel / 标注 JSON / 预览图

## 快速开始

```bat
run_app.bat
```

等价命令：

```bat
python -m streamlit run app.py --server.headless true --server.port 8501
```

浏览器打开 <http://localhost:8501>。

### 环境要求

- Python 3.10+（迁移验证环境：3.10.1）
- 安装依赖：`pip install -r requirements.txt`
- 模型权重已随项目迁移至 `models/`（约 92 MB，不进入 git）

### 校准板

测量前需打印校准板并随鱼拍摄，打印文件见 `data/calibration_board/`，
打印说明见 `docs/board_printing_instructions.md`。

### 示例图像

`examples/siganus_example_01.png` 为一张蓝子鱼标准侧位样例，可用于上手流程。

## 项目结构

```text
FishMorph Local/
├── app.py                  # Streamlit 入口（首页导航）
├── pages/                  # V1 正式页面：单鱼测量 / 批量测量 / 结果导出 / 联系我们
├── siganusmorph/           # V1 核心包（迁移自原项目，暂保留原名，待二阶段重构）
├── config/species/         # species profile 层（当前：siganus.yaml）
├── models/                 # 运行时权重（YOLO-pose v0.3 + Heatmap U-Net v0.1/v0.5）
├── assets/                 # UI 资源
├── data/calibration_board/ # 校准板打印材料
├── examples/               # 示例图像
├── tests/                  # smoke test
├── docs/                   # V1 文档 + 迁移报告 + 多物种 TODO
├── requirements.txt
└── run_app.bat
```

## Smoke Test

```bat
python tests/smoke_test.py
```

覆盖：代码可编译、核心模块可导入、三个模型权重可加载、YOLO 16 关键点推理、
heatmap 模型前向、形态参数计算与 CSV/Excel/JSON 导出链路。

## 文档

- 迁移背景与验证结果：`docs/V1_MIGRATION_REPORT.md`
- 多物种化规划：`docs/MULTISPECIES_TODO.md`
- V1 用户手册：`docs/user_manual_v1.0_draft.md`、`docs/USER_GUIDE.md`
- 16 关键点定义：`docs/keypoint_definitions.md`
