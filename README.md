# FishMorph Local

面向**多鱼种扩展**的鱼类形态参数智能测量框架，由 `SiganusMorph Local` 稳定 V1（蓝子鱼形态参数智能测量与标注系统）迁移而来。

**内置完整实现**：蓝子鱼（Siganus）V1 species profile。
**少样本接入已验证**：海鲈、罗非鱼（各标注 10 张照片微调，体部点平均误差 4.9–6.1mm，详见 `docs/FEWSHOT_EXPERIMENT_REPORT.md`）。
鲤鱼、石斑鱼等其他鱼种可按同一流程接入，扩展规划见 `docs/MULTISPECIES_TODO.md`。

## 功能

- 图像导入与标准化单鱼侧位照片处理
- 校准板识别（ArUco / ChArUco，支持个别角标被遮挡）与透视校正、比例尺换算
- AI 预标注：YOLO-pose + Heatmap U-Net 关键点预测（16 点体系）+ 尾鳍轮廓模块（P6/P7U/P7L 几何推导）
- 形态参数自动计算（体长、全长、体高、尾柄深等）
- 质量控制（QC）与人工拖拽复核
- 单鱼测量流程与批量测量流程
- 结果导出：CSV / Excel / 标注 JSON / 预览图

## 新鱼种接入（少样本流程，约半天）

每个新鱼种只需人工标注一小批照片，无需重新设计系统：

| 步骤 | 工具 |
|---|---|
| 1. 按蓝板协议拍摄，照片放入 `photo/<物种>/` | — |
| 2. 11 点标注（8 universal core + 3 caudal-fin，预标注自动填充） | `annotate_11pt.bat` |
| 3. 构建 YOLO-pose 数据集（train/val/test 三段划分） | `scripts/build_fewshot_dataset.py` |
| 4. 从蓝子鱼底座微调（`--aug-safe` 少样本安全增广） | `scripts/train_yolopose_fewshot.py` |
| 5. 嵌回测量链路 + 调 species 尾鳍参数 | `siganusmorph/caudal_fin_profile.py` |
| 6. 冻结测试集评估 | `scripts/eval_caudal_profile.py` |

已验证鱼种与结果：

| 鱼种 | 尾型 | 标注量 | 体部点平均误差 |
|---|---|---|---|
| 蓝子鱼（Siganus） | 叉形 | V1 全量训练 | 内置 |
| 海鲈 | 微叉形 | 10 张 | 6.1mm |
| 罗非鱼 | 截形 | 6 张 | 4.9mm（零样本 8.1mm） |

数据域说明与完整实验记录见 `docs/FEWSHOT_EXPERIMENT_REPORT.md`。

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
├── models/                 # 运行时权重（YOLO-pose v0.3 + Heatmap U-Net v0.1/v0.5）+ 微调产物
├── assets/                 # UI 资源
├── data/calibration_board/ # 校准板打印材料
├── examples/               # 示例图像
├── photo/                  # 各鱼种照片批次（不入 git）
├── tools/annotate_11pt.py  # 11 点少样本标注工具（annotate_11pt.bat 启动）
├── scripts/                # 少样本数据集构建 / 训练 / 评估
├── tests/                  # smoke test + 回归测试 + zero-shot 试测
├── docs/                   # V1 文档 + 迁移/实验报告 + 多物种 TODO
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
- 少样本实验（海鲈 / 罗非鱼）：`docs/FEWSHOT_EXPERIMENT_REPORT.md`
- 多物种化规划：`docs/MULTISPECIES_TODO.md`
- V1 用户手册：`docs/user_manual_v1.0_draft.md`、`docs/USER_GUIDE.md`
- 16 关键点定义：`docs/keypoint_definitions.md`
