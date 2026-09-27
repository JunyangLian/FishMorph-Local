# FishMorph 多物种化 TODO（只记录，不在迁移阶段实施）

> 原则：先保证 Siganus V1 在 FishMorph Local 中完整稳定运行（已完成），再按本文档逐步解耦。
> 当前已实现并验证的 species profile：**Siganus（蓝子鱼）**。以下任何条目实施前，V1 行为不得回退。

## 1. 需要抽象的核心设施

### Species Registry（物种注册表）
- [ ] 以 `config/species/*.yaml` 为单一事实来源，建立加载器与校验（当前仅有 `siganus.yaml`，运行时未读取）。
- [ ] 定义 profile 版本策略（`profile_version`）与向后兼容规则。

### Keypoint Schema（关键点体系）
- [ ] 把 `siganusmorph/config.py: KEYPOINT_DEFS`（16 点 + 中文引导文案）外置为 schema 文件（如 `config/species/siganus.keypoints.yaml`）。
- [ ] 抽象点数可变：模型输出通道数、可视化、测量定义均假设 16 点，需要参数化。
- [ ] 保留派生点（P7V 全长终点等）的 species-specific 计算钩子。

### Model Registry（模型注册表）
- [ ] 模型路径目前硬编码在 `preannotation.py` / `heatmap_preannotation.py` / `v06_preannotation.py`，改为由 profile 声明（`model_files` 字段已预留）。
- [ ] 引入模型条目元数据（架构、输入尺寸、类别数、keypoint schema 版本、训练数据范围）。
- [ ] 支持 per-species 模型目录布局（`models/<species_id>/...`）。

### Measurement Profile（测量定义）
- [ ] `measurements.py: MEASUREMENT_DEFS`、`dual_axis_measurement.py`、`local_normal_measurement.py` 中与"蓝子鱼体型"相关的规则外置。
- [ ] 区分通用形态量（标准长、全长、体高、尾柄深——多数鱼种通用）与 species-specific 修正（压缩尾 TL 修正等，见 `compressed_tail_tl.py`）。

### QC Profile
- [ ] `geometric_rules.py`、`preannotation.py` 的阈值与规则参数化（比例阈值、姿态约束）。
- [ ] 不同鱼种允许不同 QC 严格度（如尾鳍展开判定）。

### Calibration（校准）
- [ ] 校准框架（ArUco/ChArUco 检测、透视校正、mm/px 换算）本身是通用的，可保留；但校准板打印物料、板型配置（`A3_V2_CHARUCO_PLUMB_CONFIG`）应挂到 profile。
- [ ] 蓝板分割 `segmentation.py: segment_fish_from_blue_board` 依赖蓝色拍摄板，需按物种/拍摄协议抽象背景分割策略。

### Species-specific morphology（物种形态学约束）
- [ ] 体形先验（尾柄比、体高/体长范围）目前隐含在几何规则中，需显式化为 per-species 参数。

### Cross-species evaluation / Zero-shot / Few-shot
- [ ] 建立跨物种评测集与指标基线后再谈迁移；现阶段不训练、不标注。
- [ ] Zero-shot/few-shot 适配仅作远期方向记录，不设时间表。

### Dataset registry / Model versioning
- [ ] 数据集登记（物种、拍摄协议、标注 schema 版本、划分）。
- [ ] 模型版本与 dataset/ schema 版本绑定，导出可追溯的 model card。

## 2. 通用模块（原则上鱼种无关，应保持稳定）

- 图像上传与文件管理（pages 导入流程、`image_utils.py`）
- calibration framework（`aruco_utils.py`、`calibration.py`、`charuco_utils.py`、`plumbline.py`）
- 导出框架（CSV/Excel/JSON 结构、`build_result_row` 的行结构生成机制）
- UI framework（`ui_styles.py`、`formal_ui.py` 的流程骨架、`components.py` + `web_components/` 拖拽修正组件）
- 人工修正框架（编辑器数据流：点→测量即时重算）
- 批量任务框架（`remote_jobs.py` 队列与产物管理）

## 3. 物种相关模块（未来按 profile 切换）

- landmark 定义与引导文案（`config.py`）
- 三个模型权重（YOLO-pose、heatmap v0.1/v0.5）
- morphology measurement（`measurements.py`、`dual_axis_measurement.py`、`local_normal_measurement.py`、`tail_geometry.py`、`caudal_base_geometry.py`、`compressed_tail_tl.py`、`eye_geometry.py`、`operculum_geometry.py`）
- QC 阈值（`geometric_rules.py`）
- body-shape constraints（中轴/尾轴几何约束）
- 背景分割策略（`segmentation.py`）

## 4. 建议的落地顺序

1. **package rename**（二阶段）：`siganusmorph/` → `fishmorph/`，一次性完成 import 全量替换（本次迁移刻意保留原名，避免与解耦混做）。
2. **配置层接入**：让运行时真正读取 `config/species/<id>.yaml`（先只服务 siganus，行为零变化）。
3. **Keypoint schema 外置** → **Model registry** → **Measurement/QC profile 参数化**。
4. 每步以 `tests/smoke_test.py` + 人工复核截图作为回归门槛。

## 5. 明确暂不做

- 鲤鱼、石斑鱼等其他鱼种的接入（海鲈、罗非鱼已完成少样本接入流程验证，见 `FEWSHOT_EXPERIMENT_REPORT.md`）。
- 海鲈 / 罗非鱼的工业化 species profile（当前为实验级参数与微调模型；尾鳍模块截形尾分支、README 口径外的精度承诺待做）。
- cross-species 统一模型。
- V2 功能合并、UI 大规模重构。
