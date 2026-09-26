# FishMorph Local —— Siganus V1 迁移报告

- **迁移日期**：2026-09-26
- **原路径**：`E:\1_yanjiusheng\SiganusMorph Local`（只读，未做任何修改）
- **新路径**：`D:\1_yanjiusheng\FishMorph Local`
- **迁移原则**：只复制、不破坏原项目；只迁移稳定 V1 + 最小多物种架构预留；不迁移 V2。

---

## 1. V1 判定依据

1. **软著 V1 清单**（`release_v1.0_software_copyright/source_code_file_list.md`）确认了 V1 的核心范围：
   `app.py`、4 个正式页面、`siganusmorph/` 核心测量模块、`web_components/` 交互组件。
2. **当前主代码对照**：当前 `app.py` 使用 `st.navigation` 只挂载 4 个正式中文页面
   （`1_单鱼测量` / `2_批量测量` / `3_结果导出` / `4_联系我们`），比软著 V1（含 `9_高级工具` 入口）更新，
   并将页面回退逻辑（`formal_page_path`）内建。按"优先迁移当前稳定实现"原则，以当前 `app.py` 挂载的 4 页为准。
3. **V1 与 V2 的边界**：V2 代码全部位于独立目录 `siganusmorph_v2/` 与 `release_v2.0_software_copyright/`，
   V1 代码（`app.py` / `pages` / `siganusmorph`）中**没有任何**对 V2 的 import 或路径引用（已全量 grep 验证）。
   运行时模型链（`formal_ui.py:556-558`）确认 V1 使用 YOLO-pose v0.3 + Heatmap U-Net v0.1/v0.5。

**V1 release 与当前主代码差异**：当前主代码比 2026-06 的软著 V1 多出
`4_联系我们.py`（新增页面）、`remote_jobs.py`（本地批量任务队列）、`ui_styles.py` 品牌区更新（2026-08），
均为 V1 功能范围内的稳定性/完备性更新，已随当前版本迁移。

---

## 2. 实际迁移内容

### 2.1 代码

| 内容 | 说明 |
|---|---|
| `app.py` | Streamlit 入口（st.navigation 挂载 4 个正式页面） |
| `pages/1_单鱼测量.py`、`pages/2_批量测量.py`、`pages/3_结果导出.py`、`pages/4_联系我们.py` | V1 全部正式页面 |
| `siganusmorph/`（整包，41 个文件） | V1 核心包：校准（aruco/charuco）、透视校正、分割、YOLO 预标注、Heatmap U-Net、16 关键点体系（`config.KEYPOINT_DEFS`）、形态测量（measurements/dual_axis/local_normal/tail_geometry 等）、QC、拖拽修正组件（`components.py` + `web_components/`）、CSV/Excel/JSON 导出、批量任务（`remote_jobs.py`）、UI 样式（`ui_styles.py`） |
| `run_app.bat` | 启动脚本（命令与原项目一致） |

代码复制后经 `diff -rq` 校验，与原项目字节级一致（品牌层修改除外，见 §5）。

### 2.2 模型权重（共 3 个运行时权重 + 元数据，约 92 MB）

| 权重 | 用途 | 加载位置 |
|---|---|---|
| `models/siganusmorph_yolopose_v0.3_corrected_real_5_15/preannotation_candidate.pt`（6.0 MB） | 鱼体 16 关键点自动预标注（主模型） | `siganusmorph/preannotation.py: default_preannotation_model` → `formal_ui.py:556` |
| `models/siganusmorph_heatmap_unet_v0.1/preannotation_candidate.pt`（42.7 MB） | Heatmap 关键点回归（v0.1，`heatmap_preannotation` 链） | `siganusmorph/heatmap_preannotation.py: DEFAULT_MODEL_PATH` → `formal_ui.py:557` |
| `models/siganusmorph_heatmap_unet_v0.5/preannotation_candidate.pt`（42.7 MB） | Heatmap 关键点回归（v0.5，v06 keypointwise 链） | `siganusmorph/v06_preannotation.py: V05_MODEL_PATH` → `formal_ui.py:558` |

随权重一并复制的元数据：各模型 `keypoint_schema.json`、`README_model.md`、`train_config.yaml`/`training_config.yaml`、
v0.3 的 `data.yaml`、heatmap 的 `model_selection_report.csv`。

权重 md5 与原项目逐一核对一致。

### 2.3 资源与文档

| 内容 | 依据 |
|---|---|
| `assets/formal_ui/iridescent_hero.webp` | `ui_styles.py` 唯一引用的 UI 图片资源 |
| `data/calibration_board/bluefish_calibration_board_A3_v3b_print.pdf`、`data/calibration_board/v2_charuco_plumb/`（打印 PDF/SVG/预览/元数据） | 用户打印校准板所需材料 |
| `examples/siganus_example_01.png` | 示例图（原 `data/real_images_raw_png/real_013.png`，最小的一张），供演示与 smoke test |
| `docs/`（14 个 V1 文档） | USER_GUIDE、user_manual_v1.0_draft、keypoint_definitions、MEASUREMENT_AXIS、METHOD_OVERVIEW、DATA_STRUCTURE、imaging_protocol、board_printing_instructions、calibration_board_v2_charuco_plumbline、TROUBLESHOOTING、VERSION_HISTORY、formal_ui_style_guide、formal_ui_calibration_coordinate_consistency、formal_ui_interactive_editor_notes |

### 2.4 新建文件（FishMorph 层）

- `config/species/siganus.yaml` —— 最小 species profile 声明（V1 运行时尚不读取，作多物种接入预留）
- `requirements.txt` —— 见 §6
- `README.md` —— FishMorph 项目说明（明确"当前已验证鱼种仅为 Siganus"）
- `.gitignore` —— 基于原项目改写
- `tests/smoke_test.py` —— V1 smoke test
- `docs/V1_MIGRATION_REPORT.md`（本文件）、`docs/MULTISPECIES_TODO.md`

---

## 3. 未迁移内容

| 内容 | 排除原因 |
|---|---|
| `siganusmorph_v2/` | **V2，明确排除** |
| `release_v2.0_software_copyright/` | **V2，明确排除** |
| `archive/`、`README_legacy_v0.1.md` | 历史归档 |
| `developer_tools/`（含 `legacy_app.py`） | 无 V1 运行时 import（V1 入口链不引用） |
| `logs/`、`results/`、`runs/` | 历史运行日志/结果（新项目运行时自动重建 `results/`） |
| `datasets/`（7 个训练数据集目录） | 训练数据，非运行依赖 |
| `data/real_images_raw_png/`（25 张原图，每张约 15 MB）及 manifest | 历史采样数据；仅取 1 张作为示例图 |
| `_deployment_build/`、`_ui_style_streamlit*.log` | 构建产物/临时日志 |
| `deployment/`（remote_compute、v1_server） | 服务器部署配置，本地运行不需要 |
| `scripts/`（40 个脚本） | 均为训练/评估/数据集导出工具，V1 运行时不依赖 |
| 根目录 `yolo11n.pt`、`yolo11n-pose.pt` | 仅在训练评估脚本中作为预训练底座被提及，运行时不加载，且已被 `models/` 中的训练权重替代 |
| `models/` 其余权重 | yolopose v0.1/v0.2（v0.3 存在时永不加载的 fallback）、heatmap `best_model.pt`/`last_model.pt`（与 candidate 内容重复）、全部训练曲线图/训练批次图 |
| `pages/1_specimen_assignment.py`、`pages/2_image_curation.py`、`pages/3_realworld_review.py`、`pages/9_高级工具.py` | 未被当前 `app.py` 导航挂载的历史/研究页面（`3_realworld_review` 所依赖的 `siganusmorph.realworld_review` 模块已随整包迁移，`formal_ui` 依赖它） |
| `assets/templates/` | 全项目代码/文档均无引用 |

**V1/V2 混杂情况**：未发现。V1 代码零引用 V2；唯一的耦合点是原 `.gitignore` 尾部有一行 `siganusmorph_v2/`（新 .gitignore 已不再需要）。

---

## 4. 硬编码路径检查与修复

对 V1 运行时代码（`app.py`、`pages/`、`siganusmorph/`）做了盘符路径与项目名全量搜索：

| 位置 | 内容 | 处理 |
|---|---|---|
| `siganusmorph/real_dataset.py:33` | `REAL_SOURCE_DIR = Path(r"C:\Users\羊叽叽\Desktop\caiyang\5.15")` | **已修改新副本**：改为 `os.getenv("FISHMORPH_REAL_SOURCE_DIR", "data/real_source_images")`。该常量属于数据集工具链，正式 V1 UI 不使用；修改仅影响新项目 |
| `siganusmorph/config.py:1` | docstring 品牌文案 "SiganusMorph Local" | 改为 "FishMorph Local (Siganus V1 profile)"（纯文案，非路径） |
| 其余 | 无任何 `E:\`、绝对盘符或指向原项目的路径 | 模型/资源均通过 `project_root` 相对解析（`Path(__file__).resolve().parents[1]`） |

结论：**新项目不依赖 `E:\1_yanjiusheng\SiganusMorph Local`**，已在原项目保持只读的前提下完成独立启动验证。

---

## 5. 品牌层修改（仅新项目）

- `app.py`：浏览器页标题 → "FishMorph 蓝子鱼形态参数智能测量与标注系统 V1.0"；首页 eyebrow → "FishMorph Local"；首页描述加入多物种框架声明（明确当前已验证鱼种为蓝子鱼）。
- `siganusmorph/ui_styles.py`：侧边栏品牌 → "FishMorph V1.0 / 蓝子鱼形态测量系统（Siganus）"。
- `pages/4_联系我们.py`：默认维护者名 → "FishMorph 项目维护者"。
- 未改动任何环境变量名（`SIGANUSMORPH_*`、`SIGANUS_*` 保持原样，避免破坏行为）。

---

## 6. 依赖说明

- 验证环境：**Python 3.10.1**（Windows 10 x64），streamlit 1.52.2，opencv-contrib 4.12.0，ultralytics 8.3.252，torch 2.5.1+cu121。
- 原 `requirements.txt` **缺失 `ultralytics` 与 `torch`**（V1 通过懒加载 import 实际需要）——新 requirements 已补齐。
- 原 requirements 中的 `reportlab`、`tables` 在 V1 运行时代码中从未被 import，未携带（本地环境 `tables` 本就未安装，应用不受影响）。

---

## 7. 验证结果

### 7.1 smoke test（`python tests/smoke_test.py`）—— 9/9 PASS

| # | 验证项 | 结果 |
|---|---|---|
| 1 | 44 个 Python 文件全部可编译 | PASS |
| 2 | 13 个 siganusmorph 核心模块可导入 | PASS |
| 3 | 三个模型权重从新项目 `models/` 就位并可解析 | PASS |
| 4 | YOLO 16 关键点推理（分割→裁剪→YOLO→掩膜校正全链路），输出 16/16 关键点 | PASS |
| 5 | Heatmap U-Net v0.5 加载 + 前向（输出 1×16×256×256） | PASS |
| 6 | `calculate_measurements` 形态参数计算（66 项测量值） | PASS |
| 7 | CSV / Excel / JSON 导出链路 | PASS |
| 8 | 校准栈（ArUco 检测：示例图检出 8 个标记 → 透视校正 → 0.1 mm/px 比例尺） | PASS |
| 9 | 新项目代码无原盘 `E:\` / `C:\Users` 硬编码引用 | PASS |

### 7.2 独立启动与页面验证（浏览器实测，`http://localhost:8503`）

| 验证项 | 结果 |
|---|---|
| Streamlit 服务从 `D:\1_yanjiusheng\FishMorph Local` 独立启动 | PASS（`/_stcore/health` = ok，服务日志无异常） |
| 首页加载（品牌区、功能卡片、流程图、新框架描述） | PASS |
| 4 个 V1 页面逐一加载（单鱼测量 / 批量测量 / 结果导出 / 联系我们），页面无异常报错 | PASS |
| 单鱼测量页上传组件、样本信息表单、自动测量按钮正常渲染 | PASS |
| 服务端日志无 Traceback / Exception | PASS |

### 7.3 逐项对照任务清单（17 项）

| 任务项 | 结果 | 说明 |
|---|---|---|
| 1 Streamlit 正常启动 | PASS | |
| 2 首页加载正常 | PASS | |
| 3 V1 页面均能加载 | PASS | 4 页浏览器实测 |
| 4 图片可以上传 | NOT TESTED* | 上传控件渲染正常；自动化浏览器不支持文件选择器，完整上传-测量链路由 smoke test #4 在 Python 层覆盖（同一 `preannotate_warped_image` 入口） |
| 5 校准功能正常 | PASS | smoke test #8（真实示例图 ArUco 检出+校正+比例尺） |
| 6 鱼体识别正常 | PASS | smoke test #4（蓝板分割成功） |
| 7 模型权重能加载 | PASS | smoke test #3/#5 |
| 8 16 关键点预测 | PASS | smoke test #4（16/16） |
| 9 形态参数计算 | PASS | smoke test #6 |
| 10 QC 正常 | PASS | smoke test #4 全链路含预标注 QC 生成 |
| 11 人工拖拽修改 | NOT TESTED* | 组件（`web_components/measurement_editor`）已迁移且页面渲染正常；拖拽交互需人工在浏览器中复核 |
| 12 修改后参数重算 | NOT TESTED* | 依赖 #11 的交互 |
| 13 单鱼流程正常 | PASS（自动段） | 导入→校准→识别→测量→QC 自动链路已验证；人工复核段见 #11 |
| 14 批量流程正常 | NOT TESTED* | 批量页加载正常；多文件批处理建议人工上传 2–3 张图复核 |
| 15 CSV 导出 | PASS | smoke test #7 |
| 16 Excel 导出 | PASS | smoke test #7 |
| 17 JSON 导出 | PASS | smoke test #7 |

\* 标注 NOT TESTED 的交互项需要人工在浏览器中操作（自动化环境不支持文件选择器/拖拽），或后续补充 Playwright 脚本验证。

---

## 8. 当前 Siganus 耦合点（多物种化时需要解耦的位置）

| 耦合点 | 位置 |
|---|---|
| 16 关键点定义（P1–P11、P7U/P7L、C1–C4 及中文引导文案） | `siganusmorph/config.py: KEYPOINT_DEFS` |
| 模型权重路径硬编码（3 个模型目录名） | `siganusmorph/preannotation.py`、`heatmap_preannotation.py`、`v06_preannotation.py` |
| 测量定义（体长/全长/体高/尾柄深等 66 项） | `siganusmorph/measurements.py: MEASUREMENT_DEFS`、`dual_axis_measurement.py`、`local_normal_measurement.py` |
| 蓝板分割（针对蓝子鱼拍摄背景的蓝色校准板） | `siganusmorph/segmentation.py: segment_fish_from_blue_board` |
| 尾鳍/尾部几何规则（含压缩尾修正） | `tail_geometry.py`、`caudal_base_geometry.py`、`compressed_tail_tl.py` |
| QC 阈值与几何规则 | `geometric_rules.py`、`preannotation.py:_local_structure_qc` |
| UI 文案（中文引导语、页面标题中的"蓝子鱼"） | `config.py` 引导文案、`app.py`、各页面 |
| 环境变量命名空间 | `SIGANUSMORPH_*`、`SIGANUS_*`（本次保留未改） |
| 输出目录命名 | `results/formal_v1_user_outputs/`（`remote_jobs.py`、批量页） |
| species profile | `config/species/siganus.yaml`（新建，尚未接入运行时） |

---

## 9. Git

- 原项目 `E:\1_yanjiusheng\SiganusMorph Local` 迁移前后 `git status --short` 均为空（工作区始终干净），本次迁移未对其造成任何修改。
- 新项目初始化独立 git 仓库（不携带原 `.git/`），首次提交：`chore: migrate stable SiganusMorph V1 into FishMorph Local`。
- `.gitignore` 沿用原策略：`models/` 与 `*.pt` 不入 git（权重保留在工作目录）。
