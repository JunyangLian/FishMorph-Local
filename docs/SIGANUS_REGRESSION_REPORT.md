# Siganus V1 迁移回归报告

- **日期**：2026-09-26
- **执行脚本**：`tests/regression_siganus.py`（`run --side E|D` → `compare` → `historical`）
- **结论**：**迁移保真 PASS**——原项目代码（E:）与 FishMorph 副本（D:）在相同输入下，全链路输出**逐位一致**（552 个数值字段，最大差异 0）。

## 1. 测试设计

| 层级 | 内容 | 判定方式 |
|---|---|---|
| **Tier 1：迁移保真 A/B** | 同一批图像分别在 E: 原代码与 D: 副本代码上运行当前 V1 全链路（校准图→蓝板分割→YOLO 16 关键点→掩膜校正→尾部几何→形态测量），逐字段对比 | 门禁：差异必须为 0 |
| **Tier 2：历史对照** | 与原项目 v0.6.4 时期的 e2e 回归记录（`results/e2e_regression_v0.6.4/test_measurements.csv`）对比核心直线测量值 | 信息性（期间存在 v0.6.5–v0.6.9 流水线演进） |

**测试集**：原项目 e2e 清单中覆盖三类形态的 6 张标准校正图（fixture 复制于 `data/regression_images/`，56 MB，已被 gitignore）：
- 直体标准：`real_089`、`real_076`
- 轻度弯曲：`real_042`、`real_054`
- 复杂尾型：`real_028`、`real_063`

**输入约定**：`load_image_file`（PIL→RGB）+ `mm_per_pixel=0.1`（e2e 基线的 `board_mm_per_pixel`）。

**原项目安全措施**：E: 侧仅读取代码与执行；`sys.dont_write_bytecode=True` 防止生成 `__pycache__`；`project_root` 指向本项目内的 `data/regression_outputs/_e_side_project_root/`，全部写盘落在 D:。运行后复查：原项目 `git status --short` 仍为空，包内无新增缓存目录。

## 2. Tier 1 结果：迁移保真（门禁）

```text
[PASS] real_089: 数值字段最大差异 0
[PASS] real_076: 数值字段最大差异 0
[PASS] real_042: 数值字段最大差异 0
[PASS] real_054: 数值字段最大差异 0
[PASS] real_028: 数值字段最大差异 0
[PASS] real_063: 数值字段最大差异 0
数值字段总数 552; 全局最大差异 0
PASS: E: 原代码与 D: 副本在相同输入下输出完全一致
```

覆盖：16 关键点坐标（×2）、分割成功标志与质量、QC 字段、全部形态测量值（SL/TL/体高/头长/吻长/尾柄长/尾柄深及轴系变体、曲率指数等）。**结论：本次迁移在执行级无任何行为差异**（品牌层 5 处修改均不在推理路径上）。

当前 V1 流水线输出（D: 侧，mm）：

| 图像 | SL | TL | 备注 |
|---|---|---|---|
| real_089 | 154.15 | 188.08 | 直体 |
| real_076 | 156.74 | 189.15 | 直体 |
| real_042 | 178.02 | 213.38 | 轻度弯曲 |
| real_054 | 148.56 | 182.15 | 轻度弯曲 |
| real_028 | 150.13 | 178.68 | 复杂尾型（TL 回退=SL，P7V 无效场景） |
| real_063 | 164.77 | 200.67 | 复杂尾型 |

## 3. Tier 2 结果：与 v0.6.4 e2e 历史对照（信息性）

| 指标（mm） | max 偏差 | mean 偏差 |
|---|---|---|
| SL_straight_mm | 4.94 | 3.20 |
| body_depth_mm | 3.43 | 2.29 |
| head_length_straight_mm | 10.48 | 2.45 |
| snout_length_straight_mm | 5.24 | 2.31 |
| caudal_peduncle_length_straight_mm | 2.49 | 1.14 |
| caudal_peduncle_depth_mm | 2.31 | 1.34 |

解读：e2e 记录产自 v0.6.4 时期，其后 V1 合入了 v0.6.5–v0.6.9 的测量定义与尾部/体高几何修正（见 `docs/VERSION_HISTORY.md`），关键点最大偏移 64–104 px（集中在 P1/P3 等由掩膜/边缘规则修正的点）。偏差量级与这些演进一致，**不构成迁移回归信号**；迁移保真以 Tier 1 零差异为准。

## 4. 过程中发现并修复的问题（对后续开发有用）

1. **BGR/RGB 输入陷阱**：`siganusmorph.image_utils.ensure_rgb` 是直通实现（假定输入已是 RGB）。首次运行误用 `cv2.imread`（BGR）导致模型预测大幅漂移（real_089 SL 236.8 vs 正确 154.1）。App 本身经 `image_from_bytes`（RGB）输入，不受影响；但**任何用 OpenCV 直接读图做离线实验的脚本必须先 `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`**。罗非鱼泛化试测时务必沿用 `load_image_file`。
2. 回归脚本过滤器的顶层字段排除（`side`/`image`）需精确匹配，避免把标签差异误判为数值差异。

## 5. 尚未覆盖（需人工/后续）

- 批量页多文件流程、人工拖拽修正与改后重算（UI 交互，自动化浏览器不支持文件选择器与画布拖拽）；
- 与人工确认关键点（`realworld_review_5_24_v0.6.5`）的精度对照——建议在罗非鱼试测前补做，可同时产出当前模型在 Siganus 上的准确率基线。
