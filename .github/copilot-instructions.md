# UniTrain Copilot Instructions

## Project Overview

UniTrain 是一个**统一深度学习训练框架封装**，通过单一 API 调用 RF-DETR 和 Ultralytics (YOLO) 两套框架。核心设计原则：

- **环境隔离**：使用 `uv` 为每个框架创建独立虚拟环境 (`.venv-rfdetr`, `.venv-yolo`)，避免依赖冲突
- **数据标准化**：以 COCO 格式为标准，运行时自动转换为各框架所需格式
- **配置统一**：YAML 配置支持 nested 和 flat 两种格式，自动适配目标框架

## Architecture

```
run.sh              # 主入口：环境初始化、框架切换、命令路由
├── cli/
│   ├── train.py    # 训练入口 (训练后自动评估)
│   ├── eval.py     # 评估入口
│   ├── predict.py  # 推理入口
│   └── export.py   # 导出入口
└── unitrain/
    ├── config_gen.py      # UnifiedConfig: YAML ⟷ 框架配置转换
    ├── eval_report.py     # 统一评估报告 (11 种图表 + 多格式输出)
    ├── data_converter.py  # COCO → YOLO 格式转换
    ├── utils.py           # 共享工具函数 (输出目录时间戳等)
    └── runners/
        ├── base.py        # BaseRunner 抽象基类
        ├── rfdetr.py      # RFDETRRunner
        ├── ultralytics.py # UltralyticsRunner
        └── _scripts/      # 框架 venv 中执行的脚本
            ├── rfdetr_eval.py   # RF-DETR 评估 (monkey-patch 方式)
            ├── yolo_eval.py     # YOLO 评估
            ├── rfdetr_train.py  # RF-DETR 训练
            ├── yolo_train.py    # YOLO 训练
            └── weight_utils.py  # 权重路径解析与下载
```

**关键执行流程**：Runner 通过 `subprocess` 在对应 venv 中执行 Python 脚本，而非直接导入框架模块。

## Commands & Workflows

```bash
# 环境初始化（必须先执行）
./run.sh setup                    # 初始化所有框架环境
./run.sh setup --framework rfdetr # 仅初始化 RF-DETR

# 训练/推理/导出/评估
./run.sh train --config configs/rfdetr.yaml
./run.sh train --config configs/rfdetr.yaml --skip-eval  # 训练后不自动评估
./run.sh predict --config configs/rfdetr.yaml --source image.jpg
./run.sh export --config configs/rfdetr.yaml --format onnx
./run.sh eval --config configs/rfdetr.yaml --weights outputs/.../best.pth

# 进入框架 venv 进行调试
./run.sh shell --framework rfdetr
```

**训练后自动评估**：训练完成后默认自动运行评估，评估结果保存在 `<训练输出目录>/eval/`。使用 `--skip-eval` 跳过。

## Configuration Patterns

配置文件位于 `configs/`，支持两种格式：

```yaml
# Nested 格式 (推荐)
framework: rfdetr  # rfdetr | ultralytics
model: medium      # RF-DETR: nano|small|medium|base|large|seg
train:
  epochs: 100
  batch: 4
  output_dir: outputs

# Flat 格式 (RF-DETR 原生兼容)
framework: rfdetr
model: base
epochs: 100
batch_size: 4
dataset_dir: data/coco
```

**加载逻辑**见 [config_gen.py](unitrain/config_gen.py#L90-L130) 的 `load_config()`。

## Key Conventions

1. **新增框架**：在 [runners.py](unitrain/runners.py) 中继承 `BaseRunner`，实现 `train/predict/export/eval` 方法
2. **Runner 执行脚本时自动注入 `utils.py` 工具函数**（如 `get_timestamped_output_dir`），见 `UTILS_INJECT_CODE`
3. **输出目录**：自动添加时间戳 `outputs/rfdetr_20260203_143025/`
4. **模型映射**：`RFDETRRunner.MODEL_MAP` 将配置中的 `model: medium` 映射为 `RFDETRMedium` 类
5. **框架 venv 路径约定**：`.venv-{框架名}` 位于项目根目录
6. **权重目录**：所有预训练权重统一存放在 `weights/` 目录，脚本通过 `_scripts/weight_utils.py` 自动解析路径，支持自动下载和迁移

## Data Format

- **输入**：COCO 格式 (`data/images/`, `data/annotations/train.json`)
- **Ultralytics 自动转换**：`cli/train.py --convert-data` 生成 `data_yolo/` 含 YOLO 格式标签和 `data.yaml`
- 转换逻辑见 [data_converter.py](unitrain/data_converter.py)

## Evaluation System

### 执行流程

1. **训练后自动评估**：训练完成后默认运行评估 (`--skip-eval` 跳过)
2. **独立评估**：`./run.sh eval --config ... --weights ...`
3. Runner 调用 `_scripts/rfdetr_eval.py` 或 `_scripts/yolo_eval.py` 在对应 venv 中执行
4. 脚本输出 `eval_metrics.json`，由 `eval_report.py` 生成统一报告

### RF-DETR 评估 (Monkey-Patch 方案)

RF-DETR `model.train(eval=True)` 仅保存 `eval.pth` 后立即返回，不输出 `results.json` 或详细指标。因此采用 monkey-patch 方案：

```python
# rfdetr_eval.py 核心机制
_captured = {}  # 全局捕获字典

def _hooked_evaluate(...):
    stats, coco_evaluator = original_evaluate(...)
    _captured["stats"] = stats           # test_stats 含 results_json
    _captured["coco_evals"] = coco_evaluator.coco_eval  # {bbox, segm} 的 COCOeval 对象
    return stats, coco_evaluator

rfdetr.main.evaluate = _hooked_evaluate  # 替换原函数
model.train(eval=True)                    # 触发评估
# 之后从 _captured 提取所有指标
```

**F1-Confidence 曲线重建**：从 COCO `evalImgs` 数据 (IoU@0.50) 提取 `dtScores`、`dtMatches`、`dtIgnore`，遍历 201 个置信度阈值计算每类 + 平均 F1。

### 评估输出 (11 张图表)

两个框架产出完全相同的图表集：

```
eval/
├── eval_metrics.json        # 完整指标 JSON
├── report.csv / .md         # 表格报告
└── plots/
    ├── per_class_mAP50.png          # 每类 mAP@50 柱状图 (含 0-1 和 0.8-1 双面板)
    ├── per_class_f1.png             # 每类 F1 柱状图
    ├── per_class_precision.png      # 每类 Precision 柱状图
    ├── per_class_recall.png         # 每类 Recall 柱状图
    ├── f1_confidence_box.png        # Box F1 vs Confidence 曲线 (每类 + 平均)
    ├── f1_confidence_box_zoomed.png # Box F1 放大 (Y: 0.9-1.0)
    ├── f1_confidence_mask.png       # Mask F1 vs Confidence (分割模型)
    ├── f1_confidence_mask_zoomed.png# Mask F1 放大
    ├── training_loss.png            # 训练损失曲线
    ├── mAP_epochs.png               # mAP vs Epochs
    └── overall_summary.png          # 总体雷达图
```

### 统一指标 JSON 结构

```json
{
  "overall": {"mAP50": 0.98, "mAP50_95": 0.80, "precision": 0.98, "recall": 0.97, "f1": 0.97},
  "per_class": [{"name": "cls1", "mAP50": 0.99, "precision": 0.98, "recall": 0.96, "f1": 0.97}],
  "coco_stats": {"bbox": [12 个 COCO 标准指标], "segm": [...]},
  "f1_confidence": {"confidence": [201], "f1_per_class": {"cls1": [201]}, "f1_mean": [201], "best_conf": 0.45, "best_f1": 0.97},
  "mask_f1_confidence": { ... },
  "training_log": {"epochs": [...], "loss": [...], "mAP50": [...]}
}
```

## Vendor Submodules

`vendors/rf-detr/` 是 git 子模块。修改框架代码后需提交子模块更新。RF-DETR API 参考 [vendors/rf-detr/rfdetr/__init__.py](vendors/rf-detr/rfdetr/__init__.py)。

## Adding New Features

- 新增配置字段：同时更新 `UnifiedConfig` dataclass 和 `load_config()` 解析逻辑
- 新增框架命令：修改 `run.sh` 的 `FRAMEWORK_*` 映射表
- 依赖变更：更新 `envs/rfdetr.txt` 或 `envs/ultralytics.txt`

## Changelog 规范

每次 Plan 模式工作结束后，**必须**在项目根目录提供/更新 `CHANGELOG.md` 文件，记录本次变更内容。格式参考：

```markdown
# Changelog

## [YYYY-MM-DD] - 简要标题
### Added / Changed / Fixed
- 变更描述
```
