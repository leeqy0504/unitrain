# Changelog

## [2026-03-09] - 修正 YOLOE 模型命名

### Fixed
- **模型命名修正**: 将所有 `yolo26n-world` … `yolo26x-world` 错误名称修正为官方 YOLOE 命名 (`yoloe-26n-seg`, `yoloe-11l-seg` 等)
  - 更新 `unitrain/runners/ultralytics.py` 中 `KNOWN_MODELS` 集合，添加完整 YOLOE 模型列表 (11/v8/26 三种 backbone × Text+PF 变体)
  - 更新 `README.md` 支持框架表和模型列表，新增独立 YOLOE 开放词汇模型章节
  - 更新 `configs/README.md` YOLO26 表格，新增 YOLOE 系列模型表
  - 更新 `configs/example.yaml` 第 8 节开放词汇配置示例
  - 更新 `unitrain/config_gen.py`、`_scripts/yolo_train.py`、`yolo_predict.py`、`yolo_export.py` 中的注释

---

## [2026-03-06] - 评估报告增强：RF-DETR 与 YOLO 图表统一

### Added
- **RF-DETR F1-Confidence 曲线**: 通过 monkey-patch `rfdetr.main.evaluate` 捕获完整 COCO evaluator 数据，从 `evalImgs` 原始数据重建 F1 vs Confidence 曲线 (201 点)
  - `_install_eval_hook()`: 钩子函数捕获 `test_stats` 和 `coco_eval` 对象
  - `_f1_confidence_curves()`: 在 IoU@0.50 下遍历 201 个置信度阈值，计算每类 + 平均 F1
  - 同时支持 bbox 和 segm (mask) 两种评估类型
- **RF-DETR per-class P/R/F1 指标**: 从 `results_json` / `results_json_masks` 提取每类 Precision、Recall、F1
- **F1-Confidence 放大图**: Y 轴范围 0.9-1.0 的放大视图 (`f1_confidence_box_zoomed.png`, `f1_confidence_mask_zoomed.png`)

### Changed
- **rfdetr_eval.py 完全重写**: 从日志解析方案改为 monkey-patch 方案，直接捕获框架内部评估数据
  - 使用 `_hooked_evaluate()` 替换 `rfdetr.main.evaluate`，获取 `stats` 和 `coco_evaluator` 完整返回值
  - 支持从 `coco_eval_bbox` 和 `coco_eval_masks` 提取 COCO 12-stat
  - 从 `eval.pth` 文件解析后备指标 (precision/recall/scores 张量)
- **eval_report.py 增强**:
  - `plot_per_class_bar()`: 跳过全零/全 None 指标的绘制
  - `print_terminal_table()`: 自适应列 — 根据 `has_prf` 标志动态显示/隐藏 P/R/F1 列
  - COCO stats 迭代现支持 `"segm"` 键 (RF-DETR 分割模型)
- RF-DETR 与 YOLO 评估输出完全对齐：两个框架均产出相同的 11 张图表

### Fixed
- RF-DETR `eval=True` 模式下数据丢失问题：`model.train(eval=True)` 仅保存 `eval.pth` 后立即返回，不写入 `results.json` 或 `log.txt`。通过 monkey-patch 方案解决

---

## [2026-03-05] - 模型评估功能

### Added
- **评估命令** (`./run.sh eval`): 支持对已训练模型进行独立评估
  - RF-DETR 评估脚本 (`_scripts/rfdetr_eval.py`)
  - YOLO 评估脚本 (`_scripts/yolo_eval.py`)
  - CLI 入口 (`cli/eval.py`) 支持 `--weights`, `--split`, `--output-dir` 参数
- **训练后自动评估**: 训练完成后默认自动运行评估，结果保存在 `<训练输出目录>/eval/`
  - 训练脚本输出结构化标记 (`UNITRAIN_TRAIN_OUTPUT_DIR`, `UNITRAIN_BEST_WEIGHTS`)
  - Runner `train()` 方法返回训练元数据 (output_dir, best_weights)
  - 支持 `--skip-eval` 跳过自动评估
- **统一评估报告** (`unitrain/eval_report.py`):
  - 11 种图表: Per-class 柱状图 (mAP50/F1/Precision/Recall)、训练损失曲线、总体雷达图、mAP 曲线、F1-Confidence 曲线 (box/mask, 标准/放大)
  - 4 种输出格式: 终端表格、JSON、CSV、Markdown
- **评估配置字段**: `eval_weights`, `eval_split`, `eval_conf_threshold`, `eval_iou_threshold`, `eval_output_dir`
- `run.sh` 新增 `eval` 命令路由，`train` 命令支持 `--skip-eval` / `--skip-gpu-check` / `--convert-data` 透传

### Changed
- `BaseRunner.train()` 返回类型从 `None` 改为 `dict[str, Any] | None`
- `.github/copilot-instructions.md` 更新架构图、命令文档、Changelog 规范
