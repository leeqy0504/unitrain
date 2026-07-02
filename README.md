# UniTrain

> 通用模型训练框架 (Universal Training Framework)

轻量级深度学习目标检测框架统一封装，一键调用 RF-DETR 和 Ultralytics。

## 特性

- 🔧 **统一接口**：一套 API 调用不同框架的训练、推理、导出功能
- 🔒 **环境隔离**：使用 uv 为每个框架创建独立虚拟环境，避免依赖冲突
- 🚀 **智能初始化**：自动检测并初始化所需的子仓库和虚拟环境
- 📊 **通用配置**：YAML 配置自动转换为各框架所需格式
- 📁 **数据转换**：以 COCO 格式为标准，自动转换为 YOLO 格式

## 快速开始

### 1. 初始化环境

```bash
# 克隆项目
git clone <repo-url>
cd unitrain

# 安装 uv (如未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 一键初始化所有环境（自动克隆子仓库 + 创建虚拟环境）
./run.sh setup

# 或仅初始化特定框架
./run.sh setup --framework rfdetr
./run.sh setup --framework yolo
```

### 2. 准备数据

将数据集放入 `data/` 目录，使用 COCO 格式：

```
data/
├── images/
│   ├── train/
│   └── val/
└── annotations/
    ├── train.json
    └── val.json
```

### 3. 训练

```bash
# 使用统一配置训练（自动检查环境）
./run.sh train --config configs/rfdetr.yaml
./run.sh train --config configs/example.yaml
```

### 4. 推理

```bash
./run.sh predict --config configs/example.yaml --source image.jpg
```

### 5. 导出

```bash
./run.sh export --config configs/example.yaml --format onnx
```

### 6. 评估

```bash
# 独立评估（需指定权重路径）
./run.sh eval --config configs/rfdetr.yaml --weights outputs/.../best.pth

# 训练后自动评估（默认行为，训练完成后自动运行）
./run.sh train --config configs/rfdetr.yaml

# 跳过自动评估
./run.sh train --config configs/rfdetr.yaml --skip-eval
```

评估报告输出至 `<训练输出目录>/eval/`，包含：

| 文件 | 说明 |
|------|------|
| `eval_metrics.json` | 完整指标 JSON |
| `report.csv` / `report.md` | 表格报告 |
| `plots/per_class_mAP50.png` | 每类 mAP@50 柱状图 (含 0-1 和 0.8-1 双面板) |
| `plots/per_class_f1.png` | 每类 F1 柱状图 |
| `plots/per_class_precision.png` | 每类 Precision 柱状图 |
| `plots/per_class_recall.png` | 每类 Recall 柱状图 |
| `plots/f1_confidence_box.png` | Box F1 vs Confidence 曲线 (每类 + 平均) |
| `plots/f1_confidence_box_zoomed.png` | Box F1 放大 (Y: 0.9-1.0) |
| `plots/f1_confidence_mask.png` | Mask F1 vs Confidence (分割模型) |
| `plots/f1_confidence_mask_zoomed.png` | Mask F1 放大 |
| `plots/training_loss.png` | 训练损失曲线 |
| `plots/mAP_epochs.png` | mAP vs Epochs |
| `plots/overall_summary.png` | 总体雷达图 |

### 7. 进入框架环境（高级用法）

```bash
# 进入 RF-DETR 虚拟环境，可直接使用框架 API
./run.sh shell --framework rfdetr

# 进入 YOLO 虚拟环境
./run.sh shell --framework yolo
```

## 配置示例

```yaml
# configs/example.yaml
framework: ultralytics  # 或 rfdetr
model: yolo11n          # ultralytics: yolo11n/yolo26n/..., rfdetr: nano/small/medium/large
task: detect            # detect | segment | obb

data:
  path: data/
  format: coco          # 自动转换为框架所需格式

train:
  epochs: 100
  imgsz: 640
  batch: 16
  device: 0             # GPU 选择：0 (单GPU) | "0,1" (多GPU) | "cpu"

export:
  format: onnx          # onnx, tensorrt, etc.
```

## GPU 选择

UniTrain 支持灵活的 GPU 设备选择：

### 单 GPU 训练

```yaml
train:
  device: 0              # 使用 GPU 0
```

### 多 GPU 训练

```yaml
train:
  device: "0,1"          # 使用 GPU 0 和 1（需要用字符串格式）
  batch: 8               # 每个 GPU 的 batch size
```

### CPU 训练

```yaml
train:
  device: "cpu"          # 使用 CPU（测试用）
```

### 使用环境变量（替代方法）

```bash
# 指定使用 GPU 1
CUDA_VISIBLE_DEVICES=1 ./run.sh train --config configs/rfdetr.yaml

# 使用多个 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh train --config configs/example.yaml
```

更多 GPU 配置示例请参考：[configs/example.yaml](configs/example.yaml) 末尾的 GPU 配置参考部分

## 支持的框架

| 框架 | 模型 | 任务 |
|------|------|------|
| Ultralytics | YOLO11 (n/s/m/l/x), YOLO26 (n/s/m/l/x), YOLOE | 检测、分割、OBB、开放词汇 |
| RF-DETR | Nano/Small/Medium/Base/Large/Seg | 检测、分割 |

### YOLO26 模型列表

| 任务 | 模型名 | 配置示例 |
|------|---------|----------|
| 目标检测 | `yolo26n/s/m/l/x` | `model: yolo26n` |
| 实例分割 | `yolo26n-seg` … `yolo26x-seg` | `model: yolo26n-seg`, `task: segment` |
| 旋转框 (OBB) | `yolo26n-obb` … `yolo26x-obb` | `model: yolo26n-obb`, `task: obb` |

## 数据格式

推荐使用 **COCO 格式** 作为统一数据格式：

- RF-DETR：原生支持 COCO
- Ultralytics：自动转换为 YOLO 格式
- **OBB 任务**：数据需预先为 YOLO OBB 格式 (`class_id x1 y1 x2 y2 x3 y3 x4 y4`)，设置 `task: obb` 时自动跳过 COCO 转换

转换命令：
```bash
python -m unitrain.data_converter --input data/coco --output data/yolo --task detect
python -m unitrain.data_converter --input data/coco --output data/yolo --task segment
```

## 项目结构

```
unitrain/
├── run.sh                # 统一运行脚本（推荐入口）
├── vendors/              # Git 子模块（自动初始化）
│   ├── rf-detr/
│   └── ultralytics/
├── envs/                 # 各框架依赖配置
├── configs/              # 示例配置文件
├── unitrain/             # 核心封装代码
│   ├── config_gen.py     # 配置生成器
│   ├── eval_report.py    # 统一评估报告 (11 种图表 + 多格式输出)
│   ├── data_converter.py # 数据格式转换
│   ├── utils.py          # 通用工具函数
│   └── runners/          # 框架 Runner
│       ├── base.py       # BaseRunner 抽象基类
│       ├── rfdetr.py     # RFDETRRunner
│       ├── ultralytics.py# UltralyticsRunner
│       └── _scripts/     # 框架 venv 中执行的脚本
│           ├── rfdetr_eval.py   # RF-DETR 评估
│           ├── yolo_eval.py     # YOLO 评估
│           ├── rfdetr_train.py  # RF-DETR 训练
│           ├── yolo_train.py    # YOLO 训练
│           └── weight_utils.py  # 权重路径解析
├── cli/                  # CLI 入口脚本
│   ├── train.py          # 训练入口 (训练后自动评估)
│   ├── eval.py           # 评估入口
│   ├── predict.py        # 推理入口
│   └── export.py         # 导出入口
```
