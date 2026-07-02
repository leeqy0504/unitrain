# configs/ 配置文件说明

## 文件结构

| 文件 | 用途 | 框架 |
|------|------|------|
| `example.yaml` | **通用示例模板** — 包含所有框架/任务组合的配置参考和完整参数说明 | 全部 |
| `jn_train.yaml` | JN 数据集 RF-DETR 分割训练 | RF-DETR |
| `jn_yolo.yaml` | JN 数据集 YOLO 分割训练 | Ultralytics |
| `predict_weights.yaml` | 使用训练好的权重进行推理 | RF-DETR |

## 配置格式

所有配置统一使用 **Nested 格式**（推荐），参数嵌套在命名空间下：

```yaml
framework: rfdetr          # 框架选择: rfdetr | ultralytics
model: seg-nano            # 模型规格 (见下方可选项)
task: segment              # 任务类型: detect | segment

data:
  path: ~/DATA/dataset/    # 数据集路径
  format: coco             # 数据格式: coco
  nc: 6                    # 类别数量 (可选，自动从数据集检测)

train:
  epochs: 100              # 训练轮数
  batch: 4                 # 每卡 batch size
  device: "2,3"            # GPU 设备 (见下方 GPU 配置)
  output_dir: outputs      # 输出目录

predict:
  threshold: 0.5           # 置信度阈值
  weights: ""              # 权重路径 (可选)

export:
  format: onnx             # 导出格式: onnx | torchscript
```

> `load_config()` 同时兼容 Flat 格式（如 `epochs: 100` 在顶层），但新配置请使用 Nested 格式。

---

## 完整参数说明

### 1. 顶层参数

| 参数 | 类型 | 必填 | 说明 | 可选值 |
|------|------|------|------|--------|
| `framework` | string | ✅ | 训练框架 | `rfdetr`, `ultralytics`, `yolo` |
| `model` | string | ✅ | 模型规格 | 见下方模型列表 |
| `task` | string | ✅ | 任务类型 | `detect` (目标检测), `segment` (实例分割), `obb` (旋转框检测) |

#### RF-DETR 模型选项

**目标检测:**
- `nano` - 384×384 分辨率，最快速度
- `small` - 512×512
- `medium` - 576×576（推荐平衡）
- `base` - 560×560，高精度
- `large` - 560×560，最高精度

**实例分割:**
- `seg-nano` - 312×312，最快速度
- `seg-small` - 432×432
- `seg-medium` - 504×504（推荐平衡）
- `seg-base` - 560×560，高精度
- `seg-large` - 560×560
- `seg-xlarge` - 672×672
- `seg-2xlarge` - 672×672，最高精度

#### Ultralytics 模型选项

**YOLO11 系列:**

| 任务 | 模型名 | 说明 |
|------|---------|------|
| 目标检测 | `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x` | n=nano, x=xlarge |
| 实例分割 | `yolo11n-seg`, `yolo11s-seg`, `yolo11m-seg`, `yolo11l-seg`, `yolo11x-seg` | |
| 旋转框检测 | `yolo11n-obb`, `yolo11s-obb`, `yolo11m-obb`, `yolo11l-obb`, `yolo11x-obb` | |

**YOLO26 系列:**

| 任务 | 模型名 | 说明 |
|------|---------|------|
| 目标检测 | `yolo26n`, `yolo26s`, `yolo26m`, `yolo26l`, `yolo26x` | 新一代架构 |
| 实例分割 | `yolo26n-seg`, `yolo26s-seg`, `yolo26m-seg`, `yolo26l-seg`, `yolo26x-seg` | |
| 旋转框检测 | `yolo26n-obb`, `yolo26s-obb`, `yolo26m-obb`, `yolo26l-obb`, `yolo26x-obb` | |
| 开放词汇 | 见下方 YOLOE 模型列表 | 支持 class_prompts |

**YOLOE 系列 (开放词汇/提示检测与分割):**

| Backbone | 文本/视觉提示模型 | Prompt-Free 模型 |
|----------|-------------------|------------------|
| YOLO11 | `yoloe-11s-seg`, `yoloe-11m-seg`, `yoloe-11l-seg` | `yoloe-11s-seg-pf`, `yoloe-11m-seg-pf`, `yoloe-11l-seg-pf` |
| YOLOv8 | `yoloe-v8s-seg`, `yoloe-v8m-seg`, `yoloe-v8l-seg` | `yoloe-v8s-seg-pf`, `yoloe-v8m-seg-pf`, `yoloe-v8l-seg-pf` |
| YOLO26 | `yoloe-26n-seg` … `yoloe-26x-seg` | `yoloe-26n-seg-pf` … `yoloe-26x-seg-pf` |

---

### 2. `data` 命名空间

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `path` | string | ✅ | - | 数据集根目录路径（支持 `~` 展开） |
| `format` | string | ✅ | `coco` | 数据格式，目前支持 COCO（OBB 任务自动跳过转换） |
| `nc` | int | ❌ | `80` | 类别数量（自动从数据集检测） |
| `names` | list | ❌ | `[]` | 类别名称列表（自动生成） |

**COCO 数据集目录结构:**
```
data/
├── images/
│   ├── train/
│   └── valid/
└── annotations/
    ├── train.json
    └── valid.json
```

**OBB (旋转框) 数据集目录结构:**
```
data/obb/
├── images/
│   ├── train/
│   └── val/
├── labels/               # YOLO OBB 格式 txt
│   ├── train/            # 每行: class_id x1 y1 x2 y2 x3 y3 x4 y4
│   └── val/
└── data.yaml
```

> OBB 任务 (`task: obb`) 数据需预先为 YOLO OBB 格式，系统会自动跳过 COCO→YOLO 转换。

### 2.5 `class_prompts` (开放词汇模型专用)

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `class_prompts` | list[str] | ❌ | `[]` | 开放词汇文本类名提示 |

用于 YOLOE 等开放词汇模型，在训练/推理前通过 `model.set_classes(class_prompts)` 设置类别：

```yaml
model: yoloe-26s-seg
class_prompts:
  - person
  - car
  - dog
```

> 使用 `class_prompts` 时，建议同步设置 `data.nc` 和 `data.names` 与 prompts 一致。

---

### 3. `train` 命名空间

#### 通用训练参数

| 参数 | 类型 | 必填 | 默认值 | 说明 | 框架支持 |
|------|------|------|--------|------|----------|
| `epochs` | int | ✅ | `100` | 训练轮数 | 全部 |
| `batch` | int | ✅ | - | 每卡 batch size | 全部 |
| `device` | int/string | ✅ | `0` | GPU 设备配置（见下方详细说明） | 全部 |
| `output_dir` | string | ❌ | `outputs` | 输出目录基路径 | 全部 |

#### RF-DETR 专用参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `lr` | float | ❌ | `1e-4` | 学习率 |
| `grad_accum_steps` | int | ❌ | `4` | 梯度累积步数（等效总 batch = batch × grad_accum_steps × num_gpus） |
| `early_stopping` | bool | ❌ | `false` | 是否启用早停 |
| `early_stopping_patience` | int | ❌ | `10` | 早停等待轮数（验证指标无提升） |
| `early_stopping_min_delta` | float | ❌ | `0.001` | 早停最小改善阈值 |
| `resume` | string | ❌ | `""` | 恢复训练的 checkpoint 路径 |

#### Ultralytics 专用参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `imgsz` | int | ❌ | `640` | 输入图像尺寸 |

---

### 4. `predict` 命名空间

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `threshold` | float | ❌ | `0.5` | 置信度阈值（0-1） |
| `weights` | string | ❌ | `""` | 自定义权重文件路径（不设置则使用预训练权重） |
| `output_dir` | string | ❌ | `outputs/predict` | 推理结果输出目录 |

---

### 5. `export` 命名空间

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `format` | string | ❌ | `onnx` | 导出格式 |

**支持的导出格式:**
- `onnx` - 通用格式，推荐
- `torchscript` - PyTorch JIT 格式

---

## GPU 设备配置详解

### 单卡训练
```yaml
device: 0           # 使用 GPU 0
device: 1           # 使用 GPU 1
```

### 多卡训练（DDP）
```yaml
device: "0,1"       # 使用 GPU 0 和 1（必须用字符串）
device: "2,3"       # 使用 GPU 2 和 3
device: "0,1,2,3"   # 使用 4 张 GPU
```

**多卡训练注意事项:**
- 必须用字符串格式（带引号）
- RF-DETR 自动启用 DDP (DistributedDataParallel)
- 总等效 batch = `batch × grad_accum_steps × num_gpus`
- 例如: batch=4, grad_accum=4, 2 GPUs → 总 batch=32

### CPU 训练
```yaml
device: "cpu"       # 仅 CPU（测试用，速度较慢）
```

### 环境变量方式
```bash
# 等价于 device: 1
CUDA_VISIBLE_DEVICES=1 ./run.sh train --config configs/xxx.yaml

# 等价于 device: "2,3"
CUDA_VISIBLE_DEVICES=2,3 ./run.sh train --config configs/xxx.yaml
```

---

## 权重文件管理

### 自动权重管理
- 所有预训练权重自动下载到 `weights/` 目录
- 框架自动解析模型对应的权重文件名
- 无需手动指定，开箱即用

### 使用自定义权重
```yaml
predict:
  weights: weights/my_best.pt              # 相对路径
  weights: /absolute/path/to/weights.pth   # 绝对路径
```

### 恢复训练
```yaml
train:
  resume: outputs/project/rfdetr_20260211_123456/checkpoint.pth
```

---

## 输出目录结构

训练完成后自动生成带时间戳的输出目录：

```
outputs/
└── your_project/
    └── rfdetr_20260211_143025/      # 自动时间戳
        ├── config.yaml              # 训练配置（自动复制）
        ├── checkpoint.pth           # 最新权重
        ├── checkpoint_best_ema.pth  # 最佳 EMA 权重
        ├── checkpoint_best_regular.pth
        ├── checkpoint_best_total.pth
        ├── results.json             # 训练结果
        ├── log.txt                  # 训练日志
        └── eval/                    # 验证结果
```

---

## 快速开始

### 1. 新建配置
```bash
cp configs/example.yaml configs/my_project.yaml
# 编辑 my_project.yaml: 修改 data.path, model, device 等
```

### 2. 训练
```bash
./run.sh train --config configs/my_project.yaml
```

### 3. 推理
```bash
./run.sh predict --config configs/my_project.yaml --source image.jpg
```

### 4. 导出
```bash
./run.sh export --config configs/my_project.yaml --format onnx
```

---

## 常见配置示例

### 小数据集快速验证
```yaml
train:
  epochs: 10
  batch: 2
  device: 0
  early_stopping: true
  early_stopping_patience: 3
```

### 大规模生产训练
```yaml
train:
  epochs: 300
  batch: 8
  device: "0,1,2,3"        # 4 卡
  grad_accum_steps: 2      # 总 batch = 8×2×4 = 64
  early_stopping: true
  early_stopping_patience: 20
```

### CPU 调试训练
```yaml
train:
  epochs: 1
  batch: 1
  device: "cpu"
```

---

## 更多示例

参考现有配置文件:
- `jn_train.yaml` - RF-DETR 分割训练完整示例
- `jn_yolo.yaml` - YOLO 分割训练示例
- `predict_weights.yaml` - 推理配置示例
