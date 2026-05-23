# 智能温室番茄采摘机器人系统

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-detection-green.svg)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/opencv-4.8+-brightgreen.svg)](https://opencv.org/)

## 项目概述

基于 YOLOv8 目标检测的温室番茄采摘机器人软件模拟系统。采用伪 ROS2 架构，实现番茄检测、成熟度分类、3D 定位、优先级排序和可视化全流程。

**检测结果（100 张真实温室图像）**：

| 指标 | 值 |
|---|---|
| 检测番茄总数 | 1441 |
| 平均每张检测数 | 14.41 |
| 平均置信度 | 82.9% |
| 成熟度分布 | 全熟 170 / 半熟 286 / 未熟 985 |

## 系统架构

```
CameraNode ──image──▶ RecognitionNode ──detections──▶ PriorityNode ──target──▶ VisualizationNode
                          (YOLOv8)                    (评分排序)                 (标注输出)
                                                                           StatusNode (状态监控)
```

| 节点 | 功能 | 模型/算法 |
|---|---|---|
| CameraNode | 图像采集 + 合成深度图 | OpenCV |
| RecognitionNode | 番茄检测 + 3D 定位 | YOLOv8 (model_hub_n.pt) |
| PriorityNode | 最优采摘目标选择 | 多因素加权评分 |
| VisualizationNode | 检测结果标注输出 | OpenCV |
| StatusNode | 系统状态统计 | - |

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

依赖：opencv-python, numpy, pyyaml, ultralytics, matplotlib, pytest

### 运行检测

```bash
# 全量检测 (100 张)
python main.py

# 指定数量
python main.py --max-images 10

# 自定义参数
python main.py --data-dir data/DatasetId_360176_1652587113/Images --conf 0.5
```

### 生成分析报告

```bash
python results/analysis/generate_report.py
```

输出：
- `results/analysis/report.json` — 100 张图片逐张检测详情 + 汇总统计
- `results/analysis/detection_grid.jpg` — 9 宫格检测结果拼图
- `results/analysis/summary_report.jpg` — 总结报告图

### 运行测试

```bash
pytest
```

## 项目结构

```
.
├── config/
│   └── algorithm_params.yaml       # YOLO 参数、优先级权重、深度估计参数
├── data/
│   ├── calibration/
│   │   └── camera_params.yaml      # 相机内参 (fx=910, fy=910, cx=640, cy=360)
│   └── DatasetId_360176_1652587113/
│       ├── Annotations/            # COCO 标注
│       └── Images/                 # 真实温室番茄图像 (100张, 3024x4032)
├── external_models/
│   ├── model_hub_n.pt              # YOLOv8 番茄检测模型
│   └── tomato_detect.py            # 独立检测脚本 (参考)
├── main.py                         # 主程序入口
├── results/
│   ├── analysis/                   # 分析报告 + 脚本
│   └── detection_output/           # 检测可视化结果 (每张图标注后输出)
├── src/
│   ├── algorithms/
│   │   ├── yolo_detection.py       # YOLOv8 检测器 (Fully-ripe/Semi-ripe/Unripe)
│   │   ├── coordinate_transform.py # 单目 3D 坐标计算
│   │   └── priority_scoring.py     # 多因素优先级评分
│   ├── mock_ros2/
│   │   ├── core.py                 # Node, Topic, Publisher, Subscriber
│   │   └── message_types.py        # Image, Detection, Status 消息定义
│   └── nodes/
│       ├── camera_node.py          # 图像采集节点
│       ├── recognition_node.py     # YOLO 检测节点
│       ├── priority_node.py        # 优先级排序节点
│       ├── visualization_node.py   # 可视化标注节点
│       └── status_node.py          # 状态监控节点
├── tests/                          # 单元测试
└── 文档/                           # 课程相关文档
```

## 核心算法

### 番茄检测 (YOLOv8)

使用预训练 YOLOv8 模型，输出三个成熟度类别：

| 类别 | 颜色 | 含义 |
|---|---|---|
| Fully-ripe | 红色框 | 完全成熟，可立即采摘 |
| Semi-ripe | 橙色框 | 半熟，需等待 |
| Unripe | 绿色框 | 未成熟 |

### 3D 定位

基于单目视觉的已知尺寸深度估计：

```
Z = fx × D_real / D_pixel
X = (u - cx) × Z / fx
Y = (v - cy) × Z / fy
```

### 优先级评分

| 因素 | 权重 | 说明 |
|---|---|---|
| 置信度 | 0.3 | YOLO 检测置信度 |
| 成熟度 | 0.3 | Fully-ripe=1.0, Semi-ripe=0.6, Unripe=0.2 |
| 可采摘性 | 0.2 | 基于圆度和大小评估 |
| 距离 | 0.2 | 距离越近分数越高 |

## 配置参数

`config/algorithm_params.yaml`：

```yaml
yolo:
  model_path: "external_models/model_hub_n.pt"
  conf_threshold: 0.5
  iou_threshold: 0.45

priority_scoring:
  weights:
    confidence: 0.3
    maturity: 0.3
    accessibility: 0.2
    distance: 0.2

depth_estimation:
  real_diameter_mm: 80
```

## 可视化输出

检测框颜色按成熟度分类（红/橙/绿），黄色框表示当前最优采摘目标。

标签格式：`ID:{id} C:{confidence} M:{maturity}`

## 命令行参数

```
python main.py [OPTIONS]

  --data-dir PATH        图像目录 (default: data/DatasetId_360176_1652587113/Images)
  --config PATH          参数文件 (default: config/algorithm_params.yaml)
  --camera-params PATH   相机参数 (default: data/calibration/camera_params.yaml)
  --rate FLOAT           发布频率 Hz (default: 1.0)
  --max-images INT       最大处理数 (default: 全部)
```

## 开发者

- 课程：智能机器人
- 学校：南京理工大学 (NJUST)
- 日期：2026 年 5 月
