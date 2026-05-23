# 智能温室番茄采摘机器人系统

## Smart Greenhouse Tomato Picking Robot System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/opencv-4.8+-green.svg)](https://opencv.org/)
[![ROS2 Mock](https://img.shields.io/badge/ros2-mock-orange.svg)]()

## 项目概述 / Project Overview

本项目是一个基于计算机视觉的温室番茄采摘机器人软件系统，实现了完整的番茄识别、定位、优先级排序和可视化流程。系统采用模拟 ROS2 架构，可在无实际硬件环境下进行算法验证和演示。

This project is a computer vision-based greenhouse tomato picking robot software system that implements a complete pipeline for tomato recognition, localization, priority scoring, and visualization. The system uses a mock ROS2 architecture, enabling algorithm validation and demonstration without physical hardware.

### 核心功能 / Key Features

- **番茄检测**：基于 HSV 色彩空间的红色分割算法
- **3D 定位**：单目相机深度估计，计算番茄空间坐标
- **优先级排序**：多因素综合评分，选择最优采摘目标
- **可视化输出**：实时标注检测结果与系统状态
- **真实数据测试**：支持真实温室番茄图像数据集

## 系统架构 / System Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Camera Node │───▶│ Recognition │───▶│   Priority  │───▶│ Visualization│    │ Status Node │
│  (图像采集)  │    │   Node      │    │    Node     │    │    Node      │    │  (状态监控)  │
│             │    │ (番茄检测)   │    │ (优先级排序) │    │  (可视化输出) │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │                  │                  │
       ▼                  ▼                  ▼                  ▼                  ▼
/camera/image_raw   /detection/tomatoes  /planning/target  results/detection_output  /system/status
/camera/depth
```

### 节点说明 / Node Descriptions

| 节点 | 职责 | 订阅话题 | 发布话题 |
|------|------|----------|----------|
| CameraNode | 加载并发布测试图像与合成深度图 | - | `/camera/image_raw`, `/camera/depth` |
| RecognitionNode | 运行检测算法，识别番茄并计算3D坐标 | `/camera/image_raw`, `/camera/depth` | `/detection/tomatoes` |
| PriorityNode | 根据多因素评分选择最优采摘目标 | `/detection/tomatoes` | `/planning/target` |
| VisualizationNode | 绘制检测框、标签并保存可视化结果 | `/camera/image_raw`, `/detection/tomatoes`, `/planning/target`, `/system/status` | - |
| StatusNode | 统计处理图像与检测数量 | `/detection/tomatoes` | `/system/status` |

## 快速开始 / Quick Start

### 环境要求 / Requirements

- Python 3.8+
- OpenCV 4.8+
- NumPy, PyYAML, Matplotlib, pytest

### 安装依赖 / Install Dependencies

```bash
pip install -r requirements.txt
```

### 准备测试数据 / Prepare Test Data

将真实番茄图像数据集放入 `data/DatasetId_360176_1652587113/Images/` 目录：

```
data/
└── DatasetId_360176_1652587113/
    └── Images/              # 真实温室番茄图像 (100张)
        ├── IMG_0001.jpg
        ├── IMG_0002.jpg
        └── ...
```

> **注意**：项目已包含 100 张真实温室番茄图像，无需额外下载。

### 运行系统 / Run the System

```bash
# 处理全部图像（默认使用真实数据集）
python main.py

# 处理指定数量图像
python main.py --max-images 40

# 自定义参数
python main.py --data-dir data/DatasetId_360176_1652587113/Images --config config/algorithm_params.yaml --rate 1.0
```

### 运行测试 / Run Tests

```bash
# 运行所有测试
pytest

# 生成覆盖率报告
pytest --cov=src --cov-report=html
```

## 项目结构 / Project Structure

```
.
├── config/                     # 配置文件
│   └── algorithm_params.yaml   # 算法参数 (HSV阈值、形态学、轮廓过滤、优先级权重)
├── data/                       # 数据目录 (.gitignore)
│   ├── calibration/            # 相机标定参数
│   │   └── camera_params.yaml  #   内参 (fx=910, fy=910, cx=640, cy=360)
│   └── DatasetId_360176_1652587113/
│       ├── Annotations/        #   COCO 标注文件
│       └── Images/             #   真实温室番茄图像 (100张, 4032x3024)
├── docs/                       # 项目文档
│   └── superpowers/
│       ├── plans/              # 实现计划
│       └── specs/              # 设计文档
├── external_models/            # 外部模型 (YOLO等, .gitignore)
├── main.py                     # 主程序入口 (CLI)
├── README.md                   # 本文件
├── requirements.txt            # Python 依赖
├── results/                    # 结果输出
│   ├── analysis/               # 分析报告
│   │   ├── detection_grid.jpg  #   检测结果拼图 (9宫格)
│   │   ├── generate_report.py  #   分析脚本
│   │   └── report.json         #   统计报告 JSON
│   └── detection_output/       # 检测可视化结果 (每张图标注后输出, .gitignore)
├── src/                        # 源代码
│   ├── algorithms/             # 核心算法模块
│   │   ├── __init__.py
│   │   ├── color_segmentation.py  # HSV 色彩分割 (红/橙/绿三色番茄)
│   │   ├── contour_filter.py      # 轮廓筛选 + 成熟度评分
│   │   ├── coordinate_transform.py # 单目3D坐标计算
│   │   ├── morphological.py       # 形态学处理 (开/闭运算)
│   │   └── priority_scoring.py    # 多因素优先级评分
│   ├── mock_ros2/              # 模拟 ROS2 通信框架
│   │   ├── __init__.py
│   │   ├── core.py             #   Node, Topic, Publisher, Subscriber
│   │   └── message_types.py    #   Image, Detection, TomatoInfo, StatusMsg
│   └── nodes/                  # ROS2 功能节点
│       ├── __init__.py
│       ├── camera_node.py      #   图像采集 + 合成深度图
│       ├── recognition_node.py #   番茄检测 + 3D定位
│       ├── priority_node.py    #   优先级排序 + 目标选择
│       ├── visualization_node.py # 可视化标注输出
│       └── status_node.py      # 系统状态监控
├── tests/                      # 单元测试 (pytest)
│   ├── __init__.py
│   ├── test_color_segmentation.py
│   ├── test_contour_filter.py
│   ├── test_coordinate_transform.py
│   └── test_full_pipeline.py
└── 文档/                       # 课程相关文档
    ├── 智能温室番茄采摘机器人系统概要设计（2026 年 5 月）.pdf
    ├── 智能番茄采摘机器人_机械结构与形态设计_20260518.pdf
    ├── 处理器与硬件选型.docx
    ├── 成员1_v1.docx
    ├── 成员6_软件系统 + 算法设计_v1.docx
    └── 智能机器人传感器配置_4.docx
```

## 核心算法 / Core Algorithms

### 1. 图像预处理 / Image Preprocessing

```python
preprocess_image(image, target_size=None)
```

- **高斯模糊**：消除噪声 `(5x5 核)`
- **CLAHE 直方图均衡化**：增强对比度，适应光照变化 `(clipLimit=2.0, tileGridSize=(8,8))`

### 2. HSV 红色分割 / HSV Red Segmentation

```python
hsv_red_segmentation(image, lower1, upper1, lower2, upper2)
```

由于红色在 HSV 色域中跨越 0°/360° 边界，采用双区间阈值：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lower_red1` | `[0, 80, 50]` | 低饱和度红色下限 |
| `upper_red1` | `[10, 255, 255]` | 低饱和度红色上限 |
| `lower_red2` | `[160, 80, 50]` | 高饱和度红色下限 |
| `upper_red2` | `[180, 255, 255]` | 高饱和度红色上限 |

### 3. 形态学处理 / Morphological Operations

```python
apply_morphology(mask, open_kernel_size, close_kernel_size, iterations)
```

- **开运算**：消除小噪点 `(kernel=5x5)`
- **闭运算**：连接断裂区域 `(kernel=7x7)`

### 4. 轮廓过滤 / Contour Filtering

```python
filter_contours(image, contours, min_area, max_area, min_circularity, max_aspect_ratio)
```

| 指标 | 公式/方法 | 阈值 | 作用 |
|------|-----------|------|------|
| 面积 | `cv2.contourArea()` | 500~50000 | 过滤过小/过大区域 |
| 圆度 | `C = 4πA/P²` | ≥0.5 | 确保番茄形状 |
| 长宽比 | `max(w,h)/min(w,h)` | ≤2.0 | 排除细长物体 |
| 成熟度 | 红像素比×0.7 + 饱和度×0.3 | - | 评估成熟程度 |

### 5. 3D 坐标计算 / 3D Coordinate Transform

```python
calculate_3d_coordinates(detections, camera_params, real_diameter_mm)
```

基于单目视觉的已知尺寸深度估计：

```
Z = fx × real_diameter_mm / pixel_diameter
X = (u - cx) × Z / fx
Y = (v - cy) × Z / fy
```

### 6. 优先级评分 / Priority Scoring

```python
calculate_priority_score(detection, weights)
```

多因素加权评分模型：

| 因素 | 权重 | 说明 |
|------|------|------|
| 置信度 | 0.3 | 检测置信度 |
| 成熟度 | 0.3 | 番茄成熟程度 |
| 可采摘性 | 0.2 | 无障碍遮挡程度 |
| 距离 | 0.2 | 距离越近分数越高（归一化） |

## 配置参数 / Configuration Parameters

`config/algorithm_params.yaml`：

```yaml
hsv_thresholds:
  lower_red1: [0, 80, 50]
  upper_red1: [10, 255, 255]
  lower_red2: [160, 80, 50]
  upper_red2: [180, 255, 255]

morphology:
  open_kernel_size: 5
  close_kernel_size: 7
  iterations: 2

contour_filter:
  min_area: 500
  max_area: 50000
  min_circularity: 0.5
  max_aspect_ratio: 2.0

priority_scoring:
  weights:
    confidence: 0.3
    maturity: 0.3
    accessibility: 0.2
    distance: 0.2

depth_estimation:
  real_diameter_mm: 80  # 番茄平均直径
```

## 预期输出 / Expected Outputs

### 检测可视化 / Detection Visualization

运行后 `results/detection_output/` 目录下生成带标注的图像：

- **绿色框**：当前最优采摘目标
- **红色框**：其他检测到的番茄
- **标签**：`ID:{id} C:{confidence} M:{maturity}`
- **3D坐标**：`(X, Y, Z) mm`
- **状态栏**：系统运行状态

### 分析报表 / Analysis Report

```bash
python results/analysis/generate_report.py
```

生成：
- `results/analysis/detection_grid.jpg`：9张典型检测结果拼图
- `results/analysis/report.json`：统计报告

报告包含：
- 总处理图像数
- 总检测番茄数
- 平均每图检测数
- 平均置信度
- 平均成熟度
- 按场景分类统计

## 命令行参数 / CLI Arguments

```
python main.py [OPTIONS]

Options:
  --data-dir PATH      测试图像目录 (default: data/DatasetId_360176_1652587113/Images)
  --config PATH        算法参数文件 (default: config/algorithm_params.yaml)
  --camera-params PATH 相机参数文件 (default: data/calibration/camera_params.yaml)
  --rate FLOAT         发布频率 Hz (default: 1.0)
  --max-images INT     最大处理图像数 (default: 全部)
```

## 开发者信息 / Developer Info

- **课程**：智能机器人
- **学校**：南京理工大学 / NJUST
- **日期**：2026年5月

## 许可证 / License

本项目仅用于课程教学与学术交流。

---

*最后更新：2026-05-23*
