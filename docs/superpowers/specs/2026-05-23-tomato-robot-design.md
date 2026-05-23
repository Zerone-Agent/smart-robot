# 智能温室番茄采摘机器人 - 软件模拟系统设计文档

**日期**: 2026年5月23日  
**版本**: v1.0  
**负责模块**: 软件系统 + 算法设计（编码实现）  
**技术栈**: Python + OpenCV + 伪ROS2架构（纯软件模拟）

---

## 1. 项目背景与目标

### 1.1 背景
智能温室番茄采摘机器人课程大作业，团队负责软件系统与核心算法实现。由于没有硬件条件，采用软件模拟方式完成。

### 1.2 目标
- 实现成熟番茄识别与三维定位核心算法
- 采用伪ROS2架构演示模块化设计
- 通过软件模拟验证算法可行性
- 输出实验结果用于设计报告

---

## 2. 系统架构（伪ROS2模块化设计）

```
┌─────────────────────────────────────────────────────────────────┐
│                        应用层 (Application)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ 任务管理节点  │  │ 可视化节点    │  │ 状态监测与日志节点    │  │
│  │ (TaskNode)   │  │ (VizNode)    │  │ (StatusNode)         │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────────────┘  │
└─────────┼─────────────────┼───────────────────────────────────────┘
          │                 │
          ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                        决策层 (Decision)                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │          番茄识别与定位节点 (RecognitionNode)              │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐ │   │
│  │  │ 图像采集  │ │ HSV分割   │ │ 形态学处理│ │ 三维坐标计算  │ │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────┐  ┌──────────────────────────────┐ │
│  │   目标优先级排序节点      │  │     避障决策节点 (简化)        │ │
│  │   (PriorityNode)         │  │     (ObstacleNode)           │ │
│  └──────────────────────────┘  └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        数据层 (Data/模拟)                        │
│         模拟图像数据集 / 合成温室场景 / 预设深度信息                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心模块设计

### 3.1 Mock ROS2 核心框架

```python
# mock_ros2/core.py
class MockNode:
    """模拟 ROS2 节点的基类"""
    def __init__(self, node_name): ...
    def publish(self, topic, msg): ...
    def subscribe(self, topic, callback): ...
    def spin(self): ...

class Topic:
    """模拟 ROS2 Topic"""
    def __init__(self, name, msg_type): ...
    def publish(self, msg): ...
```

### 3.2 番茄识别节点 (RecognitionNode)

**输入**: `/camera/image_raw` (模拟)

**处理流程**:
1. 图像预处理（缩放、去噪、亮度均衡）
2. HSV 颜色空间转换
3. 红色区域阈值分割（双区间：H=0-10 和 H=160-180）
4. 形态学处理（开运算去噪、闭运算补洞）
5. 轮廓提取与圆形度筛选
6. 中心点计算与外接矩形
7. 三维坐标计算（模拟深度）

**输出**: `TomatoDetectionArray`

```python
class TomatoDetection:
    tomato_id: int
    bbox: [x, y, w, h]          # 像素坐标
    center_uv: [u, v]           # 图像中心点
    center_3d: [X, Y, Z]        # 相机坐标系（mm）
    circularity: float          # 圆形度 0-1
    maturity_score: float       # 成熟度评分
```

### 3.3 目标优先级排序节点 (PriorityNode)

**评分公式**:
```
Score = w1×Conf + w2×Maturity + w3×Accessibility + w4×Distance
```

### 3.4 数据流设计

| Topic 名称 | 发布者 | 订阅者 | 数据类型 | 说明 |
|------------|--------|--------|----------|------|
| `/camera/image_raw` | MockCamera | RecognitionNode | Image | 模拟 RGB 图像 |
| `/camera/depth` | MockCamera | RecognitionNode | DepthImage | 模拟深度图 |
| `/detection/tomatoes` | RecognitionNode | PriorityNode, VizNode | TomatoDetectionArray | 检测结果 |
| `/planning/target` | PriorityNode | VizNode | TomatoTarget | 当前采摘目标 |
| `/system/status` | StatusNode | VizNode | SystemStatus | 系统状态日志 |

---

## 4. 项目文件结构

```
smart-robot/
├── docs/
│   └── design.md                    # 本设计文档
├── src/
│   ├── mock_ros2/
│   │   ├── __init__.py
│   │   ├── core.py                  # MockNode, Topic, Publisher, Subscriber
│   │   └── message_types.py         # Image, Detection, 等消息定义
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── camera_node.py           # 模拟相机节点（读取数据集图片）
│   │   ├── recognition_node.py      # 番茄识别核心节点
│   │   ├── priority_node.py         # 优先级排序节点
│   │   ├── visualization_node.py    # 可视化节点（OpenCV显示）
│   │   └── status_node.py           # 状态监测节点
│   └── algorithms/
│       ├── __init__.py
│       ├── color_segmentation.py    # HSV颜色分割
│       ├── morphological.py         # 形态学处理
│       ├── contour_filter.py        # 轮廓筛选（圆形度、面积）
│       ├── coordinate_transform.py  # 三维坐标计算
│       └── priority_scoring.py      # 优先级评分算法
├── data/
│   ├── test_images/                 # 测试图片集
│   │   ├── single_tomato/         # 单果实场景
│   │   ├── multiple_tomatoes/     # 多果实场景
│   │   └── challenging/             # 复杂场景（遮挡、光照变化）
│   └── calibration/
│       └── camera_params.yaml       # 模拟相机内参
├── config/
│   └── algorithm_params.yaml        # 算法参数（HSV阈值等）
├── tests/
│   ├── test_color_segmentation.py
│   ├── test_contour_filter.py
│   └── test_full_pipeline.py
├── results/
│   ├── detection_output/            # 检测结果图
│   ├── logs/                        # 运行日志
│   └── analysis/                    # 数据分析图表
├── main.py                          # 主程序入口
├── requirements.txt
└── README.md
```

---

## 5. 核心算法详细设计

### 5.1 HSV 颜色分割参数

```python
# 低红色区间
hsv_lower1 = (0, 80, 50)
hsv_upper1 = (10, 255, 255)

# 高红色区间
hsv_lower2 = (160, 80, 50)
hsv_upper2 = (180, 255, 255)
```

### 5.2 圆形度筛选公式

```
C = 4πA / P²
```

其中：
- A = 轮廓面积
- P = 轮廓周长
- C 越接近 1 表示轮廓越接近圆形

### 5.3 三维坐标计算（相机坐标系）

```
X = (u - cx) × Z / fx
Y = (v - cy) × Z / fy
Z = fx × real_diameter / pixel_diameter  # 模拟深度估算
```

### 5.4 成熟度评分

```
Maturity = (red_pixels / total_pixels) × 0.7 + (avg_saturation / 255) × 0.3
```

---

## 6. 测试与验证方案

### 6.1 测试数据集

| 测试场景 | 图片数量 | 测试目的 |
|----------|----------|----------|
| 单果实正面 | 10 | 基础识别功能验证 |
| 单果实侧面/遮挡 | 10 | 非完整形态识别 |
| 多果实聚集 | 10 | 重叠情况下的分割能力 |
| 不同光照条件 | 10 | 算法鲁棒性 |
| 背景干扰（红标签等） | 5 | 误检率测试 |
| **总计** | **45** | |

### 6.2 评估指标

- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × Precision × Recall / (Precision + Recall)
- **Center Error**: 平均中心点定位误差（mm）
- **Processing Time**: 单帧处理时间（ms）

### 6.3 实验输出

| 输出项 | 格式 | 用途 |
|--------|------|------|
| 检测结果图 | JPG | 实验报告插图 |
| 识别统计表 | CSV/Excel | 数据分析 |
| 算法参数配置 | YAML | 可复现性 |
| 运行日志 | TXT/JSON | 调试与问题追踪 |
| 演示视频/GIF | MP4/GIF | 展示用 |

---

## 7. 硬件要求评估

**目标平台**: MacBook Air M5

| 组件 | 要求 | M5 表现 |
|------|------|---------|
| Python + OpenCV | 基础 | ✅ 完美支持 |
| 实时图像处理（640×480） | <100ms/帧 | ✅ 轻松达到 30+ FPS |
| 内存占用 | <2GB | ✅ 8GB 内存充足 |
| 存储 | <1GB（代码+数据集） | ✅ 充足 |

---

## 8. 交付物清单

1. [ ] 完整可运行的 Python 代码
2. [ ] 测试数据集（或数据获取脚本）
3. [ ] 实验结果（检测图片、统计数据）
4. [ ] 算法参数配置文件
5. [ ] 运行说明文档（README）

---

## 9. 设计确认

- [x] 伪 ROS2 架构方案已确认
- [x] 番茄识别算法流程已确认
- [x] 项目文件结构已确认
- [x] 测试验证方案已确认
- [x] 硬件可行性已评估

**下一步**: 编写实现计划 (writing-plans)
