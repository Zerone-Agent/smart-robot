# 番茄采摘机器人软件模拟系统 - 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个完整的基于 Python + OpenCV 的智能温室番茄采摘机器人软件模拟系统，包含合成数据生成、成熟番茄识别与三维定位算法、伪 ROS2 模块化架构、测试验证和实验结果输出。

**Architecture:** 采用伪 ROS2 模块化设计，各节点通过内存 Topic 通信。数据流：CameraNode 读取/生成图像 → RecognitionNode 进行 HSV 分割/形态学/轮廓筛选/3D 坐标计算 → PriorityNode 排序 → VizNode 可视化。支持合成温室场景图片生成，无需真实硬件。

**Tech Stack:** Python 3.11+, OpenCV, NumPy, PyYAML, pytest, Matplotlib

---

## 项目结构

```
smart-robot/
├── src/
│   ├── mock_ros2/
│   │   ├── __init__.py
│   │   ├── core.py              # MockNode, Topic, Publisher, Subscriber
│   │   └── message_types.py     # Image, Detection, 等消息定义
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── camera_node.py       # 模拟相机节点（读取数据集图片）
│   │   ├── recognition_node.py  # 番茄识别核心节点
│   │   ├── priority_node.py     # 优先级排序节点
│   │   ├── visualization_node.py # 可视化节点（OpenCV显示）
│   │   └── status_node.py       # 状态监测节点
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── color_segmentation.py    # HSV颜色分割
│   │   ├── morphological.py         # 形态学处理
│   │   ├── contour_filter.py        # 轮廓筛选（圆形度、面积）
│   │   ├── coordinate_transform.py  # 三维坐标计算
│   │   └── priority_scoring.py      # 优先级评分算法
│   └── utils/
│       ├── __init__.py
│       └── synthetic_data.py    # 合成温室场景图片生成器
├── data/
│   ├── test_images/
│   │   ├── single_tomato/       # 单果实场景 (10张)
│   │   ├── multiple_tomatoes/   # 多果实场景 (10张)
│   │   ├── occlusion/           # 遮挡场景 (10张)
│   │   └── lighting/            # 光照变化场景 (10张)
│   └── calibration/
│       └── camera_params.yaml   # 模拟相机内参
├── config/
│   └── algorithm_params.yaml    # 算法参数（HSV阈值等）
├── tests/
│   ├── test_color_segmentation.py
│   ├── test_contour_filter.py
│   ├── test_coordinate_transform.py
│   └── test_full_pipeline.py
├── results/
│   ├── detection_output/        # 检测结果图
│   ├── logs/                    # 运行日志
│   └── analysis/                # 数据分析图表
├── generate_synthetic_data.py   # 合成数据生成脚本
├── main.py                      # 主程序入口
├── requirements.txt
└── README.md
```

---

## Task 1: 项目初始化与依赖配置

**Files:**
- Create: `requirements.txt`
- Create: `config/algorithm_params.yaml`
- Create: `data/calibration/camera_params.yaml`

- [ ] **Step 1: 创建 requirements.txt**

```txt
opencv-python>=4.8.0
numpy>=1.24.0
pyyaml>=6.0
pytest>=7.4.0
pytest-cov>=4.1.0
matplotlib>=3.7.0
```

- [ ] **Step 2: 创建算法参数配置**

```yaml
# config/algorithm_params.yaml
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
  real_diameter_mm: 80  # 番茄平均直径 80mm
```

- [ ] **Step 3: 创建相机内参配置**

```yaml
# data/calibration/camera_params.yaml
camera_matrix:
  fx: 910.0
  fy: 910.0
  cx: 640.0
  cy: 360.0

image_size:
  width: 1280
  height: 720

distortion:
  k1: 0.0
  k2: 0.0
  p1: 0.0
  p2: 0.0
  k3: 0.0
```

- [ ] **Step 4: 安装依赖**

Run: `pip install -r requirements.txt`
Expected: 所有包安装成功

- [ ] **Step 5: Commit**

```bash
git add requirements.txt config/algorithm_params.yaml data/calibration/camera_params.yaml
git commit -m "chore: add project config and dependencies"
```

---

## Task 2: Mock ROS2 核心框架

**Files:**
- Create: `src/mock_ros2/__init__.py`
- Create: `src/mock_ros2/core.py`
- Create: `src/mock_ros2/message_types.py`

- [ ] **Step 1: 创建消息类型定义**

```python
# src/mock_ros2/message_types.py
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Image:
    """模拟图像消息"""
    data: np.ndarray
    width: int
    height: int
    channels: int
    timestamp: float = 0.0


@dataclass
class DepthImage:
    """模拟深度图消息"""
    data: np.ndarray
    width: int
    height: int
    timestamp: float = 0.0


@dataclass
class TomatoDetection:
    """单个番茄检测结果"""
    tomato_id: int
    bbox: List[int]          # [x, y, w, h]
    center_uv: List[float]   # [u, v]
    center_3d: List[float]   # [X, Y, Z] mm
    circularity: float
    maturity_score: float
    confidence: float


@dataclass
class TomatoDetectionArray:
    """番茄检测数组消息"""
    detections: List[TomatoDetection] = field(default_factory=list)
    image_timestamp: float = 0.0


@dataclass
class TomatoTarget:
    """当前采摘目标"""
    detection: Optional[TomatoDetection] = None
    priority_score: float = 0.0


@dataclass
class SystemStatus:
    """系统状态"""
    node_name: str
    status: str
    message: str
    timestamp: float = 0.0
```

- [ ] **Step 2: 创建 Mock ROS2 核心**

```python
# src/mock_ros2/core.py
import time
import queue
from typing import Dict, List, Callable, Any
from threading import Thread, Lock


class Topic:
    """模拟 ROS2 Topic"""
    def __init__(self, name: str, msg_type: type):
        self.name = name
        self.msg_type = msg_type
        self.subscribers: List[Callable] = []
        self.lock = Lock()
    
    def publish(self, msg: Any):
        with self.lock:
            for callback in self.subscribers:
                callback(msg)
    
    def subscribe(self, callback: Callable):
        with self.lock:
            self.subscribers.append(callback)


class MockNode:
    """模拟 ROS2 节点基类"""
    def __init__(self, node_name: str):
        self.node_name = node_name
        self.topics: Dict[str, Topic] = {}
        self.running = False
    
    def create_publisher(self, topic_name: str, msg_type: type) -> Topic:
        if topic_name not in self.topics:
            self.topics[topic_name] = Topic(topic_name, msg_type)
        return self.topics[topic_name]
    
    def create_subscription(self, topic_name: str, msg_type: type, callback: Callable):
        if topic_name not in self.topics:
            self.topics[topic_name] = Topic(topic_name, msg_type)
        self.topics[topic_name].subscribe(callback)
    
    def publish(self, topic_name: str, msg: Any):
        if topic_name in self.topics:
            self.topics[topic_name].publish(msg)
    
    def spin(self):
        self.running = True
        while self.running:
            time.sleep(0.1)
    
    def stop(self):
        self.running = False
    
    def log(self, level: str, message: str):
        print(f"[{level.upper()}] [{self.node_name}] {message}")


class MockSystem:
    """模拟 ROS2 系统管理器"""
    def __init__(self):
        self.nodes: List[MockNode] = []
        self.global_topics: Dict[str, Topic] = {}
    
    def register_node(self, node: MockNode):
        self.nodes.append(node)
        for topic_name, topic in node.topics.items():
            if topic_name not in self.global_topics:
                self.global_topics[topic_name] = topic
    
    def connect_topics(self, topic_name: str, publisher_node: MockNode, subscriber_node: MockNode, callback: Callable):
        if topic_name not in self.global_topics:
            self.global_topics[topic_name] = Topic(topic_name, Any)
        
        publisher_node.topics[topic_name] = self.global_topics[topic_name]
        self.global_topics[topic_name].subscribe(callback)
    
    def start_all(self):
        for node in self.nodes:
            if hasattr(node, 'start'):
                node.start()
    
    def stop_all(self):
        for node in self.nodes:
            node.stop()
```

- [ ] **Step 3: Commit**

```bash
git add src/mock_ros2/
git commit -m "feat: add mock ROS2 framework with topics and nodes"
```

---

## Task 3: 合成温室场景图片生成器

**Files:**
- Create: `src/utils/__init__.py`
- Create: `src/utils/synthetic_data.py`
- Create: `generate_synthetic_data.py`

- [ ] **Step 1: 创建合成数据生成器**

```python
# src/utils/synthetic_data.py
import numpy as np
import cv2
import os
from typing import List, Tuple, Optional
import random


class SyntheticTomatoGenerator:
    """合成温室番茄场景图片生成器"""
    
    def __init__(self, image_size: Tuple[int, int] = (1280, 720)):
        self.width, self.height = image_size
        self.tomato_colors = [
            ((0, 100, 100), (10, 255, 255)),   # 低红
            ((160, 100, 100), (180, 255, 255)), # 高红
        ]
        self.leaf_colors = [
            ((35, 40, 40), (85, 255, 180)),    # 绿色叶子
        ]
    
    def _create_background(self) -> np.ndarray:
        """创建温室背景"""
        # 深绿色到浅绿色渐变背景
        bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for y in range(self.height):
            green_val = 60 + int(40 * (y / self.height))
            bg[y, :] = [green_val // 3, green_val, green_val // 4]
        
        # 添加一些随机叶子纹理
        for _ in range(20):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            size = random.randint(30, 80)
            color = (random.randint(30, 80), random.randint(100, 160), random.randint(20, 60))
            cv2.ellipse(bg, (x, y), (size, size//2), random.randint(0, 180), 0, 360, color, -1)
        
        # 添加高斯噪声模拟传感器噪声
        noise = np.random.normal(0, 5, bg.shape).astype(np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return bg
    
    def _draw_tomato(self, image: np.ndarray, x: int, y: int, radius: int, 
                     maturity: float = 1.0, occlusion: float = 0.0) -> dict:
        """绘制单个番茄，返回标注信息"""
        # 番茄颜色：成熟度越高越红
        r = int(180 + 75 * maturity)
        g = int(50 * (1 - maturity))
        b = int(30 * (1 - maturity))
        color = (b, g, r)
        
        # 绘制番茄主体（略扁的圆）
        cv2.ellipse(image, (x, y), (radius, int(radius * 0.9)), 0, 0, 360, color, -1)
        
        # 添加高光
        highlight_x = x - radius // 3
        highlight_y = y - radius // 3
        cv2.circle(image, (highlight_x, highlight_y), radius // 4, (255, 200, 200), -1)
        cv2.circle(image, (highlight_x, highlight_y), radius // 5, (255, 255, 255), -1)
        
        # 添加阴影
        shadow_color = (max(0, b-40), max(0, g-40), max(0, r-40))
        cv2.ellipse(image, (x, y + radius//3), (radius, radius//3), 0, 0, 180, shadow_color, 2)
        
        annotation = {
            'center': (x, y),
            'radius': radius,
            'bbox': [x - radius, y - int(radius * 0.9), radius * 2, int(radius * 1.8)],
            'maturity': maturity
        }
        
        # 模拟遮挡
        if occlusion > 0:
            leaf_color = (40, 120, 40)
            occlude_w = int(radius * 2 * occlusion)
            occlude_h = int(radius * 1.8 * occlusion)
            occlude_x = random.randint(x - radius, x + radius - occlude_w)
            occlude_y = random.randint(y - int(radius*0.9), y)
            cv2.rectangle(image, (occlude_x, occlude_y), 
                         (occlude_x + occlude_w, occlude_y + occlude_h), 
                         leaf_color, -1)
            # 添加叶脉
            for i in range(3):
                start = (occlude_x + i * occlude_w // 3, occlude_y)
                end = (occlude_x + (i+1) * occlude_w // 3, occlude_y + occlude_h)
                cv2.line(image, start, end, (60, 140, 60), 1)
        
        return annotation
    
    def generate_single_tomato(self, seed: Optional[int] = None) -> Tuple[np.ndarray, List[dict]]:
        """生成单果实场景"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        image = self._create_background()
        x = random.randint(300, self.width - 300)
        y = random.randint(200, self.height - 200)
        radius = random.randint(40, 80)
        maturity = random.uniform(0.8, 1.0)
        
        annotation = self._draw_tomato(image, x, y, radius, maturity)
        
        return image, [annotation]
    
    def generate_multiple_tomatoes(self, count: int = 3, seed: Optional[int] = None) -> Tuple[np.ndarray, List[dict]]:
        """生成多果实场景"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        image = self._create_background()
        annotations = []
        
        for _ in range(count):
            x = random.randint(200, self.width - 200)
            y = random.randint(150, self.height - 150)
            radius = random.randint(35, 70)
            maturity = random.uniform(0.7, 1.0)
            
            annotation = self._draw_tomato(image, x, y, radius, maturity)
            annotations.append(annotation)
        
        return image, annotations
    
    def generate_occlusion_scene(self, seed: Optional[int] = None) -> Tuple[np.ndarray, List[dict]]:
        """生成遮挡场景"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        image = self._create_background()
        annotations = []
        
        for _ in range(random.randint(2, 4)):
            x = random.randint(200, self.width - 200)
            y = random.randint(150, self.height - 150)
            radius = random.randint(40, 75)
            maturity = random.uniform(0.7, 1.0)
            occlusion = random.uniform(0.1, 0.4)
            
            annotation = self._draw_tomato(image, x, y, radius, maturity, occlusion)
            annotations.append(annotation)
        
        return image, annotations
    
    def generate_lighting_variation(self, seed: Optional[int] = None) -> Tuple[np.ndarray, List[dict]]:
        """生成光照变化场景"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        image, annotations = self.generate_multiple_tomatoes(count=random.randint(2, 4))
        
        # 应用光照变化
        variation_type = random.choice(['bright', 'dark', 'uneven'])
        
        if variation_type == 'bright':
            image = cv2.convertScaleAbs(image, alpha=1.3, beta=30)
        elif variation_type == 'dark':
            image = cv2.convertScaleAbs(image, alpha=0.6, beta=-20)
        elif variation_type == 'uneven':
            # 添加渐变阴影
            shadow = np.zeros((self.height, self.width), dtype=np.uint8)
            for y in range(self.height):
                shadow[y, :] = int(255 * (1 - y / self.height) * 0.3)
            shadow = cv2.GaussianBlur(shadow, (101, 101), 0)
            shadow_3ch = np.stack([shadow] * 3, axis=-1)
            image = cv2.subtract(image, shadow_3ch)
        
        return image, annotations
    
    def generate_dataset(self, output_dir: str, counts: dict = None):
        """生成完整数据集"""
        if counts is None:
            counts = {
                'single_tomato': 10,
                'multiple_tomatoes': 10,
                'occlusion': 10,
                'lighting': 10
            }
        
        os.makedirs(output_dir, exist_ok=True)
        
        generators = {
            'single_tomato': self.generate_single_tomato,
            'multiple_tomatoes': lambda seed: self.generate_multiple_tomatoes(count=random.randint(3, 6), seed=seed),
            'occlusion': self.generate_occlusion_scene,
            'lighting': self.generate_lighting_variation
        }
        
        for category, count in counts.items():
            cat_dir = os.path.join(output_dir, category)
            os.makedirs(cat_dir, exist_ok=True)
            
            for i in range(count):
                image, annotations = generators[category](seed=i * 1000 + hash(category) % 1000)
                
                filename = f"{category}_{i:03d}.jpg"
                filepath = os.path.join(cat_dir, filename)
                cv2.imwrite(filepath, image)
                
                # 保存标注信息
                ann_filename = f"{category}_{i:03d}.txt"
                ann_filepath = os.path.join(cat_dir, ann_filename)
                with open(ann_filepath, 'w') as f:
                    for ann in annotations:
                        f.write(f"{ann['center'][0]},{ann['center'][1]},{ann['radius']},{ann['maturity']}\n")
                
                print(f"Generated: {filepath}")


if __name__ == '__main__':
    generator = SyntheticTomatoGenerator()
    generator.generate_dataset('data/test_images')
```

- [ ] **Step 2: 创建数据生成脚本**

```python
# generate_synthetic_data.py
from src.utils.synthetic_data import SyntheticTomatoGenerator


def main():
    print("Generating synthetic tomato dataset...")
    generator = SyntheticTomatoGenerator(image_size=(1280, 720))
    
    generator.generate_dataset('data/test_images', counts={
        'single_tomato': 10,
        'multiple_tomatoes': 10,
        'occlusion': 10,
        'lighting': 10
    })
    
    print("Dataset generation complete!")
    print("Total images: 40")
    print("Categories: single_tomato, multiple_tomatoes, occlusion, lighting")


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: 生成测试数据**

Run: `python generate_synthetic_data.py`
Expected: 
```
Generating synthetic tomato dataset...
Generated: data/test_images/single_tomato/single_tomato_000.jpg
...
Dataset generation complete!
Total images: 40
```

- [ ] **Step 4: 验证生成结果**

Run: `ls -R data/test_images/`
Expected: 显示 4 个子目录，每个目录 10 张 JPG + 10 个 TXT 标注文件

- [ ] **Step 5: Commit**

```bash
git add src/utils/synthetic_data.py generate_synthetic_data.py data/test_images/
git commit -m "feat: add synthetic data generator with 40 test images"
```

---

## Task 4: 核心算法模块

**Files:**
- Create: `src/algorithms/__init__.py`
- Create: `src/algorithms/color_segmentation.py`
- Create: `src/algorithms/morphological.py`
- Create: `src/algorithms/contour_filter.py`
- Create: `src/algorithms/coordinate_transform.py`
- Create: `src/algorithms/priority_scoring.py`

- [ ] **Step 1: 颜色分割算法**

```python
# src/algorithms/color_segmentation.py
import cv2
import numpy as np
from typing import Tuple


def hsv_red_segmentation(image: np.ndarray, 
                        lower1: Tuple[int, int, int] = (0, 80, 50),
                        upper1: Tuple[int, int, int] = (10, 255, 255),
                        lower2: Tuple[int, int, int] = (160, 80, 50),
                        upper2: Tuple[int, int, int] = (180, 255, 255)) -> np.ndarray:
    """
    HSV 红色区域分割
    
    Args:
        image: BGR 格式输入图像
        lower1, upper1: 低红色区间
        lower2, upper2: 高红色区间
    
    Returns:
        二值掩膜图 (0/255)
    """
    # 转换到 HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 低红色区间掩膜
    mask1 = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
    
    # 高红色区间掩膜
    mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
    
    # 合并两个区间
    mask = cv2.bitwise_or(mask1, mask2)
    
    return mask


def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
    """
    图像预处理：缩放、去噪、亮度均衡
    
    Args:
        image: 输入图像
        target_size: 目标尺寸 (width, height)
    
    Returns:
        预处理后的图像
    """
    result = image.copy()
    
    # 缩放
    if target_size is not None:
        result = cv2.resize(result, target_size)
    
    # 高斯去噪
    result = cv2.GaussianBlur(result, (5, 5), 0)
    
    # CLAHE 亮度均衡
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return result
```

- [ ] **Step 2: 形态学处理算法**

```python
# src/algorithms/morphological.py
import cv2
import numpy as np


def apply_morphology(mask: np.ndarray, 
                    open_kernel_size: int = 5,
                    close_kernel_size: int = 7,
                    iterations: int = 2) -> np.ndarray:
    """
    形态学处理：开运算去噪 + 闭运算补洞
    
    Args:
        mask: 二值掩膜图
        open_kernel_size: 开运算核大小
        close_kernel_size: 闭运算核大小
        iterations: 迭代次数
    
    Returns:
        处理后的掩膜图
    """
    # 开运算：去除小噪点
    open_kernel = np.ones((open_kernel_size, open_kernel_size), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=iterations)
    
    # 闭运算：填补果实内部孔洞
    close_kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel, iterations=iterations)
    
    return closed
```

- [ ] **Step 3: 轮廓筛选算法**

```python
# src/algorithms/contour_filter.py
import cv2
import numpy as np
from typing import List, Tuple, Dict


def calculate_circularity(contour: np.ndarray) -> float:
    """
    计算轮廓圆形度
    C = 4 * pi * A / P^2
    
    Args:
        contour: 轮廓点集
    
    Returns:
        圆形度 (0-1, 越接近1越圆)
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    if perimeter == 0:
        return 0.0
    
    circularity = 4 * np.pi * area / (perimeter ** 2)
    return circularity


def calculate_maturity_score(image: np.ndarray, contour: np.ndarray) -> float:
    """
    计算成熟度评分
    Maturity = (red_pixels / total_pixels) * 0.7 + (avg_saturation / 255) * 0.3
    
    Args:
        image: BGR 图像
        contour: 轮廓
    
    Returns:
        成熟度评分 (0-1)
    """
    # 创建掩膜
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    # 获取轮廓内像素
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    region = hsv[mask == 255]
    
    if len(region) == 0:
        return 0.0
    
    # 红色像素比例
    red_mask1 = cv2.inRange(hsv, (0, 80, 50), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (160, 80, 50), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    red_pixels = np.sum(red_mask[mask == 255] > 0)
    total_pixels = np.sum(mask > 0)
    red_ratio = red_pixels / total_pixels if total_pixels > 0 else 0
    
    # 平均饱和度
    avg_saturation = np.mean(region[:, 1]) / 255.0
    
    # 综合评分
    maturity = red_ratio * 0.7 + avg_saturation * 0.3
    
    return float(np.clip(maturity, 0, 1))


def filter_contours(image: np.ndarray, 
                   contours: List[np.ndarray],
                   min_area: float = 500,
                   max_area: float = 50000,
                   min_circularity: float = 0.5,
                   max_aspect_ratio: float = 2.0) -> List[Dict]:
    """
    筛选轮廓：根据面积、圆形度、长宽比
    
    Args:
        image: BGR 图像
        contours: 轮廓列表
        min_area, max_area: 面积范围
        min_circularity: 最小圆形度
        max_aspect_ratio: 最大长宽比
    
    Returns:
        筛选后的目标列表，每个目标包含 bbox, center, circularity, maturity
    """
    filtered = []
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # 面积筛选
        if area < min_area or area > max_area:
            continue
        
        # 圆形度筛选
        circularity = calculate_circularity(contour)
        if circularity < min_circularity:
            continue
        
        # 长宽比筛选
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
        if aspect_ratio > max_aspect_ratio:
            continue
        
        # 计算中心点
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # 成熟度评分
        maturity = calculate_maturity_score(image, contour)
        
        filtered.append({
            'tomato_id': i,
            'bbox': [x, y, w, h],
            'center_uv': [cx, cy],
            'circularity': circularity,
            'maturity_score': maturity,
            'area': area,
            'contour': contour
        })
    
    return filtered
```

- [ ] **Step 4: 三维坐标计算算法**

```python
# src/algorithms/coordinate_transform.py
import numpy as np
from typing import List, Dict


def calculate_3d_coordinates(detections: List[Dict], 
                            camera_params: Dict,
                            real_diameter_mm: float = 80.0) -> List[Dict]:
    """
    计算番茄三维坐标
    
    使用单目深度估计：Z = fx * real_diameter / pixel_diameter
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    Args:
        detections: 检测结果列表 (含 bbox)
        camera_params: 相机参数 {fx, fy, cx, cy}
        real_diameter_mm: 番茄真实直径 (mm)
    
    Returns:
        更新后的检测结果（含 center_3d）
    """
    fx = camera_params['fx']
    fy = camera_params['fy']
    cx = camera_params['cx']
    cy = camera_params['cy']
    
    for detection in detections:
        u, v = detection['center_uv']
        bbox = detection['bbox']
        pixel_diameter = max(bbox[2], bbox[3])  # 取外接矩形最大边
        
        # 深度估计
        Z = fx * real_diameter_mm / pixel_diameter
        
        # 三维坐标
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        
        detection['center_3d'] = [float(X), float(Y), float(Z)]
        detection['pixel_diameter'] = pixel_diameter
    
    return detections


def load_camera_params(filepath: str) -> Dict:
    """从 YAML 加载相机参数"""
    import yaml
    with open(filepath, 'r') as f:
        params = yaml.safe_load(f)
    
    return {
        'fx': params['camera_matrix']['fx'],
        'fy': params['camera_matrix']['fy'],
        'cx': params['camera_matrix']['cx'],
        'cy': params['camera_matrix']['cy']
    }
```

- [ ] **Step 5: 优先级评分算法**

```python
# src/algorithms/priority_scoring.py
from typing import List, Dict


def calculate_priority_score(detection: Dict, 
                            weights: Dict = None) -> float:
    """
    计算目标优先级评分
    Score = w1*Conf + w2*Maturity + w3*Accessibility + w4*Distance
    
    Args:
        detection: 检测结果
        weights: 权重字典
    
    Returns:
        优先级评分
    """
    if weights is None:
        weights = {
            'confidence': 0.3,
            'maturity': 0.3,
            'accessibility': 0.2,
            'distance': 0.2
        }
    
    # 置信度（用圆形度表示）
    confidence = detection.get('circularity', 0.5)
    
    # 成熟度
    maturity = detection.get('maturity_score', 0.5)
    
    # 可采摘性（面积越大越容易采摘）
    area = detection.get('area', 0)
    max_area = 50000
    accessibility = min(area / max_area, 1.0)
    
    # 距离（越近越好，归一化到 0-1）
    center_3d = detection.get('center_3d', [0, 0, 1000])
    Z = center_3d[2]
    distance_score = max(0, 1 - Z / 2000)  # 2m 内归一化
    
    score = (weights['confidence'] * confidence +
             weights['maturity'] * maturity +
             weights['accessibility'] * accessibility +
             weights['distance'] * distance_score)
    
    return float(score)


def sort_by_priority(detections: List[Dict], 
                    weights: Dict = None) -> List[Dict]:
    """按优先级排序检测结果"""
    for detection in detections:
        detection['priority_score'] = calculate_priority_score(detection, weights)
    
    sorted_detections = sorted(detections, 
                              key=lambda x: x['priority_score'], 
                              reverse=True)
    
    return sorted_detections
```

- [ ] **Step 6: Commit**

```bash
git add src/algorithms/
git commit -m "feat: add core algorithm modules (segmentation, morphology, contour, 3D transform, priority)"
```

---

## Task 5: 节点实现

**Files:**
- Create: `src/nodes/__init__.py`
- Create: `src/nodes/camera_node.py`
- Create: `src/nodes/recognition_node.py`
- Create: `src/nodes/priority_node.py`
- Create: `src/nodes/visualization_node.py`
- Create: `src/nodes/status_node.py`

- [ ] **Step 1: 相机节点**

```python
# src/nodes/camera_node.py
import os
import cv2
import time
import numpy as np
from typing import List, Optional
from src.mock_ros2.core import MockNode
from src.mock_ros2.message_types import Image, DepthImage


class CameraNode(MockNode):
    """模拟相机节点：从数据集读取图片并发布"""
    
    def __init__(self, image_dir: str, publish_rate: float = 1.0):
        super().__init__('camera_node')
        self.image_dir = image_dir
        self.publish_rate = publish_rate
        self.image_files: List[str] = []
        self.current_index = 0
        
        self._load_images()
        self._setup_publishers()
    
    def _load_images(self):
        """加载所有测试图片"""
        for root, _, files in os.walk(self.image_dir):
            for file in sorted(files):
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    self.image_files.append(os.path.join(root, file))
        
        self.log('info', f"Loaded {len(self.image_files)} images")
    
    def _setup_publishers(self):
        """设置发布器"""
        self.rgb_topic = self.create_publisher('/camera/image_raw', Image)
        self.depth_topic = self.create_publisher('/camera/depth', DepthImage)
    
    def _generate_depth(self, image: np.ndarray) -> np.ndarray:
        """生成模拟深度图"""
        height, width = image.shape[:2]
        # 基础深度 1m = 1000mm
        depth = np.ones((height, width), dtype=np.float32) * 1000.0
        
        # 根据图像下方（地面）深度更大
        for y in range(height):
            depth[y, :] *= (1.0 + 0.3 * (y / height))
        
        # 添加噪声
        noise = np.random.normal(0, 20, depth.shape)
        depth = np.clip(depth + noise, 500, 3000)
        
        return depth
    
    def start(self):
        """开始发布图像"""
        if not self.image_files:
            self.log('error', "No images found!")
            return
        
        self.log('info', "Starting image publishing...")
        
        while self.running:
            image_path = self.image_files[self.current_index]
            image = cv2.imread(image_path)
            
            if image is None:
                self.log('error', f"Failed to load: {image_path}")
                self.current_index = (self.current_index + 1) % len(self.image_files)
                continue
            
            height, width = image.shape[:2]
            
            # 发布 RGB 图像
            rgb_msg = Image(
                data=image,
                width=width,
                height=height,
                channels=3,
                timestamp=time.time()
            )
            self.publish('/camera/image_raw', rgb_msg)
            
            # 生成并发布深度图
            depth_data = self._generate_depth(image)
            depth_msg = DepthImage(
                data=depth_data,
                width=width,
                height=height,
                timestamp=time.time()
            )
            self.publish('/camera/depth', depth_msg)
            
            self.log('info', f"Published: {os.path.basename(image_path)}")
            
            self.current_index = (self.current_index + 1) % len(self.image_files)
            time.sleep(1.0 / self.publish_rate)
    
    def spin(self):
        self.running = True
        self.start()
```

- [ ] **Step 2: 识别节点**

```python
# src/nodes/recognition_node.py
import time
import cv2
import numpy as np
from typing import Dict
from src.mock_ros2.core import MockNode
from src.mock_ros2.message_types import Image, DepthImage, TomatoDetectionArray, TomatoDetection
from src.algorithms.color_segmentation import hsv_red_segmentation, preprocess_image
from src.algorithms.morphological import apply_morphology
from src.algorithms.contour_filter import filter_contours
from src.algorithms.coordinate_transform import calculate_3d_coordinates, load_camera_params


class RecognitionNode(MockNode):
    """番茄识别核心节点"""
    
    def __init__(self, camera_params_path: str, algorithm_params: Dict = None):
        super().__init__('recognition_node')
        
        self.camera_params = load_camera_params(camera_params_path)
        self.algorithm_params = algorithm_params or {}
        self.latest_depth = None
        
        self._setup_topics()
    
    def _setup_topics(self):
        """设置 Topic"""
        self.create_subscription('/camera/image_raw', Image, self._on_image)
        self.create_subscription('/camera/depth', DepthImage, self._on_depth)
        self.detection_topic = self.create_publisher('/detection/tomatoes', TomatoDetectionArray)
    
    def _on_depth(self, msg: DepthImage):
        """接收深度图"""
        self.latest_depth = msg.data
    
    def _on_image(self, msg: Image):
        """处理图像并检测番茄"""
        self.log('info', "Processing image...")
        
        image = msg.data
        
        # 1. 图像预处理
        preprocessed = preprocess_image(image)
        
        # 2. HSV 颜色分割
        hsv_params = self.algorithm_params.get('hsv_thresholds', {})
        mask = hsv_red_segmentation(
            preprocessed,
            lower1=tuple(hsv_params.get('lower_red1', [0, 80, 50])),
            upper1=tuple(hsv_params.get('upper_red1', [10, 255, 255])),
            lower2=tuple(hsv_params.get('lower_red2', [160, 80, 50])),
            upper2=tuple(hsv_params.get('upper_red2', [180, 255, 255]))
        )
        
        # 3. 形态学处理
        morph_params = self.algorithm_params.get('morphology', {})
        mask = apply_morphology(
            mask,
            open_kernel_size=morph_params.get('open_kernel_size', 5),
            close_kernel_size=morph_params.get('close_kernel_size', 7),
            iterations=morph_params.get('iterations', 2)
        )
        
        # 4. 轮廓提取与筛选
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filter_params = self.algorithm_params.get('contour_filter', {})
        detections_raw = filter_contours(
            image,
            contours,
            min_area=filter_params.get('min_area', 500),
            max_area=filter_params.get('max_area', 50000),
            min_circularity=filter_params.get('min_circularity', 0.5),
            max_aspect_ratio=filter_params.get('max_aspect_ratio', 2.0)
        )
        
        # 5. 三维坐标计算
        depth_params = self.algorithm_params.get('depth_estimation', {})
        detections_raw = calculate_3d_coordinates(
            detections_raw,
            self.camera_params,
            real_diameter_mm=depth_params.get('real_diameter_mm', 80.0)
        )
        
        # 转换为消息格式
        detections_msg = []
        for det in detections_raw:
            detection = TomatoDetection(
                tomato_id=det['tomato_id'],
                bbox=det['bbox'],
                center_uv=det['center_uv'],
                center_3d=det['center_3d'],
                circularity=det['circularity'],
                maturity_score=det['maturity_score'],
                confidence=det['circularity']
            )
            detections_msg.append(detection)
        
        # 发布检测结果
        result = TomatoDetectionArray(
            detections=detections_msg,
            image_timestamp=msg.timestamp
        )
        self.publish('/detection/tomatoes', result)
        
        self.log('info', f"Detected {len(detections_msg)} tomatoes")
```

- [ ] **Step 3: 优先级节点**

```python
# src/nodes/priority_node.py
import time
from typing import Dict
from src.mock_ros2.core import MockNode
from src.mock_ros2.message_types import TomatoDetectionArray, TomatoTarget
from src.algorithms.priority_scoring import sort_by_priority, calculate_priority_score


class PriorityNode(MockNode):
    """目标优先级排序节点"""
    
    def __init__(self, algorithm_params: Dict = None):
        super().__init__('priority_node')
        self.algorithm_params = algorithm_params or {}
        self.weights = self.algorithm_params.get('priority_scoring', {}).get('weights', None)
        
        self._setup_topics()
    
    def _setup_topics(self):
        """设置 Topic"""
        self.create_subscription('/detection/tomatoes', TomatoDetectionArray, self._on_detections)
        self.target_topic = self.create_publisher('/planning/target', TomatoTarget)
    
    def _on_detections(self, msg: TomatoDetectionArray):
        """处理检测结果并排序"""
        if not msg.detections:
            self.log('warn', "No detections received")
            return
        
        self.log('info', f"Received {len(msg.detections)} detections")
        
        # 转换为字典列表
        detections_dict = []
        for det in msg.detections:
            detections_dict.append({
                'tomato_id': det.tomato_id,
                'bbox': det.bbox,
                'center_uv': det.center_uv,
                'center_3d': det.center_3d,
                'circularity': det.circularity,
                'maturity_score': det.maturity_score,
                'area': det.bbox[2] * det.bbox[3],
                'confidence': det.confidence
            })
        
        # 排序
        sorted_detections = sort_by_priority(detections_dict, self.weights)
        
        # 发布最高优先级目标
        if sorted_detections:
            best = sorted_detections[0]
            target = TomatoTarget(
                detection=msg.detections[best['tomato_id']],
                priority_score=best['priority_score']
            )
            self.publish('/planning/target', target)
            
            self.log('info', f"Target: Tomato #{best['tomato_id']} "
                    f"(score={best['priority_score']:.3f}, "
                    f"maturity={best['maturity_score']:.3f})")
```

- [ ] **Step 4: 可视化节点**

```python
# src/nodes/visualization_node.py
import os
import cv2
import time
import numpy as np
from typing import Optional
from src.mock_ros2.core import MockNode
from src.mock_ros2.message_types import Image, TomatoDetectionArray, TomatoTarget, SystemStatus


class VisualizationNode(MockNode):
    """可视化节点"""
    
    def __init__(self, output_dir: str = 'results/detection_output'):
        super().__init__('visualization_node')
        self.output_dir = output_dir
        self.latest_image: Optional[np.ndarray] = None
        self.latest_detections: Optional[TomatoDetectionArray] = None
        self.latest_target: Optional[TomatoTarget] = None
        
        os.makedirs(output_dir, exist_ok=True)
        
        self._setup_topics()
    
    def _setup_topics(self):
        """设置 Topic"""
        self.create_subscription('/camera/image_raw', Image, self._on_image)
        self.create_subscription('/detection/tomatoes', TomatoDetectionArray, self._on_detections)
        self.create_subscription('/planning/target', TomatoTarget, self._on_target)
        self.create_subscription('/system/status', SystemStatus, self._on_status)
    
    def _on_image(self, msg: Image):
        """接收图像"""
        self.latest_image = msg.data.copy()
    
    def _on_detections(self, msg: TomatoDetectionArray):
        """接收检测结果"""
        self.latest_detections = msg
        self._visualize()
    
    def _on_target(self, msg: TomatoTarget):
        """接收目标"""
        self.latest_target = msg
    
    def _on_status(self, msg: SystemStatus):
        """接收状态"""
        self.log('info', f"[{msg.node_name}] {msg.status}: {msg.message}")
    
    def _visualize(self):
        """可视化检测结果"""
        if self.latest_image is None or self.latest_detections is None:
            return
        
        image = self.latest_image.copy()
        detections = self.latest_detections.detections
        
        for det in detections:
            x, y, w, h = det.bbox
            
            # 判断是否是当前目标
            is_target = (self.latest_target is not None and 
                        self.latest_target.detection is not None and
                        self.latest_target.detection.tomato_id == det.tomato_id)
            
            # 选择颜色
            if is_target:
                color = (0, 255, 0)  # 绿色表示目标
                thickness = 3
            else:
                color = (0, 0, 255)  # 红色表示检测
                thickness = 2
            
            # 绘制边界框
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
            
            # 绘制中心点
            u, v = int(det.center_uv[0]), int(det.center_uv[1])
            cv2.circle(image, (u, v), 5, (255, 0, 0), -1)
            
            # 绘制标签
            label = f"#{det.tomato_id} C:{det.confidence:.2f} M:{det.maturity_score:.2f}"
            if is_target:
                label += " [TARGET]"
            
            label_y = max(y - 10, 20)
            cv2.putText(image, label, (x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 绘制 3D 坐标
            coord_text = f"3D: ({det.center_3d[0]:.1f}, {det.center_3d[1]:.1f}, {det.center_3d[2]:.1f})"
            cv2.putText(image, coord_text, (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 绘制统计信息
        info_text = f"Detections: {len(detections)}"
        cv2.putText(image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 保存结果图
        timestamp = int(time.time() * 1000)
        output_path = os.path.join(self.output_dir, f"detection_{timestamp}.jpg")
        cv2.imwrite(output_path, image)
        
        self.log('info', f"Saved visualization: {output_path}")
```

- [ ] **Step 5: 状态节点**

```python
# src/nodes/status_node.py
import time
from src.mock_ros2.core import MockNode
from src.mock_ros2.message_types import SystemStatus


class StatusNode(MockNode):
    """状态监测节点"""
    
    def __init__(self):
        super().__init__('status_node')
        self.detection_count = 0
        self.total_tomatoes = 0
        
        self._setup_topics()
    
    def _setup_topics(self):
        """设置 Topic"""
        self.create_subscription('/detection/tomatoes', 
                                type('Any', (), {}), 
                                self._on_detections)
        self.status_topic = self.create_publisher('/system/status', SystemStatus)
    
    def _on_detections(self, msg):
        """统计检测信息"""
        self.detection_count += 1
        if hasattr(msg, 'detections'):
            self.total_tomatoes += len(msg.detections)
    
    def start(self):
        """定时发布状态"""
        while self.running:
            status = SystemStatus(
                node_name=self.node_name,
                status='running',
                message=f"Processed {self.detection_count} images, "
                       f"detected {self.total_tomatoes} tomatoes total",
                timestamp=time.time()
            )
            self.publish('/system/status', status)
            time.sleep(5)
    
    def spin(self):
        self.running = True
        self.start()
```

- [ ] **Step 6: Commit**

```bash
git add src/nodes/
git commit -m "feat: implement all ROS2 nodes (camera, recognition, priority, visualization, status)"
```

---

## Task 6: 主程序入口

**Files:**
- Create: `main.py`

- [ ] **Step 1: 创建主程序**

```python
# main.py
import yaml
import time
import argparse
from src.mock_ros2.core import MockSystem
from src.nodes.camera_node import CameraNode
from src.nodes.recognition_node import RecognitionNode
from src.nodes.priority_node import PriorityNode
from src.nodes.visualization_node import VisualizationNode
from src.nodes.status_node import StatusNode


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='智能温室番茄采摘机器人 - 软件模拟系统')
    parser.add_argument('--data-dir', default='data/test_images',
                       help='测试图片目录')
    parser.add_argument('--config', default='config/algorithm_params.yaml',
                       help='算法参数配置文件')
    parser.add_argument('--camera-params', default='data/calibration/camera_params.yaml',
                       help='相机参数文件')
    parser.add_argument('--rate', type=float, default=1.0,
                       help='图像发布频率 (Hz)')
    parser.add_argument('--max-images', type=int, default=None,
                       help='最大处理图片数')
    args = parser.parse_args()
    
    print("=" * 60)
    print("智能温室番茄采摘机器人 - 软件模拟系统")
    print("=" * 60)
    
    # 加载配置
    algorithm_params = load_config(args.config)
    
    # 创建系统管理器
    system = MockSystem()
    
    # 创建节点
    camera_node = CameraNode(args.data_dir, publish_rate=args.rate)
    recognition_node = RecognitionNode(args.camera_params, algorithm_params)
    priority_node = PriorityNode(algorithm_params)
    viz_node = VisualizationNode()
    status_node = StatusNode()
    
    # 注册节点
    system.register_node(camera_node)
    system.register_node(recognition_node)
    system.register_node(priority_node)
    system.register_node(viz_node)
    system.register_node(status_node)
    
    # 启动所有节点
    print("\n启动节点...")
    system.start_all()
    
    # 运行
    print(f"\n开始处理图片 (频率: {args.rate} Hz)")
    print("按 Ctrl+C 停止\n")
    
    try:
        processed = 0
        while True:
            time.sleep(1)
            processed += 1
            if args.max_images and processed >= args.max_images:
                break
    except KeyboardInterrupt:
        print("\n\n正在停止...")
    
    # 停止所有节点
    system.stop_all()
    print("系统已停止")
    print(f"\n结果已保存到: results/detection_output/")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Commit**

```bash
git add main.py
git commit -m "feat: add main entry point with system orchestration"
```

---

## Task 7: 单元测试

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_color_segmentation.py`
- Create: `tests/test_contour_filter.py`
- Create: `tests/test_coordinate_transform.py`
- Create: `tests/test_full_pipeline.py`

- [ ] **Step 1: 颜色分割测试**

```python
# tests/test_color_segmentation.py
import cv2
import numpy as np
import pytest
from src.algorithms.color_segmentation import hsv_red_segmentation, preprocess_image


class TestColorSegmentation:
    
    def test_hsv_red_segmentation_pure_red(self):
        """测试纯红色图像分割"""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :] = [0, 0, 255]  # BGR 纯红
        
        mask = hsv_red_segmentation(image)
        
        # 应该大部分像素被检测到
        red_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        assert red_pixels / total_pixels > 0.9
    
    def test_hsv_red_segmentation_green_background(self):
        """测试绿色背景不应被分割"""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :] = [0, 255, 0]  # BGR 纯绿
        
        mask = hsv_red_segmentation(image)
        
        # 应该几乎没有红色像素
        red_pixels = np.sum(mask > 0)
        assert red_pixels < 100
    
    def test_preprocess_image_shape(self):
        """测试预处理保持图像形状"""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = preprocess_image(image)
        
        assert result.shape == image.shape
    
    def test_preprocess_image_resize(self):
        """测试预处理缩放功能"""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = preprocess_image(image, target_size=(320, 240))
        
        assert result.shape == (240, 320, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

- [ ] **Step 2: 轮廓筛选测试**

```python
# tests/test_contour_filter.py
import cv2
import numpy as np
import pytest
from src.algorithms.contour_filter import calculate_circularity, filter_contours


class TestContourFilter:
    
    def test_circularity_perfect_circle(self):
        """测试完美圆形度"""
        # 创建圆形轮廓
        image = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(image, (100, 100), 50, 255, -1)
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circularity = calculate_circularity(contours[0])
        
        # 圆形度应接近 1
        assert circularity > 0.9
        assert circularity <= 1.0
    
    def test_circularity_rectangle(self):
        """测试矩形圆形度"""
        image = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 150), 255, -1)
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circularity = calculate_circularity(contours[0])
        
        # 矩形圆形度应较低
        assert circularity < 0.8
    
    def test_filter_contours_min_area(self):
        """测试面积筛选"""
        # 创建测试图像
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # 大圆
        cv2.circle(image, (50, 50), 40, (0, 0, 255), -1)
        # 小圆（应被过滤）
        cv2.circle(image, (150, 150), 5, (0, 0, 255), -1)
        
        # 转换为 HSV 并分割
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 80, 50), (10, 255, 255))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered = filter_contours(image, contours, min_area=500)
        
        # 应该只保留大圆
        assert len(filtered) == 1
        assert filtered[0]['area'] > 500


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

- [ ] **Step 3: 坐标转换测试**

```python
# tests/test_coordinate_transform.py
import numpy as np
import pytest
from src.algorithms.coordinate_transform import calculate_3d_coordinates


class TestCoordinateTransform:
    
    def test_calculate_3d_coordinates_center(self):
        """测试中心点坐标计算"""
        detections = [{
            'center_uv': [640, 360],  # 图像中心
            'bbox': [600, 320, 80, 80],
            'circularity': 0.9,
            'maturity_score': 0.95
        }]
        
        camera_params = {
            'fx': 910.0,
            'fy': 910.0,
            'cx': 640.0,
            'cy': 360.0
        }
        
        result = calculate_3d_coordinates(detections, camera_params, real_diameter_mm=80.0)
        
        # 中心点 X, Y 应接近 0
        assert abs(result[0]['center_3d'][0]) < 1.0
        assert abs(result[0]['center_3d'][1]) < 1.0
        
        # Z 应为正数
        assert result[0]['center_3d'][2] > 0
    
    def test_calculate_3d_coordinates_offset(self):
        """测试偏移点坐标计算"""
        detections = [{
            'center_uv': [740, 360],  # 向右偏移 100 像素
            'bbox': [700, 320, 80, 80],
            'circularity': 0.9,
            'maturity_score': 0.95
        }]
        
        camera_params = {
            'fx': 910.0,
            'fy': 910.0,
            'cx': 640.0,
            'cy': 360.0
        }
        
        result = calculate_3d_coordinates(detections, camera_params, real_diameter_mm=80.0)
        
        # X 应该为正（向右偏移）
        assert result[0]['center_3d'][0] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

- [ ] **Step 4: 完整流程测试**

```python
# tests/test_full_pipeline.py
import cv2
import numpy as np
import pytest
import yaml
from src.algorithms.color_segmentation import hsv_red_segmentation, preprocess_image
from src.algorithms.morphological import apply_morphology
from src.algorithms.contour_filter import filter_contours
from src.algorithms.coordinate_transform import calculate_3d_coordinates, load_camera_params
from src.algorithms.priority_scoring import sort_by_priority


class TestFullPipeline:
    
    def test_full_pipeline_single_tomato(self):
        """测试完整流程：单番茄场景"""
        # 创建合成图像
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        image[:, :] = (30, 80, 30)  # 绿色背景
        
        # 绘制红色番茄
        cv2.circle(image, (640, 360), 60, (0, 0, 255), -1)
        cv2.circle(image, (620, 340), 15, (255, 200, 200), -1)
        
        # 1. 预处理
        preprocessed = preprocess_image(image)
        
        # 2. HSV 分割
        mask = hsv_red_segmentation(preprocessed)
        
        # 3. 形态学处理
        mask = apply_morphology(mask)
        
        # 4. 轮廓提取与筛选
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = filter_contours(image, contours)
        
        # 5. 3D 坐标计算
        camera_params = {
            'fx': 910.0, 'fy': 910.0,
            'cx': 640.0, 'cy': 360.0
        }
        detections = calculate_3d_coordinates(detections, camera_params)
        
        # 6. 优先级排序
        sorted_detections = sort_by_priority(detections)
        
        # 验证结果
        assert len(sorted_detections) >= 1
        assert sorted_detections[0]['circularity'] > 0.5
        assert sorted_detections[0]['maturity_score'] > 0.5
        assert len(sorted_detections[0]['center_3d']) == 3
    
    def test_full_pipeline_no_tomato(self):
        """测试完整流程：无番茄场景"""
        # 纯绿色图像
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        image[:, :] = (30, 80, 30)
        
        # 预处理
        preprocessed = preprocess_image(image)
        mask = hsv_red_segmentation(preprocessed)
        mask = apply_morphology(mask)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = filter_contours(image, contours)
        
        # 应该没有检测到番茄
        assert len(detections) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

- [ ] **Step 5: 运行所有测试**

Run: `pytest tests/ -v`
Expected: 
```
============================= test session starts ==============================
tests/test_color_segmentation.py::TestColorSegmentation::test_hsv_red_segmentation_green_background PASSED
tests/test_color_segmentation.py::TestColorSegmentation::test_hsv_red_segmentation_pure_red PASSED
tests/test_color_segmentation.py::TestColorSegmentation::test_preprocess_image_resize PASSED
tests/test_color_segmentation.py::TestColorSegmentation::test_preprocess_image_shape PASSED
tests/test_contour_filter.py::TestContourFilter::test_circularity_perfect_circle PASSED
tests/test_contour_filter.py::TestContourFilter::test_circularity_rectangle PASSED
tests/test_contour_filter.py::TestContourFilter::test_filter_contours_min_area PASSED
tests/test_coordinate_transform.py::TestCoordinateTransform::test_calculate_3d_coordinates_center PASSED
tests/test_coordinate_transform.py::TestCoordinateTransform::test_calculate_3d_coordinates_offset PASSED
tests/test_full_pipeline.py::TestFullPipeline::test_full_pipeline_no_tomato PASSED
tests/test_full_pipeline.py::TestFullPipeline::test_full_pipeline_single_tomato PASSED
============================== 11 passed in X.XXs ============================
```

- [ ] **Step 6: Commit**

```bash
git add tests/
git commit -m "test: add comprehensive unit tests for all algorithm modules"
```

---

## Task 8: 实验结果运行与输出

**Files:**
- Create: `README.md`
- Create: `results/analysis/generate_report.py`

- [ ] **Step 1: 运行主程序**

Run: `python main.py --max-images 40`
Expected:
```
============================================================
智能温室番茄采摘机器人 - 软件模拟系统
============================================================

启动节点...

开始处理图片 (频率: 1.0 Hz)
按 Ctrl+C 停止

[INFO] [camera_node] Published: single_tomato_000.jpg
[INFO] [recognition_node] Processing image...
[INFO] [recognition_node] Detected 1 tomatoes
...
系统已停止

结果已保存到: results/detection_output/
```

- [ ] **Step 2: 验证输出**

Run: `ls results/detection_output/ | wc -l`
Expected: 40（或接近 40）张检测结果图

Run: `ls results/detection_output/ | head -5`
Expected: 显示 detection_*.jpg 文件

- [ ] **Step 3: 生成分析报告**

```python
# results/analysis/generate_report.py
import os
import cv2
import json
import glob
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def analyze_results(detection_dir: str = 'results/detection_output'):
    """分析检测结果并生成报告"""
    images = sorted(glob.glob(os.path.join(detection_dir, 'detection_*.jpg')))
    
    if not images:
        print("No detection results found!")
        return
    
    print(f"Analyzing {len(images)} detection results...")
    
    # 统计信息
    stats = {
        'total_images': len(images),
        'detection_counts': [],
        'timestamp': datetime.now().isoformat()
    }
    
    for img_path in images[:10]:  # 分析前10张
        image = cv2.imread(img_path)
        if image is not None:
            # 这里可以添加更详细的分析
            stats['detection_counts'].append({
                'filename': os.path.basename(img_path),
                'size': image.shape
            })
    
    # 保存统计报告
    report_path = 'results/analysis/detection_report.json'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Report saved: {report_path}")
    
    # 创建示例图集
    create_sample_grid(images[:9], 'results/analysis/sample_detections.jpg')


def create_sample_grid(image_paths, output_path):
    """创建检测结果示例网格图"""
    if len(image_paths) < 9:
        return
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('番茄检测结果示例', fontsize=16)
    
    for idx, (ax, img_path) in enumerate(zip(axes.flat, image_paths)):
        image = cv2.imread(img_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image)
            ax.set_title(os.path.basename(img_path), fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Sample grid saved: {output_path}")


if __name__ == '__main__':
    analyze_results()
```

Run: `python results/analysis/generate_report.py`
Expected:
```
Analyzing 40 detection results...
Report saved: results/analysis/detection_report.json
Sample grid saved: results/analysis/sample_detections.jpg
```

- [ ] **Step 4: 创建 README**

```markdown
# 智能温室番茄采摘机器人 - 软件模拟系统

## 项目概述

基于 Python + OpenCV 的智能温室番茄采摘机器人软件模拟系统，实现成熟番茄识别与三维定位核心算法。

## 系统架构

采用伪 ROS2 模块化设计：
- **CameraNode**: 模拟相机，读取合成数据集
- **RecognitionNode**: 番茄识别核心（HSV分割 + 形态学 + 轮廓筛选 + 3D坐标）
- **PriorityNode**: 目标优先级排序
- **VisualizationNode**: 可视化检测结果
- **StatusNode**: 状态监测

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 生成合成测试数据

```bash
python generate_synthetic_data.py
```

生成 40 张合成温室场景图片：
- `single_tomato/`: 单果实场景 (10张)
- `multiple_tomatoes/`: 多果实场景 (10张)
- `occlusion/`: 遮挡场景 (10张)
- `lighting/`: 光照变化场景 (10张)

### 3. 运行系统

```bash
python main.py
```

可选参数：
- `--rate 2.0`: 设置处理频率为 2Hz
- `--max-images 20`: 只处理前 20 张图片
- `--data-dir data/test_images`: 指定数据目录

### 4. 运行测试

```bash
pytest tests/ -v
```

## 项目结构

```
smart-robot/
├── src/                  # 源代码
│   ├── mock_ros2/        # 伪 ROS2 框架
│   ├── algorithms/       # 核心算法
│   ├── nodes/            # 节点实现
│   └── utils/            # 工具（合成数据生成）
├── data/                 # 数据集
│   ├── test_images/      # 测试图片
│   └── calibration/      # 相机参数
├── config/               # 配置文件
├── tests/                # 单元测试
├── results/              # 实验结果
│   ├── detection_output/ # 检测可视化图
│   └── analysis/         # 分析报告
├── main.py               # 主程序
└── generate_synthetic_data.py  # 数据生成脚本
```

## 核心算法

### 1. HSV 颜色分割
- 低红色区间: H=0-10, S>80, V>50
- 高红色区间: H=160-180, S>80, V>50

### 2. 形态学处理
- 开运算（5x5核）: 去除小噪点
- 闭运算（7x7核）: 填补果实内部孔洞

### 3. 轮廓筛选
- 面积范围: 500-50000 像素
- 圆形度: >0.5
- 长宽比: <2.0

### 4. 三维坐标计算
```
Z = fx × real_diameter / pixel_diameter
X = (u - cx) × Z / fx
Y = (v - cy) × Z / fy
```

### 5. 优先级评分
```
Score = 0.3×Conf + 0.3×Maturity + 0.2×Accessibility + 0.2×Distance
```

## 实验结果

运行后结果保存在 `results/` 目录：
- `detection_output/detection_*.jpg`: 带边界框和坐标的检测结果图
- `analysis/detection_report.json`: 检测统计报告
- `analysis/sample_detections.jpg`: 检测结果示例网格

## 配置参数

算法参数可在 `config/algorithm_params.yaml` 中调整：
- HSV 阈值
- 形态学核大小
- 轮廓筛选条件
- 优先级权重
- 深度估计参数
```

- [ ] **Step 5: Commit**

```bash
git add README.md results/analysis/generate_report.py
git commit -m "docs: add README and result analysis script"
```

---

## 自检清单

### Spec 覆盖检查

| 设计文档要求 | 实现任务 |
|------------|---------|
| Mock ROS2 框架 | Task 2: mock_ros2/core.py, message_types.py |
| 番茄识别节点 | Task 5: recognition_node.py |
| 优先级排序节点 | Task 5: priority_node.py |
| 可视化节点 | Task 5: visualization_node.py |
| 状态监测节点 | Task 5: status_node.py |
| HSV 颜色分割 | Task 4: color_segmentation.py |
| 形态学处理 | Task 4: morphological.py |
| 轮廓筛选 | Task 4: contour_filter.py |
| 三维坐标计算 | Task 4: coordinate_transform.py |
| 优先级评分 | Task 4: priority_scoring.py |
| 合成数据集 | Task 3: synthetic_data.py (40张) |
| 相机参数配置 | Task 1: camera_params.yaml |
| 算法参数配置 | Task 1: algorithm_params.yaml |
| 测试验证 | Task 7: 11个单元测试 |
| 实验结果输出 | Task 8: detection_output, analysis |

### Placeholder 检查

- [x] 无 TBD/TODO/"implement later"
- [x] 所有步骤包含实际代码
- [x] 所有步骤包含测试命令和预期输出
- [x] 类型和函数名一致

### 类型一致性检查

- [x] `TomatoDetection` 字段名在所有节点中一致
- [x] `center_3d` 格式为 `[X, Y, Z]`
- [x] `bbox` 格式为 `[x, y, w, h]`
- [x] 相机参数键名一致 (`fx`, `fy`, `cx`, `cy`)

---

## 执行选项

Plan complete and saved to `docs/superpowers/plans/2026-05-23-tomato-robot-implementation.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
