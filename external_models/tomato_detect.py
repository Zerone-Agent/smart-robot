#!/usr/bin/env python3
"""
YOLOv8 番茄成熟度检测脚本
模型来源: pablofntdz/yolov8-tomato-detection (HuggingFace)
类别: Fully-ripe(全熟), Semi-ripe(半熟), Unripe(未熟)

用法:
  # 单张图片检测
  python tomato_detect.py image.jpg

  # 文件夹批量检测
  python tomato_detect.py ./images/

  # 视频文件检测
  python tomato_detect.py video.mp4 --video

  # 摄像头实时检测
  python tomato_detect.py --webcam

  # 指定输出目录
  python tomato_detect.py image.jpg -o ./results/

  # 设置置信度阈值
  python tomato_detect.py image.jpg --conf 0.3
"""

import argparse
import json
import sys
from pathlib import Path

from ultralytics import YOLO

# ── 模型路径 ──────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "yolov8_tomato.pt"

# 类别名称
CLASS_NAMES = {0: "Fully-ripe", 1: "Semi-ripe", 2: "Unripe"}
CLASS_CN = {0: "全熟", 1: "半熟", 2: "未熟"}

# 类别颜色 (BGR)
CLASS_COLORS = {
    0: (0, 0, 255),    # 全熟 - 红色
    1: (0, 165, 255),  # 半熟 - 橙色
    2: (0, 255, 0),    # 未熟 - 绿色
}


def detect_image(model: YOLO, image_path: Path, output_dir: Path, conf: float):
    """检测单张图片"""
    results = model(str(image_path), conf=conf)

    result = results[0]
    boxes = result.boxes

    # 构建检测结果
    detections = []
    if boxes is not None:
        for box in boxes:
            detections.append({
                "class_id": int(box.cls),
                "class_name": CLASS_NAMES[int(box.cls)],
                "class_cn": CLASS_CN[int(box.cls)],
                "confidence": round(float(box.conf), 4),
                "bbox": [round(float(x), 2) for x in box.xyxy[0].tolist()],
            })

    # 保存标注图片
    save_path = output_dir / f"detected_{image_path.stem}.jpg"
    result.save(filename=str(save_path))

    # 保存 JSON 结果
    json_path = output_dir / f"result_{image_path.stem}.json"
    summary = summarize(detections)
    output = {
        "source": str(image_path),
        "output": str(save_path),
        "total_detections": len(detections),
        "summary": summary,
        "detections": detections,
    }
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))

    return output


def detect_video(model: YOLO, video_path: Path, output_dir: Path, conf: float):
    """检测视频文件"""
    save_path = output_dir / f"detected_{video_path.stem}.mp4"
    results = model(str(video_path), conf=conf, save=True, project=str(output_dir), name="", exist_ok=True)
    # ultralytics 自动保存到 output_dir/video_stem.mp4
    # 重命名
    auto_path = output_dir / f"{video_path.stem}.mp4"
    if auto_path.exists():
        auto_path.rename(save_path)

    print(f"\n 视频结果已保存: {save_path}")
    return {"source": str(video_path), "output": str(save_path)}


def detect_webcam(model: YOLO, conf: float):
    """摄像头实时检测"""
    print("\n 启动摄像头检测...")
    print(" 按 'q' 键退出\n")
    # stream=True 实时显示
    results = model.predict(
        source=0,          # 默认摄像头
        conf=conf,
        show=True,         # 显示窗口
        stream=True,
    )
    for _ in results:
        pass  # 持续推流，直到按 q 退出


def detect_folder(model: YOLO, folder: Path, output_dir: Path, conf: float):
    """批量检测文件夹中的所有图片"""
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    images = [f for f in folder.iterdir() if f.suffix.lower() in image_exts]

    if not images:
        print(f"⚠️  未找到图片文件: {folder}")
        return

    print(f"\n 找到 {len(images)} 张图片，开始批量检测...\n")
    all_results = []

    for i, img in enumerate(images, 1):
        print(f"  [{i}/{len(images)}] {img.name} ...", end=" ")
        output = detect_image(model, img, output_dir, conf)
        summary = output["summary"]
        print(f"检出 {output['total_detections']} 个 (全熟:{summary['Fully-ripe']} 半熟:{summary['Semi-ripe']} 未熟:{summary['Unripe']})")
        all_results.append(output)

    # 批量汇总
    total = sum(r["total_detections"] for r in all_results)
    total_summary = {
        "Fully-ripe": sum(r["summary"]["Fully-ripe"] for r in all_results),
        "Semi-ripe": sum(r["summary"]["Semi-ripe"] for r in all_results),
        "Unripe": sum(r["summary"]["Unripe"] for r in all_results),
    }

    summary_file = output_dir / "batch_summary.json"
    summary_file.write_text(json.dumps({
        "total_images": len(images),
        "total_detections": total,
        "summary": total_summary,
        "details": all_results,
    }, ensure_ascii=False, indent=2))

    print(f"\n{'─'*50}")
    print(f"  批量检测完成!")
    print(f"  图片数: {len(images)} | 总检出: {total}")
    print(f"  全熟: {total_summary['Fully-ripe']} | 半熟: {total_summary['Semi-ripe']} | 未熟: {total_summary['Unripe']}")
    print(f"  汇总报告: {summary_file}")
    print(f"{'─'*50}")


def summarize(detections: list) -> dict:
    """统计各类别检测数量"""
    summary = {"Fully-ripe": 0, "Semi-ripe": 0, "Unripe": 0}
    for d in detections:
        name = d["class_name"]
        if name in summary:
            summary[name] += 1
    return summary


def main():
    parser = argparse.ArgumentParser(description="番茄成熟度检测工具")
    parser.add_argument("source", nargs="?", help="图片路径、文件夹路径或视频文件路径")
    parser.add_argument("--video", action="store_true", help="将输入作为视频处理")
    parser.add_argument("--webcam", action="store_true", help="使用摄像头实时检测")
    parser.add_argument("-o", "--output", default="./tomato_output", help="输出目录 (默认: ./tomato_output)")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 (默认: 0.25)")

    args = parser.parse_args()

    # 加载模型
    if not MODEL_PATH.exists():
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        print("   请将 yolov8_tomato.pt 放置在与脚本同目录下")
        sys.exit(1)

    print(f"🚀 加载模型: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    print(f"   类别: {list(CLASS_NAMES.values())}")
    print(f"   置信度阈值: {args.conf}\n")

    # 摄像头模式
    if args.webcam:
        detect_webcam(model, args.conf)
        return

    # 无输入源
    if not args.source:
        parser.print_help()
        print("\n示例:")
        print("  python tomato_detect.py photo.jpg")
        print("  python tomato_detect.py ./images/")
        print("  python tomato_detect.py video.mp4 --video")
        print("  python tomato_detect.py --webcam")
        return

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"❌ 文件不存在: {source_path}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 判断输入类型
    if args.video or source_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        detect_video(model, source_path, output_dir, args.conf)
    elif source_path.is_dir():
        detect_folder(model, source_path, output_dir, args.conf)
    else:
        output = detect_image(model, source_path, output_dir, args.conf)
        summary = output["summary"]
        print(f"\n{'─'*50}")
        print(f"  检测结果: {source_path.name}")
        print(f"  检出: {output['total_detections']} 个番茄")
        print(f"    全熟(Fully-ripe): {summary['Fully-ripe']}")
        print(f"    半熟(Semi-ripe): {summary['Semi-ripe']}")
        print(f"    未熟(Unripe):    {summary['Unripe']}")
        print(f"  标注图片: {output['output']}")
        print(f"  详细结果: {output_dir / f'result_{source_path.stem}.json'}")
        print(f"{'─'*50}")


if __name__ == "__main__":
    main()
