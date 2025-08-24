#!/usr/bin/env python3
"""
Deep Cam Server - TCP视频流处理服务器

使用人脸交换技术处理TCP视频流并输出到新的TCP流
"""

import argparse
import logging
import signal
import sys
from pathlib import Path

from deep_cam.capture import VideoCapture
from deep_cam.processor import FaceEnhancer, FaceSwapper


def setup_logging(level: str = "INFO"):
    """设置日志配置"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def signal_handler(signum, frame):
    """信号处理器，用于优雅退出"""
    print("\n收到退出信号，正在停止服务器...")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Deep Cam Server - TCP视频流人脸交换处理服务器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 输入输出配置
    parser.add_argument("--input-tcp", type=str, required=True, help="输入TCP流地址")
    parser.add_argument(
        "--output-tcp",
        type=str,
        default="tcp://localhost:8554",
        help="输出TCP流地址",
    )

    # 模型配置
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/inswapper_128_fp16.onnx"),
        help="人脸交换模型路径",
    )
    parser.add_argument(
        "--target-image", type=Path, default=Path("face.jpg"), help="目标人脸图像路径"
    )

    # 视频配置
    parser.add_argument("--width", type=int, default=640, help="输出视频宽度")
    parser.add_argument("--height", type=int, default=480, help="输出视频高度")
    parser.add_argument("--fps", type=int, default=30, help="输出视频帧率")

    # 日志配置
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # 验证文件存在性
    if not args.model_path.exists():
        logger.error(f"模型文件不存在: {args.model_path}")
        logger.info("请参考 models/instructions.txt 下载所需模型文件")
        sys.exit(1)

    if not args.target_image.exists():
        logger.error(f"目标图像文件不存在: {args.target_image}")
        sys.exit(1)

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    video_capture = None
    try:
        # 创建人脸交换处理器
        logger.info("初始化人脸交换处理器...")
        logger.info(f"模型路径: {args.model_path}")
        logger.info(f"目标图像: {args.target_image}")

        face_swapper = FaceSwapper(
            model_path=args.model_path, target_image_path=args.target_image
        )
        face_enhancer = FaceEnhancer(model_path=args.model_path)

        # 创建视频捕获器
        logger.info("初始化视频捕获器...")
        logger.info(f"输入TCP: {args.input_tcp}")
        logger.info(f"输出TCP: {args.output_tcp}")
        logger.info(f"输出分辨率: {args.width}x{args.height}@{args.fps}fps")

        video_capture = VideoCapture(
            processors=[face_swapper, face_enhancer],
            tcp_url=args.input_tcp,
            tcp_output_url=args.output_tcp,
            output_width=args.width,
            output_height=args.height,
            fps=args.fps,
        )

        # 启动视频处理
        logger.info("启动视频处理服务...")
        if not video_capture.start():
            logger.error("无法启动视频处理服务")
            sys.exit(1)

        logger.info("服务器已启动，按 Ctrl+C 停止")
        logger.info(f"输出流地址: {args.output_tcp}")

        # 保持运行直到收到信号
        try:
            while video_capture.is_running():
                import time

                time.sleep(1)
        except KeyboardInterrupt:
            pass

    except Exception as e:
        logger.error(f"启动服务器时发生错误: {e}")
        sys.exit(1)

    finally:
        # 清理资源
        try:
            if video_capture is not None:
                logger.info("正在停止视频处理服务...")
                video_capture.stop()
        except Exception:
            pass
        logger.info("服务器已停止")


if __name__ == "__main__":
    main()
