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
        "--swapper-model",
        type=Path,
        help="人脸交换模型路径",
    )
    parser.add_argument(
        "--enhancer-model",
        type=Path,
        help="人脸增强模型路径",
    )
    parser.add_argument(
        "--source-image", type=Path, default=Path("source.jpg"), help="源人脸图像路径"
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
    if args.swapper_model and not args.swapper_model.exists():
        logger.error(f"人脸交换模型文件不存在: {args.swapper_model}")
        logger.info("请参考 models/instructions.txt 下载所需模型文件")
        sys.exit(1)

    if args.enhancer_model and not args.enhancer_model.exists():
        logger.error(f"人脸增强模型文件不存在: {args.enhancer_model}")
        logger.info("请参考 models/instructions.txt 下载所需模型文件")
        sys.exit(1)

    if args.swapper_model and not args.source_image.exists():
        logger.error(f"目标图像文件不存在: {args.source_image}")
        sys.exit(1)

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    video_capture = None
    try:
        # 创建处理器列表
        processors = []

        # 创建人脸交换处理器（如果指定了模型）
        if args.swapper_model:
            logger.info("初始化人脸交换处理器...")
            logger.info(f"交换模型路径: {args.swapper_model}")
            logger.info(f"目标图像: {args.source_image}")
            face_swapper = FaceSwapper(
                model_path=args.swapper_model, source_image_path=args.source_image
            )
            processors.append(face_swapper)

        # 创建人脸增强处理器（如果指定了模型）
        if args.enhancer_model:
            logger.info("初始化人脸增强处理器...")
            logger.info(f"增强模型路径: {args.enhancer_model}")
            face_enhancer = FaceEnhancer(model_path=args.enhancer_model)
            processors.append(face_enhancer)

        if not processors:
            logger.error(
                "至少需要指定一个处理器模型（--swapper-model 或 --enhancer-model）"
            )
            sys.exit(1)

        # 创建视频捕获器
        logger.info("初始化视频捕获器...")
        logger.info(f"输入TCP: {args.input_tcp}")
        logger.info(f"输出TCP: {args.output_tcp}")
        logger.info(f"输出分辨率: {args.width}x{args.height}@{args.fps}fps")

        video_capture = VideoCapture(
            processors=processors,
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
