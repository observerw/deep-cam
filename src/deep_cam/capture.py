import logging
import threading
import time
from typing import Optional

import cv2

from .processor import FrameProcessor
from .types import Frame


class VideoCapture:
    def __init__(
        self,
        processor: FrameProcessor,
        rtsp_url: str,
        rtsp_output_url: str = "rtsp://localhost:8554/stream",
        output_width: int = 640,
        output_height: int = 480,
        fps: int = 30,
    ):
        self.rtsp_url = rtsp_url
        self.processor = processor
        self.rtsp_output_url = rtsp_output_url
        self.output_width = output_width
        self.output_height = output_height
        self.fps = fps

        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # 设置日志
        self.logger = logging.getLogger(__name__)

    def _create_gstreamer_pipeline(self) -> str:
        """创建 GStreamer RTSP 输出管道"""
        pipeline = (
            f"appsrc ! "
            f"videoconvert ! "
            f"video/x-raw,format=I420,width={self.output_width},height={self.output_height},framerate={self.fps}/1 ! "
            f"x264enc tune=zerolatency bitrate=2000 speed-preset=superfast ! "
            f"rtph264pay config-interval=1 pt=96 ! "
            f"rtspsink location={self.rtsp_output_url}"
        )
        return pipeline

    def _initialize_capture(self) -> bool:
        """初始化视频捕获"""
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            if not self.cap.isOpened():
                self.logger.error(f"无法打开 RTSP 流: {self.rtsp_url}")
                return False

            # 设置缓冲区大小以减少延迟
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.logger.info(f"成功连接到 RTSP 流: {self.rtsp_url}")
            return True

        except Exception as e:
            self.logger.error(f"初始化视频捕获失败: {e}")
            return False

    def _initialize_writer(self) -> bool:
        """初始化 GStreamer 输出"""
        try:
            pipeline = self._create_gstreamer_pipeline()
            fourcc = cv2.VideoWriter.fourcc(*"H264")

            self.writer = cv2.VideoWriter(
                pipeline, fourcc, self.fps, (self.output_width, self.output_height)
            )

            if not self.writer.isOpened():
                self.logger.error("无法初始化 GStreamer 输出")
                return False

            self.logger.info(f"RTSP 服务器已初始化，输出URL: {self.rtsp_output_url}")
            return True

        except Exception as e:
            self.logger.error(f"初始化 GStreamer 输出失败: {e}")
            return False

    def _process_frame(self, frame: Frame) -> Optional[Frame]:
        """处理单帧图像"""
        try:
            # 使用处理器处理帧
            processed_frame = self.processor.process_frame(frame)

            # 调整帧大小以匹配输出尺寸
            if processed_frame.shape[:2] != (self.output_height, self.output_width):
                processed_frame = cv2.resize(
                    processed_frame, (self.output_width, self.output_height)
                )

            return processed_frame

        except Exception as e:
            self.logger.error(f"帧处理失败: {e}")
            return None

    def _capture_loop(self):
        """主捕获循环"""
        frame_time = 1.0 / self.fps

        while self.running:
            start_time = time.time()

            try:
                # 检查捕获器是否有效
                if self.cap is None:
                    break

                # 读取帧
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("无法读取帧，尝试重新连接...")
                    if not self._reconnect():
                        break
                    continue

                # 处理帧
                processed_frame = self._process_frame(frame)
                if processed_frame is None:
                    continue

                # 检查写入器是否有效
                if self.writer is None:
                    break

                # 输出帧
                self.writer.write(processed_frame)

                # 控制帧率
                elapsed = time.time() - start_time
                sleep_time = frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                self.logger.error(f"捕获循环中发生错误: {e}")
                break

    def _reconnect(self) -> bool:
        """重新连接 RTSP 流"""
        try:
            if self.cap:
                self.cap.release()

            time.sleep(1)  # 等待一秒后重试
            return self._initialize_capture()

        except Exception as e:
            self.logger.error(f"重新连接失败: {e}")
            return False

    def start(self) -> bool:
        """启动视频捕获和处理"""
        if self.running:
            self.logger.warning("视频捕获已在运行")
            return True

        # 初始化捕获和输出
        if not self._initialize_capture():
            return False

        if not self._initialize_writer():
            self.stop()
            return False

        # 启动处理线程
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        self.logger.info("视频捕获已启动")
        return True

    def stop(self):
        """停止视频捕获和处理"""
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

        if self.cap:
            self.cap.release()
            self.cap = None

        if self.writer:
            self.writer.release()
            self.writer = None

        self.logger.info("视频捕获已停止")

    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self.running and self.thread is not None and self.thread.is_alive()

    def __enter__(self) -> "VideoCapture":
        """上下文管理器入口"""
        if self.start():
            return self
        else:
            raise RuntimeError("无法启动视频捕获")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()
