import logging
import subprocess
import threading
import time
from typing import List, Optional
import atexit

import cv2

from .processor import FrameProcessor
from .types import Frame


# 全局资源管理
_capture_semaphore = threading.Semaphore(3)  # 最多同时3个VideoCapture实例
_active_captures = threading.local()  # 线程本地存储活动的捕获实例
_cleanup_registered = False


def _cleanup_all_captures():
    """清理所有活动的捕获实例"""
    if hasattr(_active_captures, 'instances'):
        for capture in _active_captures.instances:
            try:
                capture.stop()
            except Exception as e:
                logging.error(f"清理捕获实例时出错: {e}")


def _register_cleanup():
    """注册清理函数"""
    global _cleanup_registered
    if not _cleanup_registered:
        atexit.register(_cleanup_all_captures)
        _cleanup_registered = True


class VideoCapture:
    def __init__(
        self,
        processors: List[FrameProcessor],
        tcp_url: str,
        tcp_output_url: str = "tcp://localhost:8554",
        output_width: int = 640,
        output_height: int = 480,
        fps: int = 30,
    ):
        self.tcp_url = tcp_url
        self.processors = processors
        self.tcp_output_url = tcp_output_url
        self.output_width = output_width
        self.output_height = output_height
        self.fps = fps

        # 线程安全的状态管理
        self._state_lock = threading.RLock()  # 使用RLock支持递归调用
        self._shutdown_event = threading.Event()  # 优雅关闭信号
        self._frame_ready_event = threading.Event()  # 帧准备就绪信号
        
        # 保护的共享状态
        self._cap: Optional[cv2.VideoCapture] = None
        self._ffmpeg_process: Optional[subprocess.Popen] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._semaphore_acquired = False

        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 注册清理和跟踪实例
        _register_cleanup()
        self._register_instance()

    def _register_instance(self):
        """注册此实例以便全局清理"""
        if not hasattr(_active_captures, 'instances'):
            _active_captures.instances = set()
        _active_captures.instances.add(self)

    def _unregister_instance(self):
        """取消注册此实例"""
        if hasattr(_active_captures, 'instances'):
            _active_captures.instances.discard(self)

    def _create_ffmpeg_command(self) -> list[str]:
        """创建 FFmpeg TCP 输出命令"""
        # 从tcp_output_url解析端口号
        port = "8554"
        if "://" in self.tcp_output_url:
            port = self.tcp_output_url.split(":")[-1]

        command = [
            "ffmpeg",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.output_width}x{self.output_height}",
            "-r",
            str(self.fps),
            "-i",
            "-",  # 从stdin读取
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-b:v",
            "2000k",
            "-f",
            "mpegts",
            "-listen",
            "1",
            f"tcp://0.0.0.0:{port}",
        ]
        return command

    def _initialize_capture(self) -> bool:
        """初始化视频捕获 - 线程安全"""
        try:
            with self._state_lock:
                if self._cap is not None:
                    return True
                    
                self._cap = cv2.VideoCapture(self.tcp_url)
                if not self._cap.isOpened():
                    self.logger.error(f"无法打开 TCP 流: {self.tcp_url}")
                    self._cap = None
                    return False

                # 设置缓冲区大小以减少延迟
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.logger.info(f"成功连接到 TCP 流: {self.tcp_url}")
            return True

        except Exception as e:
            self.logger.error(f"初始化视频捕获失败: {e}")
            with self._state_lock:
                if self._cap:
                    self._cap.release()
                    self._cap = None
            return False

    def _initialize_writer(self) -> bool:
        """初始化 FFmpeg 输出 - 线程安全"""
        try:
            with self._state_lock:
                if self._ffmpeg_process is not None:
                    return True
                    
                command = self._create_ffmpeg_command()

                self._ffmpeg_process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                )

                if self._ffmpeg_process.poll() is not None:
                    self.logger.error("无法启动 FFmpeg 进程")
                    self._ffmpeg_process = None
                    return False

            self.logger.info(
                f"FFmpeg TCP 服务器已初始化，输出URL: {self.tcp_output_url}"
            )
            return True

        except Exception as e:
            self.logger.error(f"初始化 FFmpeg 输出失败: {e}")
            with self._state_lock:
                if self._ffmpeg_process:
                    try:
                        self._ffmpeg_process.terminate()
                        self._ffmpeg_process.wait(timeout=2)
                    except:
                        pass
                    self._ffmpeg_process = None
            return False

    def _process_frame(self, frame: Frame) -> Optional[Frame]:
        """处理单帧图像"""
        try:
            # 检查是否收到关闭信号
            if self._shutdown_event.is_set():
                return None
                
            # 依次使用所有处理器处理帧
            processed_frame = frame
            for processor in self.processors:
                processor_start_time = time.perf_counter()
                processed_frame = processor.process_frame(processed_frame)
                processor_end_time = time.perf_counter()
                print(
                    f"处理器 {processor.__class__.__name__} 处理用时: {(processor_end_time - processor_start_time) * 1000:.2f} ms"
                )
                if processed_frame is None:
                    self.logger.warning(
                        f"处理器 {processor.__class__.__name__} 返回了 None"
                    )
                    return None

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
        """主捕获循环 - 带适当的异常处理和资源清理"""
        frame_time = 1.0 / self.fps

        try:
            while not self._shutdown_event.is_set():
                start_time = time.time()

                # 读取帧
                with self._state_lock:
                    if self._cap is None:
                        break
                    ret, frame = self._cap.read()

                if not ret:
                    self.logger.warning("无法读取帧，尝试重新连接...")
                    if not self._reconnect():
                        break
                    continue

                # 处理帧
                processed_frame = self._process_frame(frame)
                if processed_frame is None:
                    continue

                # 设置帧准备事件
                self._frame_ready_event.set()

                # 检查FFmpeg进程是否有效
                with self._state_lock:
                    if (
                        self._ffmpeg_process is None
                        or self._ffmpeg_process.poll() is not None
                    ):
                        self.logger.error("FFmpeg进程已终止")
                        break

                    # 输出帧到FFmpeg
                    try:
                        if self._ffmpeg_process.stdin is None:
                            self.logger.error("FFmpeg stdin不可用")
                            break
                        frame_bytes = processed_frame.tobytes()
                        self._ffmpeg_process.stdin.write(frame_bytes)
                        self._ffmpeg_process.stdin.flush()
                    except BrokenPipeError:
                        self.logger.error("FFmpeg管道已断开")
                        break
                    except Exception as e:
                        self.logger.error(f"写入FFmpeg失败: {e}")
                        break

                # 控制帧率
                elapsed = time.time() - start_time
                sleep_time = frame_time - elapsed
                if sleep_time > 0:
                    # 使用可中断的sleep
                    if not self._shutdown_event.wait(sleep_time):
                        continue  # 如果是超时（而非事件设置）则继续循环
                    else:
                        break  # 如果是事件设置则退出

        except Exception as e:
            self.logger.error(f"捕获循环中发生错误: {e}")
        finally:
            # 确保资源清理
            self._cleanup_resources()

    def _reconnect(self) -> bool:
        """重新连接 TCP 流 - 线程安全"""
        try:
            with self._state_lock:
                if self._cap:
                    self._cap.release()
                    self._cap = None

            if not self._shutdown_event.wait(1):  # 可中断的等待1秒
                return self._initialize_capture()
            else:
                return False  # 收到关闭信号

        except Exception as e:
            self.logger.error(f"重新连接失败: {e}")
            return False

    def _cleanup_resources(self):
        """清理所有资源 - 线程安全"""
        with self._state_lock:
            # 清理VideoCapture
            if self._cap:
                try:
                    self._cap.release()
                except Exception as e:
                    self.logger.error(f"释放VideoCapture失败: {e}")
                finally:
                    self._cap = None

            # 清理FFmpeg进程
            if self._ffmpeg_process:
                try:
                    if self._ffmpeg_process.stdin:
                        self._ffmpeg_process.stdin.close()
                    self._ffmpeg_process.terminate()
                    self._ffmpeg_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._ffmpeg_process.kill()
                    self._ffmpeg_process.wait()
                except Exception as e:
                    self.logger.error(f"终止FFmpeg进程失败: {e}")
                finally:
                    self._ffmpeg_process = None

    def start(self) -> bool:
        """启动视频捕获和处理 - 线程安全"""
        with self._state_lock:
            if self._running:
                self.logger.warning("视频捕获已在运行")
                return True

            # 尝试获取信号量
            if not _capture_semaphore.acquire(blocking=False):
                self.logger.error("无法获取捕获资源信号量，可能已达到最大并发数")
                return False
            
            self._semaphore_acquired = True

            try:
                # 重置关闭事件
                self._shutdown_event.clear()
                self._frame_ready_event.clear()

                # 初始化捕获和输出
                if not self._initialize_capture():
                    return False

                if not self._initialize_writer():
                    self._cleanup_resources()
                    return False

                # 启动处理线程
                self._running = True
                self._thread = threading.Thread(
                    target=self._capture_loop, 
                    daemon=False,  # 不使用daemon线程以确保优雅关闭
                    name=f"VideoCapture-{id(self)}"
                )
                self._thread.start()

                self.logger.info("视频捕获已启动")
                return True

            except Exception as e:
                self.logger.error(f"启动视频捕获失败: {e}")
                self._cleanup_resources()
                if self._semaphore_acquired:
                    _capture_semaphore.release()
                    self._semaphore_acquired = False
                return False

    def stop(self):
        """停止视频捕获和处理 - 线程安全"""
        with self._state_lock:
            if not self._running:
                return

            self.logger.info("正在停止视频捕获...")
            
            # 设置关闭事件
            self._shutdown_event.set()
            self._running = False

        # 等待线程结束（在锁外进行以避免死锁）
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                self.logger.warning("捕获线程未能在预期时间内结束")

        # 清理资源
        self._cleanup_resources()
        
        # 释放信号量
        if self._semaphore_acquired:
            _capture_semaphore.release()
            self._semaphore_acquired = False

        # 取消注册实例
        self._unregister_instance()
        
        with self._state_lock:
            self._thread = None

        self.logger.info("视频捕获已停止")

    def is_running(self) -> bool:
        """检查是否正在运行 - 线程安全"""
        with self._state_lock:
            return self._running and self._thread is not None and self._thread.is_alive()

    def wait_for_frame_ready(self, timeout: Optional[float] = None) -> bool:
        """等待帧准备就绪"""
        return self._frame_ready_event.wait(timeout)

    def __enter__(self) -> "VideoCapture":
        """上下文管理器入口"""
        if self.start():
            return self
        else:
            raise RuntimeError("无法启动视频捕获")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()

    def __del__(self):
        """析构函数，确保资源清理"""
        try:
            self.stop()
        except Exception:
            pass
