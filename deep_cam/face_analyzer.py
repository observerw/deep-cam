from functools import cached_property
from threading import Lock, Semaphore, Event
import threading
import logging
import weakref

from insightface.app.common import Face
from insightface.app.face_analysis import FaceAnalysis

from .types import Frame


# 全局资源管理
_analyzer_instances_lock = Lock()
_analyzer_instances = weakref.WeakSet()  # 跟踪所有分析器实例
_model_ready_event = Event()  # 模型准备就绪信号


class ThreadSafeFaceAnalyzer:
    """线程安全的人脸分析器包装器"""
    
    def __init__(self, max_concurrent: int = 2):
        self._analyzer_lock = Lock()
        self._analyzer = None
        self._sem = Semaphore(max_concurrent)  # 限制并发分析数量
        self._initialized = False
        
        # 注册实例
        with _analyzer_instances_lock:
            _analyzer_instances.add(self)
        
        logging.info(f"创建ThreadSafeFaceAnalyzer实例，最大并发数: {max_concurrent}")
    
    def _create_analyzer(self) -> FaceAnalysis:
        """创建分析器实例"""
        logging.info("正在初始化FaceAnalysis模型...")
        
        analyzer = FaceAnalysis(
            name="buffalo_l",
            providers=[
                "CUDAExecutionProvider",
            ],
        )
        analyzer.prepare(ctx_id=0, det_size=(640, 640))
        
        logging.info("✓ FaceAnalysis模型初始化完成")
        return analyzer
    
    @property
    def analyzer(self) -> FaceAnalysis:
        """获取分析器实例 - 使用双重检查锁定模式"""
        if self._analyzer is None:
            with self._analyzer_lock:
                if self._analyzer is None:
                    self._analyzer = self._create_analyzer()
                    self._initialized = True
                    # 设置全局模型就绪事件
                    _model_ready_event.set()
        return self._analyzer
    
    def get_one_face(self, frame: Frame) -> Face | None:
        """获取帧中的一个人脸 - 线程安全"""
        with self._sem:
            try:
                faces: list[Face] = self.analyzer.get(frame)
            except Exception as e:
                logging.error(f"人脸检测失败: {e}")
                return None

        def bbox_x(face: Face):
            return face.bbox[0] if face.bbox is not None else float("inf")

        try:
            return min(faces, key=bbox_x)
        except ValueError:
            return None

    def get_many_faces(self, frame: Frame) -> list[Face] | None:
        """获取帧中的多个人脸 - 线程安全"""
        with self._sem:
            try:
                return self.analyzer.get(frame)
            except (IndexError, Exception) as e:
                logging.error(f"多人脸检测失败: {e}")
                return None
    
    def wait_for_ready(self, timeout: float | None = None) -> bool:
        """等待分析器准备就绪"""
        if self._initialized:
            return True
        return _model_ready_event.wait(timeout)
    
    def is_ready(self) -> bool:
        """检查分析器是否准备就绪"""
        return self._initialized


# 保持向后兼容的别名
FaceAnalyzer = ThreadSafeFaceAnalyzer


def get_global_model_ready_event() -> Event:
    """获取全局模型就绪事件"""
    return _model_ready_event


def get_analyzer_instances_count() -> int:
    """获取活动分析器实例数量"""
    with _analyzer_instances_lock:
        return len(_analyzer_instances)


def cleanup_analyzers():
    """清理所有分析器资源"""
    with _analyzer_instances_lock:
        active_count = len(_analyzer_instances)
        if active_count > 0:
            logging.info(f"清理 {active_count} 个分析器实例")
            # 注意：由于使用了WeakSet，实例会在被垃圾回收时自动移除
            # 这里只是记录日志，实际的资源清理由Python的垃圾收集器处理


# 线程安全的工厂函数
def create_face_analyzer(max_concurrent: int = 2) -> ThreadSafeFaceAnalyzer:
    """创建线程安全的人脸分析器"""
    return ThreadSafeFaceAnalyzer(max_concurrent=max_concurrent)


# 模块级别的清理注册
import atexit
atexit.register(cleanup_analyzers)
