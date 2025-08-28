import platform
import threading
import time
import logging
import weakref
from typing import Dict, List, Optional, Tuple
import atexit

import cv2


# 全局线程安全的摄像头管理
_camera_registry_lock = threading.RLock()
_camera_registry: Dict[int, 'CameraHandle'] = {}  # 摄像头注册表
_camera_detection_lock = threading.Lock()  # 摄像头检测锁
_camera_semaphore = threading.Semaphore(5)  # 限制同时打开的摄像头数量
_active_cameras = weakref.WeakValueDictionary()  # 追踪活动的摄像头实例


class CameraHandle:
    """线程安全的摄像头句柄包装器"""
    
    def __init__(self, camera_index: int):
        self.camera_index = camera_index
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.RLock()
        self._ref_count = 0
        self._last_access = time.time()
        
        logging.info(f"创建摄像头句柄: {camera_index}")
    
    def acquire(self) -> Optional[cv2.VideoCapture]:
        """获取摄像头资源"""
        with self._lock:
            if self._cap is None:
                try:
                    self._cap = cv2.VideoCapture(self.camera_index)
                    if not self._cap.isOpened():
                        self._cap.release()
                        self._cap = None
                        logging.warning(f"无法打开摄像头 {self.camera_index}")
                        return None
                    logging.info(f"成功打开摄像头 {self.camera_index}")
                except Exception as e:
                    logging.error(f"打开摄像头 {self.camera_index} 时出错: {e}")
                    return None
            
            self._ref_count += 1
            self._last_access = time.time()
            return self._cap
    
    def release(self):
        """释放摄像头资源"""
        with self._lock:
            self._ref_count = max(0, self._ref_count - 1)
            self._last_access = time.time()
            
            if self._ref_count == 0 and self._cap is not None:
                try:
                    self._cap.release()
                    logging.info(f"释放摄像头资源 {self.camera_index}")
                except Exception as e:
                    logging.error(f"释放摄像头 {self.camera_index} 时出错: {e}")
                finally:
                    self._cap = None
    
    def is_in_use(self) -> bool:
        """检查摄像头是否正在使用"""
        with self._lock:
            return self._ref_count > 0
    
    def get_ref_count(self) -> int:
        """获取引用计数"""
        with self._lock:
            return self._ref_count
    
    def get_last_access(self) -> float:
        """获取最后访问时间"""
        with self._lock:
            return self._last_access


class ThreadSafeCameraManager:
    """线程安全的摄像头管理器"""
    
    @staticmethod
    def get_camera_handle(camera_index: int) -> CameraHandle:
        """获取摄像头句柄"""
        with _camera_registry_lock:
            if camera_index not in _camera_registry:
                _camera_registry[camera_index] = CameraHandle(camera_index)
            return _camera_registry[camera_index]
    
    @staticmethod
    def acquire_camera(camera_index: int) -> Optional[cv2.VideoCapture]:
        """获取摄像头资源"""
        if not _camera_semaphore.acquire(blocking=False):
            logging.warning("已达到最大摄像头并发数限制")
            return None
        
        try:
            handle = ThreadSafeCameraManager.get_camera_handle(camera_index)
            cap = handle.acquire()
            if cap is None:
                _camera_semaphore.release()
                return None
            
            _active_cameras[id(cap)] = handle
            return cap
        except Exception as e:
            _camera_semaphore.release()
            logging.error(f"获取摄像头 {camera_index} 失败: {e}")
            return None
    
    @staticmethod
    def release_camera(cap: cv2.VideoCapture):
        """释放摄像头资源"""
        try:
            cap_id = id(cap)
            if cap_id in _active_cameras:
                handle = _active_cameras[cap_id]
                handle.release()
                del _active_cameras[cap_id]
                _camera_semaphore.release()
        except Exception as e:
            logging.error(f"释放摄像头时出错: {e}")
    
    @staticmethod
    def get_registry_stats() -> Dict:
        """获取摄像头注册表统计信息"""
        with _camera_registry_lock:
            stats = {
                'registered_cameras': len(_camera_registry),
                'active_cameras': len(_active_cameras),
                'camera_details': {}
            }
            
            for index, handle in _camera_registry.items():
                stats['camera_details'][index] = {
                    'ref_count': handle.get_ref_count(),
                    'in_use': handle.is_in_use(),
                    'last_access': handle.get_last_access()
                }
            
            return stats
    
    @staticmethod
    def cleanup_unused_cameras(max_idle_time: float = 300):  # 5分钟
        """清理长时间未使用的摄像头资源"""
        current_time = time.time()
        cleanup_list = []
        
        with _camera_registry_lock:
            for index, handle in list(_camera_registry.items()):
                if (not handle.is_in_use() and 
                    current_time - handle.get_last_access() > max_idle_time):
                    cleanup_list.append(index)
            
            for index in cleanup_list:
                try:
                    handle = _camera_registry.pop(index)
                    handle.release()  # 确保资源释放
                    logging.info(f"清理长时间未使用的摄像头句柄: {index}")
                except Exception as e:
                    logging.error(f"清理摄像头句柄 {index} 时出错: {e}")


def get_available_cameras() -> tuple[list[int], list[str]]:
    """线程安全的获取可用摄像头列表"""
    with _camera_detection_lock:
        if platform.system() == "Windows":
            return _get_windows_cameras()
        else:
            return _get_unix_cameras()


def _get_windows_cameras() -> tuple[list[int], list[str]]:
    """Windows系统摄像头检测"""
    try:
        from pygrabber.dshow_graph import FilterGraph

        graph = FilterGraph()
        devices = graph.get_input_devices()

        # Create list of indices and names
        camera_indices = list(range(len(devices)))
        camera_names = devices

        # If no cameras found through DirectShow, try OpenCV fallback
        if not camera_names:
            # Try to open camera with index -1 and 0
            test_indices = [-1, 0]
            working_cameras = []

            for idx in test_indices:
                cap = ThreadSafeCameraManager.acquire_camera(idx)
                if cap is not None:
                    working_cameras.append(f"Camera {idx}")
                    ThreadSafeCameraManager.release_camera(cap)

            if working_cameras:
                return test_indices[: len(working_cameras)], working_cameras

        # If still no cameras found, return empty lists
        if not camera_names:
            return [], ["No cameras found"]

        return camera_indices, camera_names

    except Exception as e:
        logging.error(f"Windows摄像头检测错误: {str(e)}")
        return [], ["No cameras found"]


def _get_unix_cameras() -> tuple[list[int], list[str]]:
    """Unix-like系统摄像头检测"""
    camera_indices = []
    camera_names = []

    if platform.system() == "Darwin":  # macOS specific handling
        # Try to open the default FaceTime camera first
        cap = ThreadSafeCameraManager.acquire_camera(0)
        if cap is not None:
            camera_indices.append(0)
            camera_names.append("FaceTime Camera")
            ThreadSafeCameraManager.release_camera(cap)

        # On macOS, additional cameras typically use indices 1 and 2
        for i in [1, 2]:
            cap = ThreadSafeCameraManager.acquire_camera(i)
            if cap is not None:
                camera_indices.append(i)
                camera_names.append(f"Camera {i}")
                ThreadSafeCameraManager.release_camera(cap)
    else:
        # Linux camera detection - test first 10 indices
        for i in range(10):
            cap = ThreadSafeCameraManager.acquire_camera(i)
            if cap is not None:
                camera_indices.append(i)
                camera_names.append(f"Camera {i}")
                ThreadSafeCameraManager.release_camera(cap)

    if not camera_names:
        return [], ["No cameras found"]

    return camera_indices, camera_names


def cleanup_all_cameras():
    """清理所有摄像头资源"""
    with _camera_registry_lock:
        camera_count = len(_camera_registry)
        if camera_count > 0:
            logging.info(f"清理 {camera_count} 个摄像头资源")
            
            for handle in list(_camera_registry.values()):
                try:
                    handle.release()
                except Exception as e:
                    logging.error(f"清理摄像头资源时出错: {e}")
            
            _camera_registry.clear()
            _active_cameras.clear()


# 创建全局管理器实例
camera_manager = ThreadSafeCameraManager()

# 注册清理函数
atexit.register(cleanup_all_cameras)

# 定期清理任务（可选）
_cleanup_timer: Optional[threading.Timer] = None

def _periodic_cleanup():
    """定期清理未使用的摄像头资源"""
    try:
        ThreadSafeCameraManager.cleanup_unused_cameras()
    except Exception as e:
        logging.error(f"定期清理摄像头资源时出错: {e}")
    finally:
        # 重新安排下次清理
        global _cleanup_timer
        _cleanup_timer = threading.Timer(300, _periodic_cleanup)  # 5分钟后
        _cleanup_timer.daemon = True
        _cleanup_timer.start()

# 启动定期清理
_periodic_cleanup()


# 向后兼容的函数
def acquire_camera(camera_index: int) -> Optional[cv2.VideoCapture]:
    """获取摄像头（向后兼容）"""
    return camera_manager.acquire_camera(camera_index)


def release_camera(cap: cv2.VideoCapture):
    """释放摄像头（向后兼容）"""
    camera_manager.release_camera(cap)


def get_camera_stats() -> Dict:
    """获取摄像头统计信息"""
    return camera_manager.get_registry_stats()
