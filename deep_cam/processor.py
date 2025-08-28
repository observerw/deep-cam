from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from threading import RLock, Semaphore
from typing import override, Optional, Dict, Tuple
import threading
import atexit
import weakref

import cv2
import gfpgan
import insightface
import logging
import torch
from gfpgan.utils import GFPGANer
from insightface.app.common import Face
from insightface.model_zoo.inswapper import INSwapper

from deep_cam.face_analyzer import FaceAnalyzer
from deep_cam.types import Frame


class FrameProcessor(ABC):
    @abstractmethod
    def process_frame(self, frame: Frame) -> Frame: ...


@dataclass(frozen=True)
class FaceEnhancer(FrameProcessor):
    model_path: Path

    @cached_property
    def face_analyzer(self) -> FaceAnalyzer:
        return FaceAnalyzer()

    @cached_property
    def enhancer(self) -> GFPGANer:
        # 检测 CUDA 是否可用
        cuda_available = torch.cuda.is_available()
        logging.info(f"PyTorch CUDA available: {cuda_available}")
        
        if cuda_available:
            device = torch.device("cuda")
            logging.info(f"CUDA device count: {torch.cuda.device_count()}")
            logging.info(f"Current CUDA device: {torch.cuda.current_device()}")
            logging.info(f"CUDA device name: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logging.warning("⚠ CUDA not available, using CPU for face enhancement")
        
        enhancer = gfpgan.GFPGANer(
            model_path=self.model_path.as_posix(),
            upscale=2,
            device=device,
        )
        
        # 验证设备使用情况
        if cuda_available:
            logging.info("✓ Face enhancer model is successfully running on GPU (CUDA)")
        else:
            logging.info("Face enhancer model is running on CPU")
            
        return enhancer

    @cached_property
    def sem(self) -> Semaphore:
        return Semaphore()

    @override
    def process_frame(self, frame: Frame) -> Frame:
        target_face = self.face_analyzer.get_one_face(frame)

        if not target_face:
            return frame

        with self.sem:
            _, _, enhanced_frame = self.enhancer.enhance(frame, paste_back=True)
        assert isinstance(enhanced_frame, Frame)
        return enhanced_frame


# 全局模型缓存和锁 - 使用RLock以支持嵌套调用
_model_cache: Dict[Tuple[str, str], Tuple[INSwapper, FaceAnalyzer, Face]] = {}
_cache_lock = RLock()  # 改为RLock
_cleanup_registered = False

# 弱引用跟踪活动的FaceSwapper实例
_active_instances: weakref.WeakSet = weakref.WeakSet()


def _cleanup_models():
    """清理模型缓存和GPU资源"""
    with _cache_lock:
        if _model_cache:
            logging.info(f"清理 {len(_model_cache)} 个缓存的模型")
            for cache_key, (swapper, analyzer, face) in _model_cache.items():
                try:
                    # 清理GPU资源
                    if hasattr(swapper, 'session'):
                        del swapper.session
                    if hasattr(analyzer, 'analyzer'):
                        del analyzer.analyzer
                    logging.info(f"已清理模型: {cache_key}")
                except Exception as e:
                    logging.error(f"清理模型时出错 {cache_key}: {e}")
            
            _model_cache.clear()
            
        # 强制GPU内存清理（如果使用CUDA）
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logging.info("GPU内存缓存已清理")
            except Exception as e:
                logging.error(f"清理GPU缓存时出错: {e}")


def _register_cleanup():
    """注册清理函数"""
    global _cleanup_registered
    if not _cleanup_registered:
        atexit.register(_cleanup_models)
        _cleanup_registered = True


@dataclass(frozen=True)
class FaceSwapper(FrameProcessor):
    model_path: Path
    source_image_path: Path

    def __post_init__(self):
        """初始化后处理，确保模型只加载一次"""
        _register_cleanup()
        _active_instances.add(self)
        self._ensure_models_loaded()

    def _get_cache_key(self) -> Tuple[str, str]:
        """获取缓存键"""
        return (str(self.model_path.absolute()), str(self.source_image_path.absolute()))

    def _ensure_models_loaded(self):
        """确保模型已加载到缓存中 - 使用双重检查锁定模式"""
        cache_key = self._get_cache_key()
        
        # 第一次检查（无锁）
        if cache_key in _model_cache:
            return
            
        # 获取锁进行第二次检查和初始化
        with _cache_lock:
            if cache_key not in _model_cache:
                logging.info(f"首次加载模型: {self.model_path}")
                
                # 加载人脸分析器
                face_analyzer = FaceAnalyzer()
                
                # 加载源人脸
                source_face_image = cv2.imread(self.source_image_path.as_posix())
                if source_face_image is None:
                    raise ValueError(f"无法加载源图片: {self.source_image_path}")
                
                source_face = face_analyzer.get_one_face(source_face_image)
                if not source_face:
                    raise ValueError(f"源图片中未检测到人脸: {self.source_image_path}")
                
                # 加载交换模型
                swapper = self._load_swapper_model()
                
                # 缓存所有模型
                _model_cache[cache_key] = (swapper, face_analyzer, source_face)
                logging.info(f"模型加载完成并已缓存: {cache_key}")
            else:
                logging.info(f"模型已存在于缓存中: {cache_key}")

    def _load_swapper_model(self) -> INSwapper:
        """加载交换模型"""
        # 检测可用的执行提供程序
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        logging.info(f"Available ONNX providers: {available_providers}")
        
        # 检查 CUDA 是否可用
        cuda_available = "CUDAExecutionProvider" in available_providers
        logging.info(f"CUDA provider available: {cuda_available}")
        
        model = insightface.model_zoo.get_model(
            self.model_path.as_posix(),
            providers=[
                "CUDAExecutionProvider",
            ],
        )
        
        # 检测模型实际使用的执行提供程序
        if hasattr(model, 'session') and hasattr(model.session, 'get_providers'):
            actual_providers = model.session.get_providers()
            logging.info(f"Model is using providers: {actual_providers}")
            
            # 验证是否真正使用了 CUDA
            using_cuda = "CUDAExecutionProvider" in actual_providers
            if using_cuda:
                logging.info("✓ Face swapper model is successfully running on GPU (CUDA)")
            else:
                logging.warning("⚠ Face swapper model is NOT running on GPU, falling back to CPU")
        else:
            logging.warning("Unable to detect model execution providers")
        
        assert isinstance(model, INSwapper)
        return model

    @property
    def face_analyzer(self) -> FaceAnalyzer:
        """获取人脸分析器"""
        cache_key = self._get_cache_key()
        with _cache_lock:
            return _model_cache[cache_key][1]

    @property
    def source_face(self) -> Face:
        """获取源人脸"""
        cache_key = self._get_cache_key()
        with _cache_lock:
            return _model_cache[cache_key][2]

    @property
    def swapper(self) -> INSwapper:
        """获取交换模型"""
        cache_key = self._get_cache_key()
        with _cache_lock:
            return _model_cache[cache_key][0]

    @override
    def process_frame(self, frame: Frame) -> Frame:
        """帧处理方法"""
        target_face = self.face_analyzer.get_one_face(frame)
        if not target_face:
            return frame

        swapped_frame = self.swapper.get(
            img=frame,
            source_face=self.source_face,
            target_face=target_face,
            paste_back=True,
        )
        assert isinstance(swapped_frame, Frame)
        return swapped_frame


def cleanup_models():
    """手动清理模型缓存的公共接口"""
    _cleanup_models()


def get_cache_stats() -> Dict[str, int]:
    """获取缓存统计信息"""
    with _cache_lock:
        return {
            "cached_models": len(_model_cache),
            "active_instances": len(_active_instances),
        }
