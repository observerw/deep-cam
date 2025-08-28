#!/usr/bin/env python3
"""
测量 inswapper_128 模型推理速度的基准测试脚本
参考 deep_cam/processor.py 中的 FaceSwapper 实现
"""

import time
import logging
import statistics
from pathlib import Path
from dataclasses import dataclass
from functools import cached_property
from typing import List

import cv2
import insightface
import onnxruntime as ort
from insightface.app.common import Face
from insightface.model_zoo.inswapper import INSwapper

from deep_cam.face_analyzer import FaceAnalyzer
from deep_cam.types import Frame


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class InswapperBenchmark:
    """Inswapper 模型性能基准测试类"""
    model_path: Path
    source_image_path: Path
    target_image_path: Path
    warmup_runs: int = 5
    benchmark_runs: int = 50

    @cached_property
    def face_analyzer(self) -> FaceAnalyzer:
        """人脸分析器"""
        return FaceAnalyzer()

    @cached_property
    def source_face(self) -> Face:
        """源人脸"""
        source_image = cv2.imread(self.source_image_path.as_posix())
        if source_image is None:
            raise ValueError(f"无法加载源图片: {self.source_image_path}")
        
        face = self.face_analyzer.get_one_face(source_image)
        if not face:
            raise ValueError(f"源图片中未检测到人脸: {self.source_image_path}")
        return face

    @cached_property
    def target_frame(self) -> Frame:
        """目标帧"""
        target_image = cv2.imread(self.target_image_path.as_posix())
        if target_image is None:
            raise ValueError(f"无法加载目标图片: {self.target_image_path}")
        
        # 验证目标图片中是否有人脸
        target_face = self.face_analyzer.get_one_face(target_image)
        if not target_face:
            raise ValueError(f"目标图片中未检测到人脸: {self.target_image_path}")
        
        return target_image

    @cached_property
    def swapper(self) -> INSwapper:
        """人脸交换模型"""
        # 检测可用的执行提供程序
        available_providers = ort.get_available_providers()
        logger.info(f"可用的 ONNX 执行提供程序: {available_providers}")
        
        # 检查 CUDA 是否可用
        cuda_available = "CUDAExecutionProvider" in available_providers
        logger.info(f"CUDA 提供程序可用: {cuda_available}")
        
        # 加载模型
        model = insightface.model_zoo.get_model(
            self.model_path.as_posix(),
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",  # 备用提供程序
            ],
        )
        
        # 检测模型实际使用的执行提供程序
        if hasattr(model, 'session') and hasattr(model.session, 'get_providers'):
            actual_providers = model.session.get_providers()
            logger.info(f"模型实际使用的提供程序: {actual_providers}")
            
            # 验证是否真正使用了 CUDA
            using_cuda = "CUDAExecutionProvider" in actual_providers
            if using_cuda:
                logger.info("✓ 人脸交换模型成功运行在 GPU (CUDA) 上")
            else:
                logger.warning("⚠ 人脸交换模型未运行在 GPU 上，回退到 CPU")
        else:
            logger.warning("无法检测模型执行提供程序")
        
        assert isinstance(model, INSwapper)
        return model

    def single_inference(self) -> float:
        """执行单次推理并返回耗时（秒）"""
        target_face = self.face_analyzer.get_one_face(self.target_frame)
        if not target_face:
            raise ValueError("目标帧中未检测到人脸")

        start_time = time.perf_counter()
        
        swapped_frame = self.swapper.get(
            img=self.target_frame,
            source_face=self.source_face,
            target_face=target_face,
            paste_back=True,
        )
        
        end_time = time.perf_counter()
        
        # 验证输出
        assert isinstance(swapped_frame, Frame)
        
        return end_time - start_time

    def warmup(self):
        """预热模型"""
        logger.info(f"开始预热，执行 {self.warmup_runs} 次推理...")
        for i in range(self.warmup_runs):
            inference_time = self.single_inference()
            logger.info(f"预热 {i+1}/{self.warmup_runs}: {inference_time:.4f}s")
        logger.info("预热完成")

    def benchmark(self) -> dict:
        """执行基准测试"""
        logger.info(f"开始基准测试，执行 {self.benchmark_runs} 次推理...")
        
        inference_times = []
        
        for i in range(self.benchmark_runs):
            inference_time = self.single_inference()
            inference_times.append(inference_time)
            
            if (i + 1) % 10 == 0:
                logger.info(f"进度: {i+1}/{self.benchmark_runs}")
        
        # 计算统计信息
        results = {
            'total_runs': len(inference_times),
            'mean_time': statistics.mean(inference_times),
            'median_time': statistics.median(inference_times),
            'min_time': min(inference_times),
            'max_time': max(inference_times),
            'std_dev': statistics.stdev(inference_times) if len(inference_times) > 1 else 0,
            'fps_mean': 1.0 / statistics.mean(inference_times),
            'fps_median': 1.0 / statistics.median(inference_times),
            'all_times': inference_times
        }
        
        return results

    def run_full_benchmark(self) -> dict:
        """运行完整的基准测试（包括预热）"""
        logger.info("=" * 60)
        logger.info("Inswapper 模型推理速度基准测试")
        logger.info("=" * 60)
        logger.info(f"模型路径: {self.model_path}")
        logger.info(f"源图片: {self.source_image_path}")
        logger.info(f"目标图片: {self.target_image_path}")
        logger.info(f"预热次数: {self.warmup_runs}")
        logger.info(f"测试次数: {self.benchmark_runs}")
        logger.info("-" * 60)
        
        # 预热
        self.warmup()
        
        logger.info("-" * 60)
        
        # 基准测试
        results = self.benchmark()
        
        # 输出结果
        logger.info("=" * 60)
        logger.info("基准测试结果:")
        logger.info(f"总运行次数: {results['total_runs']}")
        logger.info(f"平均推理时间: {results['mean_time']:.4f}s")
        logger.info(f"中位数推理时间: {results['median_time']:.4f}s")
        logger.info(f"最小推理时间: {results['min_time']:.4f}s")
        logger.info(f"最大推理时间: {results['max_time']:.4f}s")
        logger.info(f"标准差: {results['std_dev']:.4f}s")
        logger.info(f"平均 FPS: {results['fps_mean']:.2f}")
        logger.info(f"中位数 FPS: {results['fps_median']:.2f}")
        logger.info("=" * 60)
        
        return results


def main():
    """主函数"""
    # 配置路径 - 需要根据实际情况调整
    model_path = Path("models/inswapper_128.onnx")  # 需要指定实际的模型路径
    source_image_path = Path("source.jpg")  # 源人脸图片
    target_image_path = Path("data/man1.jpeg")  # 目标人脸图片
    
    # 检查文件是否存在
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}")
        logger.info("请下载 inswapper_128.onnx 模型文件并放置在正确位置")
        return
    
    if not source_image_path.exists():
        logger.error(f"源图片不存在: {source_image_path}")
        return
        
    if not target_image_path.exists():
        logger.error(f"目标图片不存在: {target_image_path}")
        return
    
    try:
        # 创建基准测试实例
        benchmark = InswapperBenchmark(
            model_path=model_path,
            source_image_path=source_image_path,
            target_image_path=target_image_path,
            warmup_runs=5,
            benchmark_runs=50
        )
        
        # 运行基准测试
        results = benchmark.run_full_benchmark()
        
        # 可选：保存结果到文件
        import json
        results_copy = results.copy()
        results_copy.pop('all_times', None)  # 移除详细时间数据以减少文件大小
        
        with open('benchmark_results.json', 'w', encoding='utf-8') as f:
            json.dump(results_copy, f, indent=2, ensure_ascii=False)
        
        logger.info("基准测试结果已保存到 benchmark_results.json")
        
    except Exception as e:
        logger.error(f"基准测试失败: {e}")
        raise


if __name__ == "__main__":
    main()