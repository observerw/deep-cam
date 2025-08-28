#!/usr/bin/env python3
"""
测试 deep_cam/processor.py 中 FaceSwapper 类的帧处理速度
直接使用 FaceSwapper 类进行性能基准测试
"""

import time
import logging
import statistics
import json
from pathlib import Path
from typing import List, Dict, Any

import cv2

from deep_cam.processor import FaceSwapper
from deep_cam.types import Frame


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FaceSwapperPerformanceTest:
    """FaceSwapper 性能测试类"""
    
    def __init__(
        self,
        model_path: Path,
        source_image_path: Path,
        target_image_path: Path,
        warmup_runs: int = 5,
        benchmark_runs: int = 50
    ):
        self.model_path = model_path
        self.source_image_path = source_image_path
        self.target_image_path = target_image_path
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        
        # 初始化 FaceSwapper
        self.face_swapper = None
        self.target_frame = None
        
    def setup(self):
        """初始化测试环境"""
        logger.info("初始化 FaceSwapper...")
        
        # 创建 FaceSwapper 实例
        self.face_swapper = FaceSwapper(
            model_path=self.model_path,
            source_image_path=self.source_image_path
        )
        
        # 加载目标图片
        self.target_frame = cv2.imread(self.target_image_path.as_posix())
        if self.target_frame is None:
            raise ValueError(f"无法加载目标图片: {self.target_image_path}")
        
        # 验证目标图片中是否有人脸
        target_face = self.face_swapper.face_analyzer.get_one_face(self.target_frame)
        if not target_face:
            raise ValueError(f"目标图片中未检测到人脸: {self.target_image_path}")
        
        logger.info("FaceSwapper 初始化完成")
        
    def single_inference(self) -> float:
        """执行单次帧处理并返回耗时（秒）"""
        start_time = time.perf_counter()
        
        # 使用 FaceSwapper 的 process_frame 方法
        processed_frame = self.face_swapper.process_frame(self.target_frame)
        
        end_time = time.perf_counter()
        
        # 验证输出
        assert isinstance(processed_frame, Frame)
        
        return end_time - start_time
    
    def warmup(self):
        """预热模型"""
        logger.info(f"开始预热，执行 {self.warmup_runs} 次推理...")
        for i in range(self.warmup_runs):
            inference_time = self.single_inference()
            logger.info(f"预热 {i+1}/{self.warmup_runs}: {inference_time:.4f}s")
        logger.info("预热完成")
    
    def benchmark(self) -> Dict[str, Any]:
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
    
    def run_full_test(self) -> Dict[str, Any]:
        """运行完整的性能测试（包括初始化、预热和基准测试）"""
        logger.info("=" * 60)
        logger.info("FaceSwapper 帧处理速度性能测试")
        logger.info("=" * 60)
        logger.info(f"模型路径: {self.model_path}")
        logger.info(f"源图片: {self.source_image_path}")
        logger.info(f"目标图片: {self.target_image_path}")
        logger.info(f"预热次数: {self.warmup_runs}")
        logger.info(f"测试次数: {self.benchmark_runs}")
        logger.info("-" * 60)
        
        # 初始化
        self.setup()
        
        logger.info("-" * 60)
        
        # 预热
        self.warmup()
        
        logger.info("-" * 60)
        
        # 基准测试
        results = self.benchmark()
        
        # 输出结果
        logger.info("=" * 60)
        logger.info("性能测试结果:")
        logger.info(f"总运行次数: {results['total_runs']}")
        logger.info(f"平均处理时间: {results['mean_time']:.4f}s")
        logger.info(f"中位数处理时间: {results['median_time']:.4f}s")
        logger.info(f"最小处理时间: {results['min_time']:.4f}s")
        logger.info(f"最大处理时间: {results['max_time']:.4f}s")
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
        # 创建性能测试实例
        performance_test = FaceSwapperPerformanceTest(
            model_path=model_path,
            source_image_path=source_image_path,
            target_image_path=target_image_path,
            warmup_runs=5,
            benchmark_runs=50
        )
        
        # 运行性能测试
        results = performance_test.run_full_test()
        
        # 保存结果到文件
        results_copy = results.copy()
        results_copy.pop('all_times', None)  # 移除详细时间数据以减少文件大小
        
        output_file = 'face_swapper_performance_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_copy, f, indent=2, ensure_ascii=False)
        
        logger.info(f"性能测试结果已保存到 {output_file}")
        
    except Exception as e:
        logger.error(f"性能测试失败: {e}")
        raise


if __name__ == "__main__":
    main()