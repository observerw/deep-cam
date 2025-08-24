from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from threading import Semaphore
from typing import override

import cv2
import gfpgan
import insightface
import torch
from gfpgan.utils import GFPGANer
from insightface.app.common import Face
from insightface.model_zoo.inswapper import INSwapper

from deep_cam.face_analyzer import FaceAnalyzer
from deep_cam.types import Frame


class FrameProcessor(ABC):
    @abstractmethod
    def process_frame(self, frame: Frame) -> Frame: ...


@dataclass
class FaceEnhancer(FrameProcessor):
    model_path: Path

    @cached_property
    def face_analyzer(self) -> FaceAnalyzer:
        return FaceAnalyzer()

    @cached_property
    def enhancer(self) -> GFPGANer:
        return gfpgan.GFPGANer(
            model_path=self.model_path.as_posix(),
            upscale=1,
            device=torch.device("cuda"),
        )

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


@dataclass
class FaceSwapper(FrameProcessor):
    model_path: Path
    target_image_path: Path

    @cached_property
    def face_analyzer(self) -> FaceAnalyzer:
        return FaceAnalyzer()

    @cached_property
    def target_face(self) -> Face:
        target_face_image = cv2.imread(self.target_image_path.as_posix())
        assert isinstance(target_face_image, Frame)
        face = self.face_analyzer.get_one_face(target_face_image)
        if not face:
            raise ValueError("No face found in target face image")
        return face

    @cached_property
    def swapper(self) -> INSwapper:
        model = insightface.model_zoo.get_model(
            self.model_path.as_posix(),
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        assert isinstance(model, INSwapper)
        return model

    def process_frame(self, frame: Frame) -> Frame:
        # color correction
        temp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        source_face = self.face_analyzer.get_one_face(temp_frame)
        if not source_face:
            return frame

        swapped_frame = self.swapper.get(
            img=frame,
            target_face=self.target_face,
            source_face=source_face,
            paste_back=True,
        )
        assert isinstance(swapped_frame, Frame)
        return swapped_frame
