from functools import cached_property

from insightface.app.common import Face
from insightface.app.face_analysis import FaceAnalysis

from .types import Frame


class FaceAnalyzer:
    @cached_property
    def analyzer(self) -> FaceAnalysis:
        analyser = FaceAnalysis(
            name="buffalo_l",
            providers=[
                "CUDAExecutionProvider",
            ],
        )
        analyser.prepare(ctx_id=0, det_size=(640, 640))
        return analyser

    def get_one_face(self, frame: Frame) -> Face | None:
        face: list[Face] = self.analyzer.get(frame)

        def bbox_x(face: Face):
            return face.bbox[0] if face.bbox is not None else float("inf")

        try:
            return min(face, key=bbox_x)
        except ValueError:
            return None

    def get_many_faces(self, frame: Frame) -> list[Face] | None:
        try:
            return self.analyzer.get(frame)
        except IndexError:
            return None
