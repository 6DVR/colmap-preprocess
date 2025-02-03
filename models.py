from enum import Enum, auto
import numpy as np
import numpy.typing as NPT
from typing import NamedTuple, Optional
from PIL.Image import Image
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, computed_field

from transformers import focal2fov


class BasicPointCloud(NamedTuple):
    points: NPT.NDArray[np.float64]
    colors: NPT.NDArray[np.float64]
    normals: NPT.NDArray[np.float64]


class CameraExtrinsic(NamedTuple):
    id: int
    qvec: NPT.NDArray[np.float64]
    tvec: NPT.NDArray[np.float64]
    camera_id: int
    name: str
    xys: NPT.ArrayLike
    point3D_ids: NPT.ArrayLike


class CameraType(Enum):
    OPENCV = auto()
    OPENCV_FISHEYE = auto()
    EQUIRECTANGULAR = auto()
    PINHOLE = auto()
    SIMPLE_PINHOLE = auto()
    UNKNOWN = auto()

class CameraModel(Enum):
    SIMPLE_PINHOLE = (0, 3)
    PINHOLE = (1, 4)
    SIMPLE_RADIAL = (2, 4)
    RADIAL = (3, 5)
    OPENCV = (4, 8)
    OPENCV_FISHEYE = (5, 8)
    FULL_OPENCV = (6, 12)
    FOV = (7, 5)
    SIMPLE_RADIAL_FISHEYE = (8, 4)
    RADIAL_FISHEYE = (9, 5)
    THIN_PRISM_FISHEYE = (10, 12)

    def __init__(self, id: int, params: int) -> None:
        super().__init__()
        self.id = id
        self.params = params


class CameraIntrinsic(NamedTuple):
    id: int
    model: CameraModel
    width: int
    height: int
    params: NPT.NDArray[np.float64]


class CameraInfo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    uid: int
    R: NPT.NDArray[np.float64]
    T: NPT.NDArray[np.float64]
    f1_x: float
    f1_y: float
    cx: float = 0.0
    cy: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    omega: float = 0.0
    image: Image
    image_path: str
    image_name: str
    width: int
    height: int
    video_timestamp: Optional[float] = None
    camera_type: CameraType = CameraType.OPENCV

    @computed_field
    @property
    def FovX(self) -> float:
        return focal2fov(self.f1_x, self.width)

    @computed_field
    @property
    def FovY(self) -> float:
        return focal2fov(self.f1_y, self.height)



@dataclass
class TrainingSceneInfo:
    point_cloud: BasicPointCloud | None
    cameras: list[CameraInfo]
    ply_path: str | None
