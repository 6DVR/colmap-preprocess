import os
import numpy as np
import numpy.typing as NPT
import numpy.testing as nptest
from pathlib import Path
from camera import getProjectionMatrix, getWorld2View2
from models import CameraInfo
from colmap_sceneinfo import get_cam_matrices, readColmapSceneInfo


def load_camera(scene_path: str, idx: int) -> CameraInfo:
    scene_info = readColmapSceneInfo(scene_path)
    return scene_info.cameras[idx]


def calc_scene_pose(
    camera_info: CameraInfo,
    zfar: float = 100.0,
    znear: float = 0.01,
    trans: np.ndarray = np.array([0.0, 0.0, 0.0]),
    scale: float = 1.0,
) -> NPT.NDArray[np.float64]:
    world_view_transform = getWorld2View2(camera_info.R, camera_info.T, trans, scale)

    projection_matrix = getProjectionMatrix(
        znear=znear, zfar=zfar, fovX=camera_info.FovX, fovY=camera_info.FovY
    ).transpose(1, 0)

    distort_matrix = np.array(
        [
            [camera_info.k1, camera_info.k2, camera_info.k3, 0],
            [camera_info.k4, camera_info.p1, camera_info.p2, 0],
        ]
    ).transpose()

    _pose_encoding = np.concatenate(
        [world_view_transform, projection_matrix, distort_matrix], axis=1
    )

    return _pose_encoding


def _list_scenes(dataset_dir: str) -> list[Path]:
    # Convert string path to Path object if needed
    base = Path(dataset_dir)

    # Get all second-tier folders and format as "number/random_string"
    second_tier = [
        f"{subfolder.parent.name}/{subfolder.name}"
        for number_folder in base.iterdir()
        if number_folder.is_dir()
        for subfolder in number_folder.iterdir()
        if subfolder.is_dir()
    ]

    # Print the results
    for folder in second_tier:
        print(folder)

    return second_tier


def run(path: str):
    # path = "/Users/eos/Downloads/datasets/mvimgnet"
    pose_folder = "poses"
    scenes = _list_scenes(path)
    with open(os.path.join(path, "scenes_list.txt"), "w") as f:
        for scene in scenes:
            scene_path = os.path.join(path, scene)
            ce, ci = get_cam_matrices(scene_path)
            if ce is None or ci is None:
                continue
            f.write(f"{scene}\n")
            scene_info = readColmapSceneInfo(scene_path)
            os.makedirs(os.path.join(scene_path, pose_folder), exist_ok=True)
            for camera in scene_info.cameras:
                pose_file_name = f"pe_{camera.image_name}.npy"
                pose_file_path = os.path.join(scene_path, pose_folder, pose_file_name)
                pose_encoding = calc_scene_pose(camera)
                np.save(pose_file_path, pose_encoding)

def check(path: str):
    # path = "/Users/eos/Downloads/datasets/mvimgnet"
    pose_folder = "poses"
    scenes = _list_scenes(path)

    scene_list = set(
        open(os.path.join(path, "scenes_list.txt"), "r").read().splitlines()
    )

    for scene in scenes:
        scene_path = os.path.join(path, scene)
        ce, ci = get_cam_matrices(scene_path)
        if ce is None or ci is None:
            continue
        assert scene in scene_list
        scene_info = readColmapSceneInfo(scene_path)
        for camera in scene_info.cameras:
            pose_file_name = f"pe_{camera.image_name}.npy"
            pose_file_path = os.path.join(scene_path, pose_folder, pose_file_name)
            assert os.path.exists(pose_file_path)
            pose_encoding_calc = calc_scene_pose(camera)
            pose_encoding_load = np.load(pose_file_path)
            nptest.assert_array_equal(pose_encoding_calc, pose_encoding_load)


if __name__ == "__main__":
    check()
