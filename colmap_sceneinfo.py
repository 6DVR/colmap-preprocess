import os
import sys
import numpy as np
from PIL import Image

from colmap_reader import (
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)
from models import (
    CameraExtrinsic,
    CameraInfo,
    CameraIntrinsic,
    CameraModel,
    CameraType,
    TrainingSceneInfo,
)
from transformers import qvec2rotmat


def get_cam_matrices(
    data_path: str,
) -> tuple[dict[int, CameraExtrinsic] | None, dict[int, CameraIntrinsic] | None]:
    sparse_folder = os.path.join(data_path, "sparse/0")
    if os.path.exists(os.path.join(sparse_folder, "images.bin")):
        cameras_extrinsic_file = os.path.join(sparse_folder, "images.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    elif os.path.exists(os.path.join(sparse_folder, "images.txt")):
        cameras_extrinsic_file = os.path.join(sparse_folder, "images.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    else:
        cam_extrinsics = None

    if os.path.exists(os.path.join(sparse_folder, "cameras.bin")):
        cameras_intrinsic_file = os.path.join(sparse_folder, "cameras.bin")
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    elif os.path.exists(os.path.join(sparse_folder, "cameras.txt")):
        cameras_intrinsic_file = os.path.join(sparse_folder, "cameras.txt")
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    else:
        cam_intrinsics = None

    return cam_extrinsics, cam_intrinsics


def readColmapSceneInfo(data_path: str):
    cam_extrinsics, cam_intrinsics = get_cam_matrices(data_path)
    if cam_extrinsics is None or cam_intrinsics is None:
        raise ValueError("No camera matrices found in the sparse folder")

    images_folder = os.path.join(data_path, "images")

    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=images_folder,
    )

    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    scene_info = TrainingSceneInfo(
        point_cloud=None,
        cameras=cam_infos,
        ply_path=None,
    )
    return scene_info


def readColmapCameras(
    cam_extrinsics: dict[int, CameraExtrinsic],
    cam_intrinsics: dict[int, CameraIntrinsic],
    images_folder: str,
):
    cam_infos: list[CameraInfo] = []

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        uid = extr.camera_id

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        height = intr.height
        width = intr.width
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)  # type: ignore
        match intr.model:
            case CameraModel.SIMPLE_PINHOLE:
                cam_info = CameraInfo(
                    uid=uid,
                    R=R,
                    T=T,
                    f1_x=intr.params[0],
                    f1_y=intr.params[0],
                    cx=intr.params[1],
                    cy=intr.params[2],
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                )
            case CameraModel.PINHOLE:
                cam_info = CameraInfo(
                    uid=uid,
                    R=R,
                    T=T,
                    f1_x=intr.params[0],
                    f1_y=intr.params[1],
                    cx=intr.params[2],
                    cy=intr.params[3],
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                )
            case CameraModel.SIMPLE_RADIAL:
                cam_info = CameraInfo(
                    uid=uid,
                    R=R,
                    T=T,
                    f1_x=intr.params[0],
                    f1_y=intr.params[0],
                    cx=intr.params[1],
                    cy=intr.params[2],
                    k1=intr.params[3],
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                )
            case CameraModel.RADIAL:
                cam_info = CameraInfo(
                    uid=uid,
                    R=R,
                    T=T,
                    f1_x=intr.params[0],
                    f1_y=intr.params[0],
                    cx=intr.params[1],
                    cy=intr.params[2],
                    k1=intr.params[3],
                    k2=intr.params[4],
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                )
            case CameraModel.OPENCV:
                cam_info = CameraInfo(
                    uid=uid,
                    R=R,
                    T=T,
                    f1_x=intr.params[0],
                    f1_y=intr.params[1],
                    cx=intr.params[2],
                    cy=intr.params[3],
                    k1=intr.params[4],
                    k2=intr.params[5],
                    p1=intr.params[6],
                    p2=intr.params[7],
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                )
            case CameraModel.OPENCV_FISHEYE:
                cam_info = CameraInfo(
                    uid=uid,
                    R=R,
                    T=T,
                    f1_x=intr.params[0],
                    f1_y=intr.params[1],
                    cx=intr.params[2],
                    cy=intr.params[3],
                    k1=intr.params[4],
                    k2=intr.params[5],
                    k3=intr.params[6],
                    k4=intr.params[7],
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                    camera_type=CameraType.OPENCV_FISHEYE,
                )
            case CameraModel.FULL_OPENCV:
                raise NotImplementedError(
                    f"{intr.model} camera model not supported yet!"
                )
                cam_info = CameraInfo(
                    uid=uid,
                    R=R,
                    T=T,
                    f1_x=intr.params[0],
                    f1_y=intr.params[1],
                    cx=intr.params[2],
                    cy=intr.params[3],
                    k1=intr.params[4],
                    k2=intr.params[5],
                    p1=intr.params[6],
                    p2=intr.params[7],
                    k3=intr.params[8],
                    k4=intr.params[9],
                    k5=intr.params[10],
                    k6=intr.params[11],
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                    camera_type=CameraType.UNKNOWN,
                )
            case CameraModel.FOV:
                raise NotImplementedError(
                    f"{intr.model} camera model not supported yet!"
                )
                cam_info = CameraInfo(
                    uid=uid,
                    R=R,
                    T=T,
                    f1_x=intr.params[0],
                    f1_y=intr.params[1],
                    cx=intr.params[2],
                    cy=intr.params[3],
                    omega=intr.params[4],
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                    camera_type=CameraType.UNKNOWN,
                )
            case CameraModel.SIMPLE_RADIAL_FISHEYE:
                cam_info = CameraInfo(
                    uid=uid,
                    R=R,
                    T=T,
                    f1_x=intr.params[0],
                    f1_y=intr.params[0],
                    cx=intr.params[1],
                    cy=intr.params[2],
                    k1=intr.params[3],
                    k2=0,
                    k3=0,
                    k4=0,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                    camera_type=CameraType.OPENCV_FISHEYE,
                )
            case CameraModel.RADIAL_FISHEYE:
                cam_info = CameraInfo(
                    uid=uid,
                    R=R,
                    T=T,
                    f1_x=intr.params[0],
                    f1_y=intr.params[0],
                    cx=intr.params[1],
                    cy=intr.params[2],
                    k1=intr.params[3],
                    k2=intr.params[4],
                    k3=0,
                    k4=0,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                    camera_type=CameraType.OPENCV_FISHEYE,
                )
            case _:
                raise KeyError(
                    "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
                )

        # cam_info = CameraInfo(
        #     uid=uid,
        #     R=R,
        #     T=T,
        #     FovY=FovY,
        #     FovX=FovX,
        #     image=image,
        #     image_path=image_path,
        #     image_name=image_name,
        #     width=width,
        #     height=height,
        # )

        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos
