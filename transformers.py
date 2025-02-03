import numpy as np
import numpy.typing as NPT
import math


def qvec2rotmat(qvec: NPT.NDArray[np.float64]) -> NPT.NDArray[np.float64]:
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R: NPT.NDArray[np.float64]):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def fov2focal(fov: float, pixels: float):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal: float, pixels: float):
    return 2 * math.atan(pixels / (2 * focal))


# def PILtoTorch(image: Image, resolution: tuple[int, int]):
#     resized_image_PIL = image.resize(resolution)
#     resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0  # type: ignore
#     if len(resized_image.shape) == 3:
#         return resized_image.permute(2, 0, 1)
#     else:
#         return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


# def build_covariance_from_scaling_rotation(
#     scaling: torch.Tensor, scaling_modifier: float, rotation: torch.Tensor
# ):
#     L = build_scaling_rotation(scaling_modifier * scaling, rotation)
#     actual_covariance = L @ L.transpose(1, 2)
#     symm = strip_lowerdiag(actual_covariance)
#     return symm


# def build_scaling_rotation(s: torch.Tensor, r: torch.Tensor):
#     L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
#     R = build_rotation(r)

#     L[:, 0, 0] = s[:, 0]
#     L[:, 1, 1] = s[:, 1]
#     L[:, 2, 2] = s[:, 2]

#     L = R @ L
#     return L


# def build_rotation(r: torch.Tensor):
#     norm = torch.sqrt(
#         r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
#     )

#     q = r / norm[:, None]

#     R = torch.zeros((q.size(0), 3, 3), device="cuda")

#     r = q[:, 0]
#     x = q[:, 1]
#     y = q[:, 2]
#     z = q[:, 3]

#     R[:, 0, 0] = 1 - 2 * (y * y + z * z)
#     R[:, 0, 1] = 2 * (x * y - r * z)
#     R[:, 0, 2] = 2 * (x * z + r * y)
#     R[:, 1, 0] = 2 * (x * y + r * z)
#     R[:, 1, 1] = 1 - 2 * (x * x + z * z)
#     R[:, 1, 2] = 2 * (y * z - r * x)
#     R[:, 2, 0] = 2 * (x * z - r * y)
#     R[:, 2, 1] = 2 * (y * z + r * x)
#     R[:, 2, 2] = 1 - 2 * (x * x + y * y)
#     return R


# def strip_lowerdiag(L: torch.Tensor):
#     uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

#     uncertainty[:, 0] = L[:, 0, 0]
#     uncertainty[:, 1] = L[:, 0, 1]
#     uncertainty[:, 2] = L[:, 0, 2]
#     uncertainty[:, 3] = L[:, 1, 1]
#     uncertainty[:, 4] = L[:, 1, 2]
#     uncertainty[:, 5] = L[:, 2, 2]
#     return uncertainty


# def inverse_sigmoid(x: torch.Tensor):
#     return torch.log(x / (1 - x))


# def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
#     """
#     Returns torch.sqrt(torch.max(0, x))
#     but with a zero subgradient where x is 0.
#     Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
#     """
#     ret = torch.zeros_like(x)
#     positive_mask = x > 0
#     ret[positive_mask] = torch.sqrt(x[positive_mask])
#     return ret


# def rotation2quad(matrix: torch.Tensor) -> torch.Tensor:
#     """
#     Convert rotations given as rotation matrices to quaternions.
#     Args:
#         matrix: Rotation matrices as tensor of shape (..., 3, 3).
#     Returns:
#         quaternions with real part first, as tensor of shape (..., 4).
#     Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
#     """
#     if matrix.size(-1) != 3 or matrix.size(-2) != 3:
#         raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

#     # if not isinstance(matrix, torch.Tensor):
#     #     matrix = torch.tensor(matrix).cuda()

#     batch_dim = matrix.shape[:-2]
#     m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
#         matrix.reshape(batch_dim + (9,)), dim=-1
#     )

#     q_abs = _sqrt_positive_part(
#         torch.stack(
#             [
#                 1.0 + m00 + m11 + m22,
#                 1.0 + m00 - m11 - m22,
#                 1.0 - m00 + m11 - m22,
#                 1.0 - m00 - m11 + m22,
#             ],
#             dim=-1,
#         )
#     )

#     # we produce the desired quaternion multiplied by each of r, i, j, k
#     quat_by_rijk = torch.stack(
#         [
#             # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
#             #  `int`.
#             torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
#             # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
#             #  `int`.
#             torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
#             # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
#             #  `int`.
#             torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
#             # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
#             #  `int`.
#             torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
#         ],
#         dim=-2,
#     )

#     # We floor here at 0.1 but the exact level is not important; if q_abs is small,
#     # the candidate won't be picked.
#     flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
#     quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

#     # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
#     # forall i; we pick the best-conditioned one (with the largest denominator)

#     return quat_candidates[
#         F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
#     ].reshape(batch_dim + (4,))


# def quad2rotation(q: torch.Tensor):
#     """
#     Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.
#     Args:
#         quad (tensor, batch_size*4): quaternion.
#     Returns:
#         rot_mat (tensor, batch_size*3*3): rotation.
#     """
#     norm = torch.sqrt(
#         q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]
#     )
#     q = q / norm[:, None]
#     rot = torch.zeros((q.size(0), 3, 3)).to(q)
#     r = q[:, 0]
#     x = q[:, 1]
#     y = q[:, 2]
#     z = q[:, 3]
#     rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
#     rot[:, 0, 1] = 2 * (x * y - r * z)
#     rot[:, 0, 2] = 2 * (x * z + r * y)
#     rot[:, 1, 0] = 2 * (x * y + r * z)
#     rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
#     rot[:, 1, 2] = 2 * (y * z - r * x)
#     rot[:, 2, 0] = 2 * (x * z - r * y)
#     rot[:, 2, 1] = 2 * (y * z + r * x)
#     rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
#     return rot


# def get_tensor_from_camera(RT: torch.Tensor):
#     """
#     Convert transformation matrix to quaternion and translation.
#     """

#     rot = RT[:3, :3].unsqueeze(0).detach()
#     quat = rotation2quad(rot).squeeze()
#     tran = RT[:3, 3].detach()

#     return torch.cat([quat, tran])


def get_expon_lr_func(
    lr_init: float,
    lr_final: float,
    lr_delay_steps: int = 0,
    lr_delay_mult: float = 1.0,
    max_steps: int = 1000000,
):
    """
    Copied from Plenoxels
    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step: int) -> float:
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


# def quadmultiply(q1: torch.Tensor, q2: torch.Tensor):
#     """
#     Multiply two quaternions together using quaternion arithmetic
#     """
#     # Extract scalar and vector parts of the quaternions
#     w1, x1, y1, z1 = q1.unbind(dim=-1)
#     w2, x2, y2, z2 = q2.unbind(dim=-1)
#     # Calculate the quaternion product
#     result_quaternion = torch.stack(
#         [
#             w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
#             w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
#             w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
#             w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
#         ],
#         dim=-1,
#     )

#     return result_quaternion
