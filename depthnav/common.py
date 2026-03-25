import numpy as np
import torch
import torch as th
import torch.nn.functional as F
from typing import Optional, Tuple, List
from scipy.spatial.transform import Rotation as Rotation
from enum import Enum


class ExitCode(Enum):
    SUCCESS = 0
    ERROR = 1
    EARLY_STOP = 2
    NOT_FOUND = 3
    TIMEOUT = 4
    KEYBOARD_INTERRUPT = 5


def std_to_habitat(
    std_pos: Optional[torch.Tensor] = None,
    std_ori: Optional[torch.Tensor] = None,
    format="enu",
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """_summary_
        axes transformation, from std to habitat-sim

    Args:
        std_pos (_type_): _description_
        std_ori (_type_): _description_
        format (str, optional): _description_. Defaults to "enu".

    Returns:
        _type_: _description_
    """
    assert format in ["enu"]

    if std_ori is None:
        hab_ori = None
    else:
        hab_ori = std_ori.clone().detach().cpu().numpy() @ np.array(
            [[1, 0, 0, 0], [0, 0, 0, -1], [0, -1, 0, 0], [0, 0, 1, 0]]
        )

    if std_pos is None:
        hab_pos = None
    else:
        if len(std_pos.shape) == 1:
            hab_pos = (
                std_pos.clone().detach().cpu().unsqueeze(0).numpy()
                @ np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
            ).squeeze()
        elif std_pos.shape[1] == 3:
            hab_pos = std_pos.clone().detach().cpu().numpy() @ np.array(
                [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]
            )
        else:
            raise ValueError("std_pos shape error")

    return hab_pos, hab_ori


def habitat_to_std(
    habitat_pos: Optional[np.ndarray] = None,
    habitat_ori: Optional[np.ndarray] = None,
    format="enu",
):
    """_summary_
        axes transformation, from habitat-sim to std

    Args:
        habitat_pos (_type_): _description_
        habitat_ori (_type_): _description_
        format (str, optional): _description_. Defaults to "enu".

    Returns:
        _type_: _description_
    """
    # habitat_pos, habitat_ori = np.atleast_2d(habitat_pos), np.atleast_2d(habitat_ori)
    assert format in ["enu"]

    if habitat_pos is None:
        std_pos = None
    else:
        # assert habitat_pos.shape[1] == 3
        std_pos = th.as_tensor(
            np.atleast_2d(habitat_pos) @ np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]),
            dtype=th.float32,
        )
        # if len(habitat_pos.shape) == 1:
        #     std_pos = habitat_pos

    if habitat_ori is None:
        std_ori = None
    else:
        # assert habitat_ori.shape[1] == 4
        std_ori = th.from_numpy(
            np.atleast_2d(habitat_ori)
            @ np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1], [0, -1, 0, 0]])
        )
    return std_pos, std_ori


def obs_list2array(obs_dict: List, row: int, column: int):
    obs_indice = 0
    obs_array = []
    for i in range(column):
        obs_row = []
        for j in range(row):
            obs_row.append(obs_dict[obs_indice]["depth"])
            obs_indice += 1
        obs_array.append(np.hstack(obs_row))
    return np.vstack(obs_array)


def rgba2rgb(image):
    if isinstance(image, List):
        return [rgba2rgb(img) for img in image]
    else:
        return image[:, :, :3]


def observation_to_device(obs, device):
    return {k: v.to(device) for k, v in obs.items()}

def _quaternion_wxyz_to_matrix(q: th.Tensor) -> th.Tensor:
    q = F.normalize(q, dim=1, eps=1e-6)
    w, x, y, z = q.unbind(dim=1)

    row0 = th.stack([w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=1)
    row1 = th.stack([2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)], dim=1)
    row2 = th.stack([2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z], dim=1)
    return th.stack([row0, row1, row2], dim=1)


def _depth_gradient_geodesic(
    depth: th.Tensor, state: th.Tensor, target_direction: th.Tensor, min_depth: float = 0.1
) -> th.Tensor:
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError(f"Expected depth shape (B,1,H,W), got {tuple(depth.shape)}")

    depth = th.nan_to_num(depth.float(), nan=0.0, posinf=0.0, neginf=0.0)
    state = state.float()
    target_direction = F.normalize(target_direction.float(), dim=1, eps=1e-6)

    batch_size, _, height, width = depth.shape
    device = depth.device
    dtype = depth.dtype

    valid_mask = depth > min_depth
    smoothed_depth = F.avg_pool2d(depth, kernel_size=5, stride=1, padding=2)

    sobel_x = th.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3) / 8.0
    sobel_y = th.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3) / 8.0

    grad_x = F.conv2d(smoothed_depth, sobel_x, padding=1)
    grad_y = F.conv2d(smoothed_depth, sobel_y, padding=1)
    grad_mag = th.sqrt(grad_x.square() + grad_y.square())

    depth_scale = smoothed_depth.amax(dim=(2, 3), keepdim=True).clamp_min(min_depth)
    grad_scale = grad_mag.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    depth_score = (smoothed_depth / depth_scale).clamp(0.0, 1.0).squeeze(1)
    smooth_score = (1.0 - grad_mag / grad_scale).clamp(0.0, 1.0).squeeze(1)

    xs = th.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    ys = th.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    grid_x = xs.view(1, width).expand(height, width)
    grid_y = ys.view(height, 1).expand(height, width)
    rays_body = th.stack([th.ones_like(grid_x), -grid_x, -grid_y], dim=-1)
    rays_body = F.normalize(rays_body, dim=-1, eps=1e-6).unsqueeze(0).expand(batch_size, -1, -1, -1)

    rot_ib = _quaternion_wxyz_to_matrix(state[:, :4]).to(device=device, dtype=dtype)
    rays_inertial = th.einsum('bij,bhwj->bhwi', rot_ib, rays_body)
    alignment = (rays_inertial * target_direction[:, None, None, :]).sum(dim=-1).clamp_min(0.0)

    score = alignment * (0.7 * depth_score + 0.3 * smooth_score)
    score = score.masked_fill(~valid_mask.squeeze(1), float('-inf'))

    flat_score = score.flatten(1)
    best_idx = flat_score.argmax(dim=1)
    best_score = flat_score.gather(1, best_idx.unsqueeze(1)).squeeze(1)
    best_rays = rays_inertial.view(batch_size, -1, 3)[th.arange(batch_size, device=device), best_idx]

    invalid = ~th.isfinite(best_score) | (best_score <= 0.0)
    best_rays = th.where(invalid[:, None], target_direction, best_rays)
    return F.normalize(best_rays, dim=1, eps=1e-6)


def replace_geodesic_observation(obs, geodesic_mode: str = 'native'):
    if geodesic_mode == 'native':
        return obs

    obs = dict(obs)
    target_direction = F.normalize(obs['target'][:, :3], dim=1, eps=1e-6)

    if geodesic_mode == 'target':
        geodesic = target_direction
    elif geodesic_mode == 'zero':
        geodesic = th.zeros_like(target_direction)
    elif geodesic_mode == 'depth_gradient':
        geodesic = _depth_gradient_geodesic(obs['depth'], obs['state'], target_direction)
    else:
        raise ValueError(f'Unsupported geodesic_mode: {geodesic_mode}')

    obs['geodesic'] = geodesic
    obs['geodesic_valid'] = th.ones(
        (geodesic.shape[0], 1), device=geodesic.device, dtype=geodesic.dtype
    )
    return obs

