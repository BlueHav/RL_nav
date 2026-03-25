
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch as th
import torch.nn.functional as F


def _quaternion_wxyz_to_matrix(q: th.Tensor) -> th.Tensor:
    q = F.normalize(q, dim=1, eps=1e-6)
    w, x, y, z = q.unbind(dim=1)

    row0 = th.stack(
        [
            w * w + x * x - y * y - z * z,
            2 * (x * y - w * z),
            2 * (x * z + w * y),
        ],
        dim=1,
    )
    row1 = th.stack(
        [
            2 * (x * y + w * z),
            w * w - x * x + y * y - z * z,
            2 * (y * z - w * x),
        ],
        dim=1,
    )
    row2 = th.stack(
        [
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            w * w - x * x - y * y + z * z,
        ],
        dim=1,
    )
    return th.stack([row0, row1, row2], dim=1)


def _extract_runs(mask: th.Tensor) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, is_open in enumerate(mask.tolist()):
        if is_open and start is None:
            start = index
        elif not is_open and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


@dataclass
class TopologyCandidate:
    position: th.Tensor
    direction: th.Tensor
    clearance: float
    angular_width: float
    support: float
    confidence: float


@dataclass
class TopologyNode:
    node_id: int
    position: th.Tensor
    direction: th.Tensor
    clearance: float
    angular_width: float
    support: float
    confidence: float
    age: int
    last_seen: int


@dataclass
class TopologyExtractionMeta:
    column_score: th.Tensor
    traversability: th.Tensor
    smoothed_depth: th.Tensor


class TopologyGuidance:
    def __init__(
        self,
        min_depth: float = 0.1,
        min_clearance: float = 0.75,
        open_score_threshold: float = 0.22,
        min_sector_width_px: int = 5,
        match_angle_threshold_deg: float = 18.0,
        match_clearance_threshold: float = 1.5,
        ema_alpha: float = 0.45,
        unmatched_decay: float = 0.88,
        closed_node_decay: float = 0.45,
        min_node_confidence: float = 0.15,
        max_staleness: int = 8,
        max_nodes: int = 8,
        score_width_norm: float = 0.12,
        score_clearance_norm: float = 3.0,
        score_support_norm: float = 10.0,
    ) -> None:
        self.min_depth = min_depth
        self.min_clearance = min_clearance
        self.open_score_threshold = open_score_threshold
        self.min_sector_width_px = min_sector_width_px
        self.match_angle_threshold = th.deg2rad(
            th.tensor(match_angle_threshold_deg)
        ).item()
        self.match_clearance_threshold = match_clearance_threshold
        self.ema_alpha = ema_alpha
        self.unmatched_decay = unmatched_decay
        self.closed_node_decay = closed_node_decay
        self.min_node_confidence = min_node_confidence
        self.max_staleness = max_staleness
        self.max_nodes = max_nodes
        self.score_width_norm = score_width_norm
        self.score_clearance_norm = score_clearance_norm
        self.score_support_norm = score_support_norm

        self._batch_size = 0
        self.reset()

    def reset(self, batch_size: Optional[int] = None) -> None:
        if batch_size is not None:
            self._batch_size = int(batch_size)
        self.graphs: List[List[TopologyNode]] = [[] for _ in range(self._batch_size)]
        self._step_index = [0 for _ in range(self._batch_size)]
        self._last_selected_direction: List[Optional[th.Tensor]] = [
            None for _ in range(self._batch_size)
        ]
        self._current_positions: Optional[th.Tensor] = None
        self._current_quaternions: Optional[th.Tensor] = None
        self._current_target_direction: Optional[th.Tensor] = None
        self._current_image_shape: Optional[Tuple[int, int]] = None
        self._next_node_id = 0

    def _ensure_batch_size(self, batch_size: int) -> None:
        if self._batch_size != batch_size:
            self.reset(batch_size=batch_size)

    @staticmethod
    def _safe_normalize(vec: th.Tensor, dim: int = -1) -> th.Tensor:
        return F.normalize(vec, dim=dim, eps=1e-6)

    @staticmethod
    def _make_camera_rays(
        quaternions: th.Tensor, height: int, width: int, dtype: th.dtype
    ) -> th.Tensor:
        device = quaternions.device
        xs = th.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        ys = th.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        grid_x = xs.view(1, width).expand(height, width)
        grid_y = ys.view(height, 1).expand(height, width)
        rays_body = th.stack([th.ones_like(grid_x), -grid_x, -grid_y], dim=-1)
        rays_body = F.normalize(rays_body, dim=-1, eps=1e-6)
        rot = _quaternion_wxyz_to_matrix(quaternions.to(dtype=dtype))
        return th.einsum('bij,hwj->bhwi', rot, rays_body)

    def _extract_candidates_batch(
        self,
        depth: th.Tensor,
        positions: th.Tensor,
        quaternions: th.Tensor,
    ) -> Tuple[List[List[TopologyCandidate]], List[TopologyExtractionMeta]]:
        depth = th.nan_to_num(depth.float(), nan=0.0, posinf=0.0, neginf=0.0)
        batch_size, _, height, width = depth.shape
        device = depth.device
        dtype = depth.dtype

        smoothed_depth = F.avg_pool2d(depth, kernel_size=5, stride=1, padding=2)
        valid_mask = smoothed_depth > self.min_depth

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

        depth_scale = smoothed_depth.amax(dim=(2, 3), keepdim=True).clamp_min(
            self.min_depth
        )
        grad_scale = grad_mag.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        depth_score = (smoothed_depth / depth_scale).clamp(0.0, 1.0).squeeze(1)
        smooth_score = (1.0 - grad_mag / grad_scale).clamp(0.0, 1.0).squeeze(1)

        ys = th.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        row_weight = (1.0 - 0.35 * ys.abs()).view(1, height, 1)
        traversability = (
            valid_mask.squeeze(1).float()
            * (0.75 * depth_score + 0.25 * smooth_score)
            * row_weight
        )

        column_score = traversability.amax(dim=1)
        column_score = F.avg_pool1d(
            column_score.unsqueeze(1), kernel_size=5, stride=1, padding=2
        ).squeeze(1)
        column_clearance = smoothed_depth.squeeze(1).amax(dim=1)
        open_mask = (column_score > self.open_score_threshold) & (
            column_clearance > self.min_clearance
        )

        rays_world = self._make_camera_rays(quaternions, height, width, dtype)
        candidates: List[List[TopologyCandidate]] = []
        metas: List[TopologyExtractionMeta] = []

        for batch_index in range(batch_size):
            runs = _extract_runs(open_mask[batch_index])
            env_candidates: List[TopologyCandidate] = []
            for start, end in runs:
                width_px = end - start
                if width_px < self.min_sector_width_px:
                    continue

                segment_scores = column_score[batch_index, start:end].clamp_min(1e-6)
                segment_indices = th.arange(start, end, device=device, dtype=dtype)
                center_col = int(
                    th.round(
                        (segment_scores * segment_indices).sum() / segment_scores.sum()
                    ).item()
                )
                center_col = max(0, min(center_col, width - 1))

                row_scores = traversability[batch_index, :, center_col].clamp_min(1e-6)
                row_indices = th.arange(height, device=device, dtype=dtype)
                center_row = int(
                    th.round((row_scores * row_indices).sum() / row_scores.sum()).item()
                )
                center_row = max(0, min(center_row, height - 1))

                clearance_patch = smoothed_depth[
                    batch_index,
                    0,
                    max(0, center_row - 1) : min(height, center_row + 2),
                    start:end,
                ]
                clearance = float(clearance_patch.amax().item())
                direction = self._safe_normalize(
                    rays_world[batch_index, center_row, center_col], dim=0
                )
                position = positions[batch_index] + direction * clearance
                angular_width = float(width_px / max(width, 1))
                support = float(width_px)
                avg_segment_score = float(segment_scores.mean().item())
                confidence = min(1.5, 0.7 * avg_segment_score + 0.8 * angular_width)

                env_candidates.append(
                    TopologyCandidate(
                        position=position.detach().clone(),
                        direction=direction.detach().clone(),
                        clearance=clearance,
                        angular_width=angular_width,
                        support=support,
                        confidence=confidence,
                    )
                )

            env_candidates.sort(key=lambda node: node.confidence, reverse=True)
            candidates.append(env_candidates)
            metas.append(
                TopologyExtractionMeta(
                    column_score=column_score[batch_index].detach().clone(),
                    traversability=traversability[batch_index].detach().clone(),
                    smoothed_depth=smoothed_depth[batch_index, 0].detach().clone(),
                )
            )

        return candidates, metas

    @staticmethod
    def _angle_between(vec_a: th.Tensor, vec_b: th.Tensor) -> float:
        dot = th.clamp(th.dot(vec_a, vec_b), -1.0, 1.0)
        return float(th.acos(dot).item())

    def _project_direction_to_image(
        self,
        direction_world: th.Tensor,
        quaternion: th.Tensor,
        height: int,
        width: int,
    ) -> Optional[Tuple[int, int, float]]:
        rot = _quaternion_wxyz_to_matrix(quaternion.unsqueeze(0))[0]
        direction_body = rot.transpose(0, 1) @ direction_world
        forward = float(direction_body[0].item())
        if forward <= 1e-4:
            return None

        grid_x = float((-direction_body[1] / forward).item())
        grid_y = float((-direction_body[2] / forward).item())
        if abs(grid_x) > 1.2 or abs(grid_y) > 1.2:
            return None

        col = int(round(((max(-1.0, min(1.0, grid_x)) + 1.0) * 0.5) * (width - 1)))
        row = int(round(((max(-1.0, min(1.0, grid_y)) + 1.0) * 0.5) * (height - 1)))
        return row, col, forward

    def _update_env_graph(
        self,
        env_index: int,
        candidates: List[TopologyCandidate],
        meta: TopologyExtractionMeta,
        position: th.Tensor,
        quaternion: th.Tensor,
    ) -> None:
        nodes = self.graphs[env_index]
        current_step = self._step_index[env_index]
        for node in nodes:
            node.age += 1

        matched_candidate_indices = set()
        for node in nodes:
            predicted_vec = node.position - position
            predicted_clearance = float(predicted_vec.norm().item())
            if predicted_clearance <= 1e-4:
                continue
            predicted_direction = self._safe_normalize(predicted_vec, dim=0)

            best_index = None
            best_cost = None
            for candidate_index, candidate in enumerate(candidates):
                if candidate_index in matched_candidate_indices:
                    continue
                angle_error = self._angle_between(predicted_direction, candidate.direction)
                clearance_error = abs(predicted_clearance - candidate.clearance)
                if angle_error > self.match_angle_threshold:
                    continue
                if clearance_error > self.match_clearance_threshold:
                    continue
                cost = (
                    angle_error / max(self.match_angle_threshold, 1e-6)
                    + clearance_error / max(self.match_clearance_threshold, 1e-6)
                    - 0.15 * candidate.confidence
                )
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_index = candidate_index

            if best_index is None:
                projection = self._project_direction_to_image(
                    predicted_direction,
                    quaternion,
                    meta.traversability.shape[0],
                    meta.traversability.shape[1],
                )
                node.confidence *= self.unmatched_decay
                if projection is not None:
                    row, col, _ = projection
                    local_score = float(meta.traversability[row, col].item())
                    local_depth = float(meta.smoothed_depth[row, col].item())
                    if local_score < self.open_score_threshold or local_depth < (
                        0.75 * predicted_clearance
                    ):
                        node.confidence *= self.closed_node_decay
                continue

            candidate = candidates[best_index]
            matched_candidate_indices.add(best_index)
            predicted_direction = self._safe_normalize(node.position - position, dim=0)
            node.position = th.lerp(
                node.position, candidate.position.to(node.position.device), self.ema_alpha
            ).detach()
            node.direction = self._safe_normalize(
                th.lerp(
                    predicted_direction,
                    candidate.direction.to(node.direction.device),
                    self.ema_alpha,
                ),
                dim=0,
            ).detach()
            node.clearance = (
                (1.0 - self.ema_alpha) * predicted_clearance
                + self.ema_alpha * candidate.clearance
            )
            node.angular_width = (
                (1.0 - self.ema_alpha) * node.angular_width
                + self.ema_alpha * candidate.angular_width
            )
            node.support = min(
                node.support + 0.5 * candidate.support, 2.0 * self.score_support_norm
            )
            node.confidence = min(
                1.5,
                (1.0 - self.ema_alpha) * node.confidence
                + self.ema_alpha * candidate.confidence
                + 0.1,
            )
            node.last_seen = current_step

        for candidate_index, candidate in enumerate(candidates):
            if candidate_index in matched_candidate_indices:
                continue
            nodes.append(
                TopologyNode(
                    node_id=self._next_node_id,
                    position=candidate.position.detach().clone(),
                    direction=candidate.direction.detach().clone(),
                    clearance=candidate.clearance,
                    angular_width=candidate.angular_width,
                    support=candidate.support,
                    confidence=candidate.confidence,
                    age=1,
                    last_seen=current_step,
                )
            )
            self._next_node_id += 1

        pruned_nodes: List[TopologyNode] = []
        for node in nodes:
            staleness = current_step - node.last_seen
            predicted_clearance = float((node.position - position).norm().item())
            if node.confidence < self.min_node_confidence:
                continue
            if staleness > self.max_staleness:
                continue
            if predicted_clearance < 0.25:
                continue
            pruned_nodes.append(node)

        pruned_nodes.sort(
            key=lambda node: (
                node.confidence,
                node.support,
                node.angular_width,
                -float((node.position - position).norm().item()),
            ),
            reverse=True,
        )
        self.graphs[env_index] = pruned_nodes[: self.max_nodes]

    @th.no_grad()
    def update(
        self,
        depth: th.Tensor,
        positions: th.Tensor,
        quaternions: th.Tensor,
        target_direction: th.Tensor,
    ) -> None:
        batch_size = depth.shape[0]
        self._ensure_batch_size(batch_size)

        device = depth.device
        positions = positions.to(device=device, dtype=depth.dtype)
        quaternions = self._safe_normalize(
            quaternions.to(device=device, dtype=depth.dtype), dim=1
        )
        target_direction = self._safe_normalize(
            target_direction.to(device=device, dtype=depth.dtype), dim=1
        )

        candidates, metas = self._extract_candidates_batch(depth, positions, quaternions)
        self._current_positions = positions.detach().clone()
        self._current_quaternions = quaternions.detach().clone()
        self._current_target_direction = target_direction.detach().clone()
        self._current_image_shape = (depth.shape[2], depth.shape[3])

        for env_index in range(batch_size):
            self._step_index[env_index] += 1
            self._update_env_graph(
                env_index,
                candidates[env_index],
                metas[env_index],
                positions[env_index],
                quaternions[env_index],
            )

    @th.no_grad()
    def best_direction(self) -> Tuple[th.Tensor, th.Tensor]:
        if (
            self._current_positions is None
            or self._current_quaternions is None
            or self._current_target_direction is None
        ):
            raise RuntimeError('TopologyGuidance.best_direction() called before update().')

        if self._current_image_shape is None:
            raise RuntimeError('TopologyGuidance image shape is unavailable.')

        height, width = self._current_image_shape
        device = self._current_positions.device
        batch_size = self._current_positions.shape[0]
        directions: List[th.Tensor] = []
        valid_mask = th.zeros((batch_size, 1), dtype=th.bool, device=device)

        for env_index in range(batch_size):
            position = self._current_positions[env_index]
            quaternion = self._current_quaternions[env_index]
            target_direction = self._current_target_direction[env_index]
            forward = _quaternion_wxyz_to_matrix(quaternion.unsqueeze(0))[0, :, 0]
            turn_reference = self._last_selected_direction[env_index]
            if turn_reference is None:
                turn_reference = forward

            best_score = None
            best_direction = target_direction

            for node in self.graphs[env_index]:
                predicted_vec = node.position - position
                predicted_clearance = float(predicted_vec.norm().item())
                if predicted_clearance <= 1e-4:
                    continue

                predicted_direction = self._safe_normalize(predicted_vec, dim=0)
                projection = self._project_direction_to_image(
                    predicted_direction,
                    quaternion,
                    height=height,
                    width=width,
                )
                if projection is None:
                    continue

                goal_alignment = float(th.dot(predicted_direction, target_direction).item())
                clearance_score = float(
                    th.tanh(
                        th.tensor(
                            predicted_clearance / max(self.score_clearance_norm, 1e-6),
                            device=device,
                        )
                    ).item()
                )
                width_score = min(
                    node.angular_width / max(self.score_width_norm, 1e-6), 1.5
                )
                persistence_score = min(
                    node.support / max(self.score_support_norm, 1e-6), 1.5
                ) * min(node.confidence, 1.5)
                turning_cost = 1.0 - float(
                    th.dot(
                        predicted_direction,
                        self._safe_normalize(turn_reference, dim=0),
                    ).item()
                )
                staleness_penalty = (
                    self._step_index[env_index] - node.last_seen
                ) / max(self.max_staleness, 1)

                score = (
                    1.35 * goal_alignment
                    + 0.70 * clearance_score
                    + 0.40 * width_score
                    + 0.55 * persistence_score
                    - 0.35 * turning_cost
                    - 0.45 * staleness_penalty
                )

                if best_score is None or score > best_score:
                    best_score = score
                    best_direction = predicted_direction

            if best_score is not None:
                valid_mask[env_index, 0] = True
                best_direction = self._safe_normalize(best_direction, dim=0)
                self._last_selected_direction[env_index] = best_direction.detach().clone()
            directions.append(best_direction.detach().clone())

        return th.stack(directions, dim=0), valid_mask
