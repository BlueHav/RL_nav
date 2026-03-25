import numpy as np
import pytest
import torch as th

spaces = pytest.importorskip('gymnasium', reason='gymnasium is required').spaces

from depthnav.common import replace_geodesic_observation
from depthnav.policies.multi_input_policy import MultiInputPolicy
from depthnav.topology_guidance import TopologyGuidance


HEIGHT = 72
WIDTH = 128


def make_obs_space():
    return spaces.Dict(
        {
            'state': spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
            'target': spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            'geodesic': spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            'depth': spaces.Box(low=0.0, high=np.inf, shape=(1, HEIGHT, WIDTH), dtype=np.float32),
        }
    )


def make_depth(openings=None, batch_size=1, blocked_depth=0.2, open_depth=6.0):
    depth = th.full((batch_size, 1, HEIGHT, WIDTH), blocked_depth, dtype=th.float32)
    openings = openings or []
    for start, end in openings:
        depth[:, :, HEIGHT // 4 : 3 * HEIGHT // 4, start:end] = open_depth
    return depth


def make_pose(batch_size=1, position=None):
    if position is None:
        position = th.zeros((batch_size, 3), dtype=th.float32)
    quaternion = th.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=th.float32).expand(batch_size, -1).clone()
    target_direction = th.tensor([[1.0, 0.0, 0.0]], dtype=th.float32).expand(batch_size, -1).clone()
    state = th.cat([quaternion, th.zeros((batch_size, 3), dtype=th.float32)], dim=1)
    target = th.cat([target_direction, th.ones((batch_size, 1), dtype=th.float32)], dim=1)
    return position, quaternion, target_direction, state, target


def make_obs(depth, state, target):
    return {
        'depth': depth.clone(),
        'state': state.clone(),
        'target': target.clone(),
    }


def test_topology_guidance_extracts_openings_and_returns_direction():
    guidance = TopologyGuidance(min_sector_width_px=4)
    depth = make_depth(openings=[(40, 82)])
    position, quaternion, target_direction, _, _ = make_pose()

    guidance.update(depth, position, quaternion, target_direction)
    direction, valid = guidance.best_direction()

    assert valid.shape == (1, 1)
    assert valid[0, 0].item() is True
    assert len(guidance.graphs[0]) >= 1
    assert guidance.graphs[0][0].support > 0
    assert guidance.graphs[0][0].angular_width > 0.0
    assert direction.shape == (1, 3)
    assert th.allclose(direction.norm(dim=1), th.ones(1), atol=1e-4)
    assert direction[0, 0] > 0.5


def test_topology_guidance_keeps_node_identity_on_small_pose_changes():
    guidance = TopologyGuidance(min_sector_width_px=4)
    depth = make_depth(openings=[(30, 70)])
    position, quaternion, target_direction, _, _ = make_pose()

    guidance.update(depth, position, quaternion, target_direction)
    direction_1, valid_1 = guidance.best_direction()
    node_id = guidance.graphs[0][0].node_id

    next_position = position + th.tensor([[0.25, 0.0, 0.0]], dtype=th.float32)
    shifted_depth = make_depth(openings=[(32, 72)])
    guidance.update(shifted_depth, next_position, quaternion, target_direction)
    direction_2, valid_2 = guidance.best_direction()

    assert valid_1[0, 0].item() is True
    assert valid_2[0, 0].item() is True
    assert node_id in {node.node_id for node in guidance.graphs[0]}
    cosine = th.nn.functional.cosine_similarity(direction_1, direction_2, dim=1)
    assert cosine[0] > 0.95


def test_topology_guidance_downweights_closed_nodes_and_reset_clears_graph():
    guidance = TopologyGuidance(
        min_sector_width_px=4,
        unmatched_decay=0.6,
        closed_node_decay=0.2,
        min_node_confidence=0.05,
        max_staleness=3,
    )
    open_depth = make_depth(openings=[(44, 84)])
    closed_depth = make_depth(openings=[])
    position, quaternion, target_direction, _, _ = make_pose()

    guidance.update(open_depth, position, quaternion, target_direction)
    initial_confidence = guidance.graphs[0][0].confidence

    for _ in range(3):
        guidance.update(closed_depth, position, quaternion, target_direction)

    _, valid = guidance.best_direction()
    remaining_confidence = guidance.graphs[0][0].confidence if guidance.graphs[0] else 0.0
    assert remaining_confidence < initial_confidence
    assert valid[0, 0].item() is False or len(guidance.graphs[0]) == 0

    guidance.reset(batch_size=1)
    assert guidance.graphs == [[]]


def test_topology_only_steers_from_currently_visible_nodes():
    guidance = TopologyGuidance(min_sector_width_px=4)
    position, quaternion, target_direction, state, target = make_pose()

    open_obs = make_obs(make_depth(openings=[(44, 84)]), state, target)
    closed_obs = make_obs(make_depth(openings=[]), state, target)
    replace_geodesic_observation(
        open_obs,
        geodesic_mode='topology',
        guidance=guidance,
        positions=position,
        quaternions=quaternion,
        target_direction=target_direction,
    )
    closed = replace_geodesic_observation(
        closed_obs,
        geodesic_mode='topology',
        guidance=guidance,
        positions=position,
        quaternions=quaternion,
        target_direction=target_direction,
    )

    assert closed['geodesic_valid'][0, 0].item() == 0.0


def test_replace_geodesic_observation_topology_preserves_interface_and_falls_back():
    guidance = TopologyGuidance(min_sector_width_px=4)
    position, quaternion, target_direction, state, target = make_pose()
    obs = make_obs(make_depth(openings=[]), state, target)

    replaced = replace_geodesic_observation(
        obs,
        geodesic_mode='topology',
        guidance=guidance,
        positions=position,
        quaternions=quaternion,
        target_direction=target_direction,
    )

    assert replaced['geodesic'].shape == (1, 3)
    assert replaced['geodesic_valid'].shape == (1, 1)
    assert replaced['geodesic_valid'][0, 0].item() == 0.0
    assert th.allclose(replaced['geodesic'].norm(dim=1), th.ones(1), atol=1e-4)


def test_topology_geodesic_stays_close_to_depth_gradient_baseline():
    guidance = TopologyGuidance(min_sector_width_px=4)
    position, quaternion, target_direction, state, target = make_pose()
    obs = make_obs(make_depth(openings=[(52, 96)]), state, target)

    depth_gradient = replace_geodesic_observation(
        obs,
        geodesic_mode='depth_gradient',
        quaternions=quaternion,
        target_direction=target_direction,
    )
    topology = replace_geodesic_observation(
        obs,
        geodesic_mode='topology',
        guidance=guidance,
        positions=position,
        quaternions=quaternion,
        target_direction=target_direction,
    )

    cosine = th.nn.functional.cosine_similarity(
        depth_gradient['geodesic'], topology['geodesic'], dim=1
    )
    assert cosine[0] > 0.95


def test_topology_geodesic_replacement_stays_policy_compatible():
    guidance = TopologyGuidance(min_sector_width_px=4)
    position, quaternion, target_direction, state, target = make_pose(batch_size=2)
    depth = make_depth(openings=[(36, 74)], batch_size=2)
    obs = make_obs(depth, state, target)
    replaced = replace_geodesic_observation(
        obs,
        geodesic_mode='topology',
        guidance=guidance,
        positions=position,
        quaternions=quaternion,
        target_direction=target_direction,
    )

    policy = MultiInputPolicy(
        observation_space=make_obs_space(),
        net_arch={
            'enable_geodesic_aux': True,
            'geo_aux_head_hidden': 64,
            'recurrent': {'class': 'LayerNormGRUCell', 'kwargs': {'hidden_size': 192}},
            'mlp_layer': [4],
        },
        activation_fn='leaky_relu',
        output_activation_fn='identity',
        feature_extractor_class='StateTargetGeodesicSpatialExtractor',
        feature_extractor_kwargs={
            'activation_fn': 'leaky_relu',
            'net_arch': {
                'state': {'mlp_layer': [32], 'ln': True},
                'target': {'mlp_layer': [32], 'ln': True},
                'geodesic': {'mlp_layer': [32], 'ln': True},
                'output_hw': [3, 5],
                'depth_out_channels': 128,
                'fusion_channels': 192,
            },
        },
        device='cpu',
    )

    latent = th.zeros((2, 192), dtype=th.float32)
    actions, next_latent, aux = policy(replaced, latent, return_aux=True)

    assert actions.shape == (2, 4)
    assert next_latent.shape == (2, 192)
    assert aux['geodesic_direction'].shape == (2, 3)
