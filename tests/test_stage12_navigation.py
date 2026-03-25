import numpy as np
import pytest
import torch as th

spaces = pytest.importorskip('gymnasium', reason='gymnasium is required').spaces

from depthnav.policies.extractors import StateTargetGeodesicSpatialExtractor
from depthnav.policies.multi_input_policy import MultiInputPolicy


@pytest.mark.skipif(
    pytest.importorskip('habitat_sim', reason='habitat_sim is required for scene_manager import') is None,
    reason='habitat_sim is unavailable',
)
def test_build_fmm_speed_smoothstep_monotonic():
    from depthnav.envs.scene_manager import SceneManager

    robot_radius = 0.3
    safe_radius = 0.6
    slow_speed = 0.1
    tsdf = np.array([0.0, robot_radius, 0.45, safe_radius, 0.8], dtype=np.float32)
    occupancy = np.zeros_like(tsdf, dtype=np.uint8)

    speed = SceneManager.build_fmm_speed(
        tsdf,
        occupancy,
        robot_radius=robot_radius,
        safe_radius=safe_radius,
        slow_speed=slow_speed,
    )

    assert speed[0] == pytest.approx(slow_speed)
    assert speed[1] == pytest.approx(slow_speed)
    assert speed[3] == pytest.approx(1.0)
    assert speed[4] == pytest.approx(1.0)
    assert np.all(np.diff(speed) >= -1e-6)


def make_obs_space():
    return spaces.Dict(
        {
            'state': spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
            'target': spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            'geodesic': spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            'depth': spaces.Box(low=0.0, high=np.inf, shape=(1, 72, 128), dtype=np.float32),
        }
    )


def make_obs(batch_size=2):
    geodesic = th.tensor([[1.0, 0.0, 0.0], [0.3, 0.4, 0.5]], dtype=th.float32)
    geodesic = th.nn.functional.normalize(geodesic, dim=1)
    return {
        'state': th.randn(batch_size, 7),
        'target': th.randn(batch_size, 4),
        'geodesic': geodesic[:batch_size],
        'depth': th.rand(batch_size, 1, 72, 128) + 0.25,
    }


def test_spatial_extractor_output_shape():
    extractor = StateTargetGeodesicSpatialExtractor(
        make_obs_space(),
        net_arch={
            'state': {'mlp_layer': [32], 'ln': True},
            'target': {'mlp_layer': [32], 'ln': True},
            'geodesic': {'mlp_layer': [32], 'ln': True},
            'output_hw': [3, 5],
            'depth_out_channels': 128,
            'fusion_channels': 192,
        },
        activation_fn='leaky_relu',
    )

    features = extractor(make_obs())
    assert features.shape == (2, 192)


def test_multi_input_policy_return_aux_shapes():
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

    obs = make_obs()
    latent = th.zeros((2, 192), dtype=th.float32)
    actions, next_latent, aux = policy(obs, latent, return_aux=True)

    assert actions.shape == (2, 4)
    assert next_latent.shape == (2, 192)
    assert 'geodesic_direction' in aux
    assert aux['geodesic_direction'].shape == (2, 3)
    norms = aux['geodesic_direction'].norm(dim=1)
    assert th.allclose(norms, th.ones_like(norms), atol=1e-4)
