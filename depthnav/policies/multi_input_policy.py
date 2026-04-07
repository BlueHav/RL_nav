import torch as th
import torch.nn.functional as F
from torch import nn
from typing import Type, Optional, Dict, Any, Union, List
from gymnasium import spaces

from .extractors import (
    FeatureExtractor,
    StateExtractor,
    StateTargetExtractor,
    ImageExtractor,
    StateImageExtractor,
    StateTargetImageExtractor,
    StateTargetGeodesicSpatialExtractor,
)
from .mlp_policy import MlpPolicy


class LayerNormGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_ih = nn.Linear(input_size, 3 * hidden_size)
        self.linear_hh = nn.Linear(hidden_size, 3 * hidden_size)
        self.ln_ih = nn.LayerNorm(3 * hidden_size)
        self.ln_hh = nn.LayerNorm(3 * hidden_size)

    def forward(self, x, h):
        gi = self.ln_ih(self.linear_ih(x))
        gh = self.ln_hh(self.linear_hh(h))
        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)

        r = th.sigmoid(i_r + h_r)
        z = th.sigmoid(i_z + h_z)
        n = th.tanh(i_n + r * h_n)
        h_next = (1 - z) * n + z * h
        return h_next


class MultiInputPolicy(MlpPolicy):
    """
    Builds an actor policy network with specifications from a dictionary.
    """

    feature_extractor_alias = {
        # "flatten": FlattenExtractor,
        "StateExtractor": StateExtractor,
        "ImageExtractor": ImageExtractor,
        "StateTargetExtractor": StateTargetExtractor,
        "StateImageExtractor": StateImageExtractor,
        "StateTargetImageExtractor": StateTargetImageExtractor,
        "StateTargetGeodesicSpatialExtractor": StateTargetGeodesicSpatialExtractor,
    }
    recurrent_alias = {
        "GRUCell": th.nn.GRUCell,
        "LayerNormGRUCell": LayerNormGRUCell,
    }

    def __init__(
        self,
        observation_space: spaces.Space,
        net_arch: Dict[str, List[int]],
        activation_fn: Union[str, nn.Module],
        output_activation_fn: Union[str, nn.Module],
        feature_extractor_class: Type[FeatureExtractor],
        output_activation_kwargs: Optional[Dict[str, Any]] = None,
        feature_extractor_kwargs: Optional[Dict[str, Any]] = None,
        device: th.device = "cuda",
    ):
        if isinstance(feature_extractor_class, str):
            feature_extractor_class = self.feature_extractor_alias[
                feature_extractor_class
            ]
        feature_extractor_kwargs = feature_extractor_kwargs or {}

        if isinstance(activation_fn, str):
            aux_activation_fn = self.activation_fn_alias[activation_fn]
        else:
            aux_activation_fn = activation_fn

        # get the size of features_dim before initializing MlpPolicy
        feature_extractor = feature_extractor_class(
            observation_space, **feature_extractor_kwargs
        )
        feature_norm = nn.LayerNorm(feature_extractor.features_dim)

        # add recurrent layer after feature_extractor
        _is_recurrent = False
        if net_arch.get("recurrent", None) is not None:
            _is_recurrent = True
            rnn_setting = net_arch.get("recurrent")
            rnn_class = rnn_setting.get("class", "LayerNormGRUCell")
            kwargs = rnn_setting.get("kwargs", {})

            if isinstance(rnn_class, str):
                rnn_class = self.recurrent_alias[rnn_class]

            recurrent_extractor = rnn_class(
                input_size=feature_extractor.features_dim, **kwargs
            )
            in_dim = kwargs.get("hidden_size")
        else:
            in_dim = feature_extractor.features_dim

        super().__init__(
            in_dim,
            net_arch,
            activation_fn,
            output_activation_fn,
            output_activation_kwargs,
            device,
        )

        self.feature_extractor = feature_extractor.to(device)
        self.feature_norm = feature_norm.to(device)
        if _is_recurrent:
            self._is_recurrent = True
            self._latent_dim = in_dim
            self.recurrent_extractor = recurrent_extractor.to(device)

        self.enable_geodesic_aux = (
            net_arch.get("enable_geodesic_aux", True)
            and "geodesic" in getattr(observation_space, "spaces", {})
        )
        if self.enable_geodesic_aux:
            aux_hidden_dim = net_arch.get("geo_aux_head_hidden", 64)
            self.geodesic_aux_head = nn.Sequential(
                nn.Linear(in_dim, aux_hidden_dim),
                aux_activation_fn(),
                nn.Linear(aux_hidden_dim, 3),
            ).to(device)

        self.to(device)

    def forward(self, obs, latent=None, return_aux=False):
        features = self.feature_extractor(obs)
        features = self.feature_norm(features)
        head_input = features
        next_latent = latent
        if self.is_recurrent:
            next_latent = self.recurrent_extractor(features, latent)
            head_input = next_latent

        actions = super().forward(head_input)

        if return_aux:
            aux = {}
            if hasattr(self, "geodesic_aux_head"):
                aux["geodesic_direction"] = F.normalize(
                    self.geodesic_aux_head(head_input), dim=1, eps=1e-6
                )
            if self.is_recurrent:
                return actions, next_latent, aux
            return actions, None, aux

        if self.is_recurrent:
            return actions, next_latent
        return actions
