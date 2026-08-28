import os
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, Sequence

import torch
import torch.nn as nn

from instinct_rl.modules.mlp import MlpModel
from instinct_rl.utils.utils import get_subobs_by_components, get_subobs_size


class AMEEncoder(nn.Module):
    """AME-style map encoder for visual observations.

    The encoder removes one map/image component from the flat observation, encodes it with a small CNN and a
    proprioception-conditioned attention query, then appends the latent back to the remaining observation terms.
    Unlike lidar/radar xyz-map variants, this implementation does not append xy coordinate channels unless explicitly
    requested with ``add_xy=True``.
    """

    def __init__(self, input_segments: Dict[str, tuple], block_configs: Dict[str, object]):
        super().__init__()

        cfg = deepcopy(block_configs)
        cfg.pop("class_name", None)

        self.input_segments = input_segments
        self.map_component_name = cfg.pop("map_component_name", "depth_image")
        self.output_name = cfg.pop("output_name", "ame_latent")
        self.output_size = cfg.pop("output_size", 64)
        self.mha_dim = cfg.pop("mha_dim", 64)
        self.num_heads = cfg.pop("num_heads", 16)
        self.cnn_downsample = cfg.pop("cnn_downsample", True)
        self.attach_global = cfg.pop("attach_global", False)
        self.add_xy = cfg.pop("add_xy", False)
        self.image_layout = cfg.pop("image_layout", "auto")
        self.query_latest_proprio = cfg.pop("query_latest_proprio", False)
        self.proprio_history_length = cfg.pop("proprio_history_length", 1)
        nonlinearity = cfg.pop("nonlinearity", "ReLU")
        normlayer = cfg.pop("normlayer", "BatchNorm2d")
        global_hidden_sizes = cfg.pop("global_hidden_sizes", [256, 128])

        if cfg:
            print(f"AMEEncoder.__init__ got unexpected arguments, which will be ignored: {list(cfg.keys())}")

        if self.map_component_name not in self.input_segments:
            raise KeyError(f"AMEEncoder map component '{self.map_component_name}' is not in {list(input_segments)}")

        self.map_shape = tuple(self.input_segments[self.map_component_name])
        self.map_channels, self.map_height, self.map_width = self._infer_bchw_shape(self.map_shape, self.image_layout)
        cnn_in_channels = self.map_channels + (2 if self.add_xy else 0)

        self.proprio_component_names = [name for name in self.input_segments.keys() if name != self.map_component_name]
        self.proprio_dim = get_subobs_size(self.input_segments, self.proprio_component_names)
        if self.proprio_dim <= 0:
            raise ValueError("AMEEncoder requires at least one non-map observation component for the attention query.")
        self.query_proprio_dim = self._query_dim(self.proprio_dim)

        if isinstance(nonlinearity, str):
            nonlinearity = getattr(nn, nonlinearity)
        if isinstance(normlayer, str):
            normlayer = getattr(nn, normlayer)

        self.cnn_output_dim = self.mha_dim
        if self.cnn_downsample:
            self.map_cnn = nn.Sequential(
                nn.Conv2d(cnn_in_channels, 16, kernel_size=5, padding=2, stride=2),
                nonlinearity(),
                normlayer(16),
                nn.Conv2d(16, self.cnn_output_dim, kernel_size=3, padding=1),
                nonlinearity(),
                normlayer(self.cnn_output_dim),
            )
        else:
            self.map_cnn = nn.Sequential(
                nn.Conv2d(cnn_in_channels, 16, kernel_size=5, padding=2),
                nonlinearity(),
                normlayer(16),
                nn.Conv2d(16, self.cnn_output_dim, kernel_size=5, padding=2),
                nonlinearity(),
                normlayer(self.cnn_output_dim),
            )

        self.proprio_embedding = nn.Linear(self.query_proprio_dim, self.mha_dim)
        self.mha = nn.MultiheadAttention(embed_dim=self.mha_dim, num_heads=self.num_heads, batch_first=True)

        latent_dim = self.mha_dim
        if self.attach_global:
            self.global_encoder = MlpModel(
                self.mha_dim,
                global_hidden_sizes,
                output_size=self.mha_dim,
                nonlinearity="ELU",
            )
            self.query_projector = nn.Linear(self.mha_dim * 2, self.mha_dim)
            latent_dim += self.mha_dim
        else:
            self.global_encoder = None
            self.query_projector = None

        self.output_projection = (
            nn.Identity() if latent_dim == self.output_size else nn.Linear(latent_dim, self.output_size)
        )
        self.build_output_segment()

    def _query_dim(self, proprio_dim: int) -> int:
        if not self.query_latest_proprio:
            return proprio_dim
        if self.proprio_history_length < 1 or proprio_dim % self.proprio_history_length != 0:
            raise ValueError(
                "Cannot split latest proprio frame: "
                f"proprio_dim={proprio_dim}, proprio_history_length={self.proprio_history_length}"
            )
        return proprio_dim // self.proprio_history_length

    def _query_proprio(self, proprio_obs: torch.Tensor) -> torch.Tensor:
        if not self.query_latest_proprio:
            return proprio_obs
        return proprio_obs[:, -self.query_proprio_dim :]

    def share_terrain_encoder_with(self, other: "AMEEncoder") -> None:
        """Share map encoder modules and BN statistics with another AME encoder."""
        if self.map_shape != other.map_shape:
            raise ValueError(f"Cannot share AME terrain encoder with different map shapes: {self.map_shape} vs {other.map_shape}")
        if self.map_channels != other.map_channels or self.add_xy != other.add_xy:
            raise ValueError("Cannot share AME terrain encoder with different channel configuration.")
        self.map_cnn = other.map_cnn
        self.mha = other.mha
        self.global_encoder = other.global_encoder
        self.query_projector = other.query_projector
        self.output_projection = other.output_projection

    def build_output_segment(self):
        self.output_segment = OrderedDict(
            [(name, shape) for name, shape in self.input_segments.items() if name != self.map_component_name]
        )
        self.output_segment[self.output_name] = (self.output_size,)
        self.numel_output = get_subobs_size(self.output_segment)
        return self.output_segment

    @staticmethod
    def _infer_bchw_shape(shape: Sequence[int], image_layout: str):
        if len(shape) == 2:
            return 1, shape[0], shape[1]

        if len(shape) == 3:
            if image_layout == "chw" or (image_layout == "auto" and shape[0] <= 16 and shape[-1] > 16):
                return shape[0], shape[1], shape[2]
            if image_layout in ("hwc", "auto"):
                return shape[2], shape[0], shape[1]

        if len(shape) == 4:
            if image_layout == "thwc" or (image_layout == "auto" and shape[-1] <= 16):
                return shape[0] * shape[3], shape[1], shape[2]
            if image_layout in ("tchw", "auto"):
                return shape[0] * shape[1], shape[2], shape[3]

        raise ValueError(f"Unsupported image shape/layout for AMEEncoder: shape={shape}, image_layout={image_layout}")

    def _map_to_bchw(self, map_obs: torch.Tensor):
        b = map_obs.shape[0]
        image = map_obs.reshape(b, *self.map_shape)

        if len(self.map_shape) == 2:
            image = image.unsqueeze(1)
        elif len(self.map_shape) == 3:
            if self.image_layout == "chw" or (
                self.image_layout == "auto" and self.map_shape[0] <= 16 and self.map_shape[-1] > 16
            ):
                pass
            else:
                image = image.permute(0, 3, 1, 2)
        elif len(self.map_shape) == 4:
            if self.image_layout == "thwc" or (self.image_layout == "auto" and self.map_shape[-1] <= 16):
                image = image.permute(0, 1, 4, 2, 3).reshape(b, self.map_channels, self.map_height, self.map_width)
            else:
                image = image.reshape(b, self.map_channels, self.map_height, self.map_width)

        if self.add_xy:
            y = torch.linspace(-1.0, 1.0, self.map_height, device=image.device, dtype=image.dtype).view(1, 1, -1, 1)
            x = torch.linspace(-1.0, 1.0, self.map_width, device=image.device, dtype=image.dtype).view(1, 1, 1, -1)
            xy = torch.cat(
                (
                    x.expand(b, 1, self.map_height, self.map_width),
                    y.expand(b, 1, self.map_height, self.map_width),
                ),
                dim=1,
            )
            image = torch.cat((image, xy), dim=1)
        return image

    def run_encoder(self, flat_input: torch.Tensor):
        map_obs = get_subobs_by_components(flat_input, [self.map_component_name], self.input_segments)
        proprio_obs = get_subobs_by_components(flat_input, self.proprio_component_names, self.input_segments)

        image = self._map_to_bchw(map_obs)
        cnn_features = self.map_cnn(image)
        local_features = cnn_features.flatten(2).transpose(1, 2)

        proprio_embedding = self.proprio_embedding(self._query_proprio(proprio_obs))
        if self.attach_global:
            global_features = self.global_encoder(local_features)
            global_features_max, _ = torch.max(global_features, dim=1)
            proprio_embedding = self.query_projector(torch.cat((global_features_max, proprio_embedding), dim=-1))

        mha_output, _ = self.mha(
            query=proprio_embedding.unsqueeze(1),
            key=local_features,
            value=local_features,
        )
        latent = mha_output.squeeze(1)
        if self.attach_global:
            latent = torch.cat((global_features_max, latent), dim=-1)
        latent = self.output_projection(latent)

        outputs = []
        for component_name in self.output_segment.keys():
            if component_name == self.output_name:
                outputs.append(latent)
            else:
                outputs.append(get_subobs_by_components(flat_input, [component_name], self.input_segments))
        return torch.cat(outputs, dim=-1)

    def forward(self, flat_input: torch.Tensor) -> torch.Tensor:
        leading_dim = flat_input.shape[:-1]
        flat_input_2d = flat_input.reshape(-1, flat_input.shape[-1])
        output = self.run_encoder(flat_input_2d)
        return output.reshape(*leading_dim, -1)

    def __str__(self):
        return (
            f"AMEEncoder(map={self.map_component_name}, map_shape={self.map_shape}, "
            f"latent={self.output_size}, query_latest_proprio={self.query_latest_proprio}, add_xy={self.add_xy})"
        )

    def export_as_onnx(self, flat_input, filedir: str, *args, **kwargs):
        self.eval()
        with torch.no_grad():
            exported_program = torch.onnx.export(
                self,
                flat_input,
                "/tmp/ame_encoder.onnx",
                input_names=["input"],
                output_names=["output"],
                dynamo=True,
                opset_version=15,
            )
            exported_program.save(os.path.join(filedir, "ame_encoder.onnx"))
            print(f"Exported ame_encoder to {os.path.join(filedir, 'ame_encoder.onnx')}")
