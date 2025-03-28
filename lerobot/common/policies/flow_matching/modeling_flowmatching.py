import torch
import torch.nn.functional as F
from torch import nn, Tensor
import einops
import torchvision
from collections import deque
from lerobot.common.policies import PreTrainedPolicy
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.flow_matching.configuration_flowmatching import FlowMatchingConfig
from flow_matching.path import CondOTProbPath
from lerobot.common.policies.utils import (
    get_device_from_parameters,
    populate_queues,
)
import math
import numpy as np
from typing import Callable

class FlowMatchingPolicy(PreTrainedPolicy):
    config_class = FlowMatchingConfig
    name = "flowmatching"

    def __init__(self, config: FlowMatchingConfig, dataset_stats: dict = None):
        super().__init__(config)
        self.config = config
        
        # Initialize normalization
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(config.output_features, config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize(config.output_features, config.normalization_mapping, dataset_stats)

        # Initialize RGB encoders for image observations if needed
        global_cond_dim = config.robot_state_feature.shape[0]
        if config.image_features:
            num_images = len(config.image_features)
            if config.use_separate_rgb_encoder_per_camera:
                encoders = [FlowMatchingRGBEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = FlowMatchingRGBEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if config.env_state_feature:
            global_cond_dim += config.env_state_feature.shape[0]

        # Initialize flow components
        self.velocity_model = ConditionalUNet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)
        self._init_path()

        # Queues for observations/actions
        self.reset()

    def _init_path(self):
        """Initialize the probability path based on config"""
        if self.config.path_type == "CondOTProbPath":
            self.path = CondOTProbPath()
        else:
            raise ValueError(f"Unsupported path type: {self.config.path_type}")

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and concatenate them along with the state vector."""
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        global_cond_feats = [batch["observation.state"]]
        
        # Extract image features if they exist
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                # Combine batch and sequence dims while rearranging to make the camera index dimension first
                images_per_camera = einops.rearrange(batch["observation.images"], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ]
                )
                # Separate batch and sequence dims back out
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                # Combine batch, sequence, and "which camera" dims
                img_features = self.rgb_encoder(
                    einops.rearrange(batch["observation.images"], "b s n ... -> (b s n) ...")
                )
                # Separate batch dim and sequence dim back out
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch["observation.environment_state"])

        # Concatenate features then flatten
        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def compute_loss(self, batch: dict) -> torch.Tensor:
        # Normalize inputs and targets
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        
        # Prepare conditioning
        global_cond = self._prepare_global_conditioning(batch)
        
        # Sample from path
        x1 = batch["action"]  # Ground truth actions
        x0 = torch.randn_like(x1)  # Sample from noise distribution
        t = torch.rand(x1.size(0), device=x1.device)  # Random time steps
        
        # Get path components
        sample = self.path.sample(t=t, x0=x0, x1=x1)
        
        # Predict velocity field
        pred_v = self.velocity_model(sample.x_t, t, global_cond=global_cond)
        
        # Compute MSE loss
        loss = F.mse_loss(pred_v, sample.dx_t)
        return loss

    def generate_actions(self, batch: dict) -> torch.Tensor:
        """Generate actions using ODE solver"""
        global_cond = self._prepare_global_conditioning(batch)
        batch_size = batch["observation.state"].shape[0]
        
        # Initial noise sample
        x = torch.randn(
            (batch_size, self.config.horizon, self.config.action_feature.shape[0]),
            device=global_cond.device
        )
        
        # ODE solver steps
        dt = 1.0 / self.config.num_inference_steps
        for t in torch.linspace(0, 1, self.config.num_inference_steps):
            v = self.velocity_model(x, t.expand(x.size(0)), global_cond=global_cond)
            x = x + v * dt
        
        return self.unnormalize_outputs({"action": x})["action"]

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.
        
        Similar to DiffusionPolicy, this method:
        - Caches n_obs_steps of observations
        - Generates horizon steps of actions using flow matching
        - Returns n_action_steps worth of actions
        """
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues["action"]) == 0:
            # stack n latest observations from the queue
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            actions = self.generate_actions(batch)

            # Unnormalize outputs
            actions = self.unnormalize_outputs({"action": actions})["action"]

            self._queues["action"].extend(actions.transpose(0, 1))

        action = self._queues["action"].popleft()
        return action

    # Inherit other methods from DiffusionPolicy as needed
    # (e.g., _prepare_global_conditioning, reset, select_action, etc.)


class ConditionalUNet1d(nn.Module):
    """1D UNet for flow matching with FiLM conditioning"""
    
    def __init__(self, config: FlowMatchingConfig, global_cond_dim: int):
        super().__init__()
        self.config = config
        
        # Time embedding
        self.time_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.time_embed_dim),
            nn.Linear(config.time_embed_dim, config.time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.time_embed_dim * 4, config.time_embed_dim)
        )
        
        # Combined conditioning (time + global)
        cond_dim = config.time_embed_dim + global_cond_dim
        
        # Define UNet architecture
        self.down_modules, self.up_modules = self._create_unet_layers(cond_dim)
        self.mid_modules = self._create_mid_layers(cond_dim)
        self.final_conv = self._create_final_layers()

    def _create_unet_layers(self, cond_dim):
        # Similar to diffusion UNet but adapted for flow matching
        in_out = [(self.config.action_feature.shape[0], self.config.down_dims[0])]
        in_out += list(zip(self.config.down_dims[:-1], self.config.down_dims[1:]))

        # Down blocks
        down_modules = nn.ModuleList()
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResBlock1d(dim_in, dim_out, cond_dim,
                                    self.config.kernel_size, self.config.n_groups,
                                    self.config.use_film_scale_modulation),
                ConditionalResBlock1d(dim_out, dim_out, cond_dim,
                                    self.config.kernel_size, self.config.n_groups,
                                    self.config.use_film_scale_modulation),
                nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity()
            ]))

        # Up blocks
        up_modules = nn.ModuleList()
        for idx, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = idx >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResBlock1d(dim_in*2, dim_out, cond_dim,
                                   self.config.kernel_size, self.config.n_groups,
                                   self.config.use_film_scale_modulation),
                ConditionalResBlock1d(dim_out, dim_out, cond_dim,
                                   self.config.kernel_size, self.config.n_groups,
                                   self.config.use_film_scale_modulation),
                nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity()
            ]))
        
        return down_modules, up_modules

    def forward(self, x: Tensor, t: Tensor, global_cond: Tensor = None):
        # x: (B, T, D), t: (B,), global_cond: (B, C)
        x = einops.rearrange(x, 'b t d -> b d t')
        
        # Time embedding
        t_embed = self.time_encoder(t)
        if global_cond is not None:
            cond = torch.cat([t_embed, global_cond], dim=-1)
        else:
            cond = t_embed

        # UNet processing
        skips = []
        for res1, res2, down in self.down_modules:
            x = res1(x, cond)
            x = res2(x, cond)
            skips.append(x)
            x = down(x)

        for mid in self.mid_modules:
            x = mid(x, cond)

        for res1, res2, up in self.up_modules:
            x = torch.cat([x, skips.pop()], dim=1)
            x = res1(x, cond)
            x = res2(x, cond)
            x = up(x)

        x = self.final_conv(x)
        return einops.rearrange(x, 'b d t -> b t d')

    def _create_mid_layers(self, cond_dim):
        """Create middle layers of the UNet"""
        mid_modules = nn.ModuleList([
            ConditionalResBlock1d(
                self.config.down_dims[-1], self.config.down_dims[-1], cond_dim,
                self.config.kernel_size, self.config.n_groups,
                self.config.use_film_scale_modulation
            ),
            ConditionalResBlock1d(
                self.config.down_dims[-1], self.config.down_dims[-1], cond_dim,
                self.config.kernel_size, self.config.n_groups,
                self.config.use_film_scale_modulation
            ),
        ])
        return mid_modules

    def _create_final_layers(self):
        """Create final convolutional layers"""
        return nn.Sequential(
            nn.Conv1d(self.config.down_dims[0], self.config.down_dims[0], self.config.kernel_size, padding=self.config.kernel_size//2),
            nn.GroupNorm(self.config.n_groups, self.config.down_dims[0]),
            nn.Mish(),
            nn.Conv1d(self.config.down_dims[0], self.config.action_feature.shape[0], 1)
        )

class FlowMatchingRGBEncoder(nn.Module):
    """ResNet-based visual encoder with Spatial Softmax (same as diffusion)"""
    
    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        # Same implementation as DiffusionRgbEncoder
        if config.crop_shape:
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            self.maybe_random_crop = (torchvision.transforms.RandomCrop(config.crop_shape) 
                                    if config.crop_is_random else self.center_crop)
        
        backbone = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        if config.use_group_norm:
            self.backbone = _replace_submodules(
                self.backbone,
                lambda m: isinstance(m, nn.BatchNorm2d),
                lambda _: nn.GroupNorm(32, _.num_features)
            )
        
        # Spatial softmax
        dummy_input = torch.randn(1, 3, *config.crop_shape)
        with torch.no_grad():
            features = self.backbone(dummy_input)
        self.spatial_softmax = SpatialSoftmax(
            features.shape[1:], 
            num_kp=config.spatial_softmax_num_keypoints
        )
        
        self.projection = nn.Sequential(
            nn.Linear(config.spatial_softmax_num_keypoints*2, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        if hasattr(self, 'maybe_random_crop'):
            if self.training:
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)
        
        features = self.backbone(x)
        keypoints = self.spatial_softmax(features)
        return self.projection(keypoints.flatten(1))

class ConditionalResBlock1d(nn.Module):
    """FiLM-conditioned residual block"""
    
    def __init__(self, in_dim, out_dim, cond_dim, 
                 kernel_size=3, n_groups=8, use_scale=True):
        super().__init__()
        self.use_scale = use_scale
        
        # Main layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size, padding=kernel_size//2),
            nn.GroupNorm(n_groups, out_dim),
            nn.Mish()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size, padding=kernel_size//2),
            nn.GroupNorm(n_groups, out_dim),
            nn.Mish()
        )
        
        # Conditioning
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, out_dim*2 if use_scale else out_dim),
            nn.Mish()
        )
        
        # Residual
        self.residual = nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x: Tensor, cond: Tensor):
        residual = self.residual(x)
        
        # First convolution
        x = self.conv1(x)
        
        # FiLM conditioning
        film_params = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_scale:
            scale, bias = torch.chunk(film_params, 2, dim=1)
            x = x * (1 + scale) + bias
        else:
            x = x + film_params
        
        # Second convolution
        x = self.conv2(x)
        
        return x + residual

class DiffusionSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    """

    def __init__(self, input_shape, num_kp=None):
        """
        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have the same number of channels as input.
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints

def _replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    Replace all submodules in root_module that satisfy the given predicate with the output of func.
    Args:
        root_module: Root module whose submodules will be replaced.
        predicate: Function that takes a nn.Module and returns True if it should be replaced.
        func: Function that takes a nn.Module and returns its replacement.
    Returns:
        Root module with replaced submodules.
    """
    for name, module in root_module.named_children():
        if predicate(module):
            setattr(root_module, name, func(module))
        else:
            _replace_submodules(module, predicate, func)
    return root_module