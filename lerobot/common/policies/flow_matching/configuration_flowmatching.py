from dataclasses import dataclass, field
from lerobot.common.optim.optimizers import AdamConfig
from lerobot.common.optim.schedulers import DiffuserSchedulerConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode

@PreTrainedConfig.register_subclass("flowmatching")
@dataclass
class FlowMatchingConfig(PreTrainedConfig):
    """Configuration class for Flow Matching Policy."""

    # Input/output structure
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )
    drop_n_last_frames: int = 7  # Same as diffusion for consistency

    # Architecture
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False
    
    # UNet architecture parameters
    down_dims: tuple[int, ...] = (512, 1024, 2048)  # Feature dimensions for UNet stages
    kernel_size: int = 5  # Kernel size for UNet convolutions
    n_groups: int = 8  # Number of groups for GroupNorm layers
    time_embed_dim: int = 128  # Time embedding dimension
    use_film_scale_modulation: bool = True  # Whether to use FiLM scale modulation

    # Flow Matching specific parameters
    path_type: str = "CondOTProbPath"  # Options: ["CondOTProbPath", "GaussianPath"]
    path_sigma: float = 0.1  # For Gaussian paths
    num_inference_steps: int = 100  # Number of ODE solver steps during sampling

    # Training
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()
        # Validate path type
        supported_paths = ["CondOTProbPath", "GaussianPath"]
        if self.path_type not in supported_paths:
            raise ValueError(f"path_type must be one of {supported_paths}. Got {self.path_type}.")

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )