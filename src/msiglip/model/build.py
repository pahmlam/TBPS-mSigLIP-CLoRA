"""Build backbone model for TBPS.
Image size: 384x128"""

import logging
import warnings

from msiglip.utils.layer_resize import (
    resize_pos_embed,
    resize_text_pos_embedding,
    resize_token_embedding,
)
from msiglip.utils.parse_module_str import parse_module_str

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def build_backbone_with_proper_layer_resize(backbone_cfg, checkpoint_path: str = None, state_dict=None):
    """
    Build backbone model with proper layer resizing for TBPS.

    Args:
        backbone_cfg (dict): Configuration for the backbone model.
    Returns:
        model (nn.Module): Backbone model with proper layer resizing.
    """
    # Extract configuration
    modifiable_backbone_cfg = backbone_cfg.copy()
    backbone_type = modifiable_backbone_cfg.pop("type")
    logger.info(f"Building backbone model: {backbone_type}")

    if not checkpoint_path:
        checkpoint_path = modifiable_backbone_cfg.pop("path")
    original_model_include_cls_token = modifiable_backbone_cfg.pop(
        "original_model_include_cls_token"
    )
    image_height, image_width = modifiable_backbone_cfg.vision_config.image_size
    patch_size = modifiable_backbone_cfg.vision_config.patch_size
    config_type = modifiable_backbone_cfg.pop("config_type")

    # Parse configuration into Config object
    config = parse_module_str(config_type)(**modifiable_backbone_cfg)

    # Load the model state dict
    if checkpoint_path.split(".")[-1] == "bin":
        import torch

        original_state_dict = torch.load(checkpoint_path, map_location="cpu")

    elif checkpoint_path.split(".")[-1] == "safetensors":
        from safetensors.torch import load_file

        original_state_dict = load_file(checkpoint_path)

    # Parse the model sekeleton
    model = parse_module_str(backbone_type)(config=config)

    # Check the name of position embeddings and token embeddings for text and image models
    for name, param in model.named_parameters():
        if "visual" not in name and "vision" not in name:
            # Check the name of position embeddings of the text model
            if "position" in name and "embed" in name:
                text_model_position_embedding = name
                text_model_position_embedding_shape = param.shape[0]
            # Check the name of token embeddings of the text model
            if "visual" not in name and "token" in name and "embed" in name:
                text_model_token_embedding = name
                text_model_token_embedding_shape = param.shape[0]
        # Check the name of position embeddings of the image model
        if "position" in name and "embed" in name:
            if "vision" in name or "visual" in name or "image" in name:
                image_model_position_embedding = name
                image_model_position_embedding_shape = param.shape[0]

    logger.info(
        f"Name of position embeddings for text model: {text_model_position_embedding}"
    )
    logger.info(
        f"Name of token embeddings for text model: {text_model_token_embedding}"
    )
    logger.info(
        f"Name of position embeddings for image model: {image_model_position_embedding}"
    )

    # Check dimension mismatch for interpolation

    # Resize position embeddings for text model
    if (
        original_state_dict[text_model_position_embedding].shape[0]
        != text_model_position_embedding_shape
    ):
        logger.info(
            f"Resized {text_model_position_embedding} from {original_state_dict[text_model_position_embedding].shape[0]} to {text_model_position_embedding_shape}"
        )
        original_state_dict[text_model_position_embedding] = resize_text_pos_embedding(
            original_state_dict[text_model_position_embedding],
            target_dim=text_model_position_embedding_shape,
        )

    # Resize token embeddings for text model
    if (
        original_state_dict[text_model_token_embedding].shape[0]
        != text_model_token_embedding_shape
    ):
        logger.info(
            f"Resized {text_model_token_embedding} for text model from {original_state_dict[text_model_token_embedding].shape[0]} to {text_model_token_embedding_shape}"
        )
        original_state_dict[text_model_token_embedding] = resize_token_embedding(
            original_state_dict[text_model_token_embedding],
            new_num_tokens=text_model_token_embedding_shape,
        )

    # Resize position embeddings for image model
    if (
        original_state_dict[image_model_position_embedding].shape[0]
        != image_model_position_embedding_shape
    ):
        logger.info(
            f"Resized {image_model_position_embedding} for image model from {original_state_dict[image_model_position_embedding].shape[0]} to {image_model_position_embedding_shape}"
        )
        num_visual_vertical_patches = int(image_height / patch_size)
        num_visual_horizontal_patches = int(image_width / patch_size)
        original_state_dict[image_model_position_embedding] = resize_pos_embed(
            original_state_dict[image_model_position_embedding],
            height=num_visual_vertical_patches,
            width=num_visual_horizontal_patches,
            original_model_include_cls_token=original_model_include_cls_token,
        )

    # Delete unnecessary keys in the state dict
    necessary_keys = model.state_dict().keys()
    for key in list(original_state_dict.keys()):
        if key not in necessary_keys:
            del original_state_dict[key]

    model.load_state_dict(original_state_dict)

    return model
