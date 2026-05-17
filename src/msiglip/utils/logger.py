import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any, Tuple, Union

from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger, Logger
from lightning.pytorch.callbacks import ModelCheckpoint
from hydra.core.hydra_config import HydraConfig


def get_config_overrides() -> Dict[str, Any]:
    """Extract and process config overrides from Hydra."""
    hydra_cfg = HydraConfig.get()
    overrides = {}

    if hasattr(hydra_cfg, "overrides") and getattr(hydra_cfg.overrides, "task", None):
        for override in hydra_cfg.overrides.task:
            if override[0] == "+":
                continue  # Skip the adding of the task
            key, value = override.split("=")
            try:
                # Try to evaluate the value if it's a number
                value = eval(value)
            except:
                pass
            overrides[key] = value

    return overrides


def generate_experiment_name(config_name: str, overrides: Dict[str, Any]) -> str:
    """Generate experiment name based on config overrides."""
    if not overrides:
        return f"{config_name}_base"

    # Sort overrides for consistent naming
    override_parts = []
    for key, value in sorted(overrides.items()):
        # Skip special Hydra keys and None values
        if key.startswith("hydra.") or value is None:
            continue
        # Handle nested configs
        if isinstance(value, (dict, DictConfig)):
            value = "custom"
        override_parts.append(f"{key}_{value}")

    return f"{config_name}_{'_'.join(override_parts)}"


def setup_logger(
    config: DictConfig,
    experiment_name: Optional[str] = None,
) -> Logger:
    """Set up TensorBoard logger with base config name and overridden config as version."""

    # Get base config name and overrides
    hydra_cfg = HydraConfig.get()
    config_name = hydra_cfg.job.config_name  # Base config name (e.g., 'vn3k_en')
    overrides = get_config_overrides()

    # Generate experiment name if not already provided
    if experiment_name is None:
        experiment_name = config.logger.experiment_name
        if experiment_name is None:
            experiment_name = generate_experiment_name(config_name, overrides)

    # Save checkpoint and log files in hydra output directory
    log_dir = HydraConfig.get().runtime.output_dir

    logger = None

    if config.logger.logger_type == "tensorboard":
        # Initialize TensorBoard logger
        logger = TensorBoardLogger(
            save_dir=log_dir,
            name=experiment_name,  # If overridden config is empty, use 'base'
            version=None,  # Overrides as version (e.g., 'aug_irra_loss_irra')
        )
    elif config.logger.logger_type == "wandb":
        # Create {save_dir}/{experiment_name}/wandb directory
        os.makedirs(os.path.join(log_dir, "wandb"), exist_ok=True)
        logger = WandbLogger(
            name=experiment_name,
            save_dir=log_dir,
            version=None,
            project="PERSON_SEARCH",
            config=OmegaConf.to_container(config, resolve=True),
            reinit=True,  # Force new run for each Hydra job
        )

    logging.info("Logger setup:")
    logging.info(f"Base config: {config_name}")
    logging.info(f"Overrided config: {overrides}")
    logging.info(f"Log directory: {logger.save_dir}")

    if logger:
        return logger
    else:
        raise ValueError("No logger specified in config.")


def setup_checkpoint_callback(
    config: DictConfig,
    log_dir: str,
) -> ModelCheckpoint:
    """Set up checkpoint callback."""
    checkpoint_path = os.path.join(log_dir, "checkpoints")
    logging.info(f"Checkpoint path: {checkpoint_path}")
    return ModelCheckpoint(
        dirpath=checkpoint_path,
        filename=config.logger.checkpoint.filename,
        save_top_k=config.logger.checkpoint.save_top_k,
        monitor=config.logger.checkpoint.monitor,
        mode=config.logger.checkpoint.mode,
        save_last=config.logger.checkpoint.save_last,
    )


def setup_logging(
    config: DictConfig,
    experiment_name: Optional[str] = None,
) -> Tuple[Union[WandbLogger, TensorBoardLogger], ModelCheckpoint]:
    """Set up logging and checkpoints."""
    # Configure logging levels
    logging.basicConfig(
        level=getattr(logging, config.logger.console.level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Set up TensorBoard logger
    training_logger = setup_logger(config, experiment_name)

    # Set up checkpoint callback
    checkpoint_callback = setup_checkpoint_callback(config, training_logger.save_dir)

    # Add file logging if configured
    if config.logger.file.filename:
        file_handler = RotatingFileHandler(
            os.path.join(
                HydraConfig.get().runtime.output_dir, config.logger.file.filename
            ),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
        )
        file_handler.setLevel(getattr(logging, config.logger.file.level))
        logging.getLogger().addHandler(file_handler)

    return training_logger, checkpoint_callback
