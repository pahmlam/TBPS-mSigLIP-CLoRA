from peft import get_peft_model
from hydra.utils import instantiate


def get_lora_model(model, lora_config):
    """
    Get the LoRa model with configuration

    Args:
        model: The model to be used.
        lora_config: Configuration for the LoRa model.
            Example format:
            __target__: peft.LoraConfig
            r=16
            lora_alpha=16
            target_modules=["query", "value"]
            lora_dropout=0.1
            bias="none"
            modules_to_save=["classifier"]
    Returns:
        The LoRa model with the given configuration.
    """
    lora_config = instantiate(lora_config)
    model = get_peft_model(model, lora_config)
    return model
