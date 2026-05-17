from msiglip.utils.parse_module_str import parse_module_str
from transformers import PreTrainedTokenizer


def get_tokenizer(tokenizer_args: dict) -> PreTrainedTokenizer:
    """
    Get tokenizer from tokenizer_args
    Notes:
        - Added special token ids. These ids will be ignored during masking if use MLM
        - Added true_vocab_size to get the true vocab size which matches the embedding matrix size
    Args:
        tokenizer_args: dict, tokenizer arguments
    Returns:
        tokenizer: PreTrainedTokenizer, tokenizer object
    """
    modifiable_tokenizer_args = tokenizer_args.copy()
    tokenizer_type = modifiable_tokenizer_args.pop("type")
    vocab_size = modifiable_tokenizer_args.pop("vocab_size")
    add_mask_token = modifiable_tokenizer_args.pop("add_mask_token", False)

    if not vocab_size:
        raise ValueError("vocab_size is required for tokenizer")

    tokenizer = parse_module_str(tokenizer_type).from_pretrained(
        **modifiable_tokenizer_args
    )

    if add_mask_token:
        tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        tokenizer.special_token_ids = [
            tokenizer.convert_tokens_to_ids(token)
            for token in tokenizer.special_tokens_map.values()
        ]
    # Trick to get the true vocab size which matches the embedding matrix size
    tokenizer.true_vocab_size = vocab_size

    return tokenizer
