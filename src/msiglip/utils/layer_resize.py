import torch
import math
import torch.nn.functional as F


@torch.no_grad()
def resize_pos_embed(posemb, height, width, original_model_include_cls_token=True):
    """
    Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224

    Note:
        Originally, CLIP added a CLS token to the position embeddings, hence, the first token is ignored and only interpolate the rest of the grid.
        However, as SigLIP does not use a CLS token, the first token is included in the interpolation.

    Args:
        posemb (`torch.Tensor`):
            Position embeddings to resize.
        height (`int`):
            Target height of the grid.
        width (`int`):
            Target width of the grid.
        include_cls_token (`bool`, *optional*):
            Whether to include the first token in the interpolation. Default is `True`.
    """
    posemb = posemb.unsqueeze(0)

    if original_model_include_cls_token:
        posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    else:
        posemb_grid = posemb[0, :]

    # Reshape into a 2D grid to keep the structure of an image
    gs_old = int(math.sqrt(len(posemb_grid)))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(height, width), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, height * width, -1)

    # Concatenate the first token if it was ignored if the originam posemb had a CLS token
    if original_model_include_cls_token:
        posemb = torch.cat([posemb_token, posemb_grid], dim=1)
        return posemb.squeeze(0)

    return posemb_grid.squeeze(0)


@torch.no_grad()
def resize_text_pos_embedding(pos_embed, target_dim=77, ignore="none"):
    if pos_embed.size(0) == target_dim:
        return pos_embed

    pos_tokens = pos_embed[1:-1, :].unsqueeze(0).permute(0, 2, 1)
    start_token = pos_embed[:1, :]
    end_token = pos_embed[-1:, :]

    if ignore == "first":
        # Concate pos_tokens with last token for interpolation
        size = target_dim - 1
        pos_tokens = torch.cat(
            [pos_tokens, end_token.unsqueeze(0).permute(0, 2, 1)], dim=2
        )
    elif ignore == "last":
        size = target_dim - 1
        # Concate pos_tokens with first token for interpolation
        pos_tokens = torch.cat(
            [start_token.unsqueeze(0).permute(0, 2, 1), pos_tokens], dim=2
        )
    else:
        size = target_dim - 2

    pos_tokens = F.interpolate(pos_tokens, size=size, mode="linear")
    pos_tokens = pos_tokens.squeeze(0).t()

    if ignore == "first":
        pos_tokens = torch.cat([start_token, pos_tokens], dim=0)
    elif ignore == "last":
        pos_tokens = torch.cat([pos_tokens, end_token], dim=0)
    else:
        pos_tokens = torch.cat([start_token, pos_tokens, end_token], dim=0)
    return pos_tokens


@torch.no_grad()
def resize_token_embedding(old_weight, new_num_tokens=None, pad_to_multiple_of=1):
    """Build a resized Embedding Module from a provided token Embedding weight. Increasing the size will add ewly
    initialized vectors at the end. Reducing the size will remove vectors nfrom the end.
    Args:
        old_weight (`torch.Tensor`):
            Old Embedding weight to resize.
        new_num_tokens (`int`, *optional*):
            New number of tokens in the embedding matrix.
            Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
            vectors from the end. If not provided or `None`, just returns the input tensor
            without doing anything.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the embedding matrix to a multiple of the provided value. If `new_num_tokens` is set to
            `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.
            Set to `None` or `1` to disable. Default is `1`.
    Return:
        `torch.Tensor`: Resized Embedding weight.
    """
    if new_num_tokens is None and pad_to_multiple_of is None:
        return old_weight

    old_num_tokens, embedding_dim = old_weight.size()

    if new_num_tokens is None:
        new_num_tokens = old_num_tokens

    if pad_to_multiple_of is not None:
        new_num_tokens = (
            math.ceil(new_num_tokens / pad_to_multiple_of) * pad_to_multiple_of
        )

    if new_num_tokens == old_num_tokens:
        return old_weight

    # Initialize a new weight tensor
    new_weight = torch.empty(
        (new_num_tokens, embedding_dim),
        dtype=old_weight.dtype,
        device=old_weight.device,
    )

    # Copy the old weights
    min_tokens = min(old_num_tokens, new_num_tokens)
    new_weight[:min_tokens, :] = old_weight[:min_tokens, :]

    # If expanding, initialize new tokens using the same distribution as the original weights
    if new_num_tokens > old_num_tokens:
        mean = old_weight.mean().item()
        std = old_weight.std().item()
        new_weight[old_num_tokens:, :].normal_(mean=mean, std=std)

    return new_weight
