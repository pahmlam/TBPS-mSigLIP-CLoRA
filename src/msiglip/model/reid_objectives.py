import torch
import torch.nn.functional as F


def euclidean_dist(x, y):
    """
    Calculate the pairwise Euclidean distance between two tensors.

    Args:
        x (torch.Tensor): A 2D tensor of shape (m, d).
        y (torch.Tensor): A 2D tensor of shape (n, d).

    Returns:
        torch.Tensor: A 2D tensor of shape (m, n) representing pairwise distances.
    """
    xx = x.pow(2).sum(dim=1, keepdim=True)
    yy = y.pow(2).sum(dim=1, keepdim=True).t()
    dist = xx + yy - 2 * x @ y.t()
    return dist.clamp(min=1e-12).sqrt()  # Ensure numerical stability


def batch_hard(mat_distance, mat_label, return_indices=False):
    """
    Find the hardest positive and negative samples in a batch based on pairwise distances.

    Args:
        mat_distance (torch.Tensor): Pairwise distance matrix of shape (batch_size, batch_size).
        mat_label (torch.Tensor): Binary label matrix of shape (batch_size, batch_size).
        return_indices (bool): If True, return the indices of hard positives and negatives.

    Returns:
        torch.Tensor: Hard positive distances of shape (batch_size,).
        torch.Tensor: Hard negative distances of shape (batch_size,).
        (Optional) torch.Tensor: Indices of hard positives.
        (Optional) torch.Tensor: Indices of hard negatives.
    """
    # Adjust distances for positives
    positive_distances = mat_distance + (-1e7) * (1 - mat_label)
    hardest_positive, positive_indices = positive_distances.max(dim=1)

    # Adjust distances for negatives
    negative_distances = mat_distance + (1e7) * mat_label
    hardest_negative, negative_indices = negative_distances.min(dim=1)

    if return_indices:
        return hardest_positive, hardest_negative, positive_indices, negative_indices
    return hardest_positive, hardest_negative


def triplet_loss(
    embeddings, labels, margin=None, normalize_features=False, return_precision=False
):
    """
    Compute the triplet loss with batch hard mining.

    Args:
        embeddings (torch.Tensor): Embedding vectors of shape (batch_size, embedding_dim).
        labels (torch.Tensor): Ground truth labels of shape (batch_size,).
        margin (float, optional): Margin for the loss. If None, soft margin loss is used.
        normalize_features (bool): If True, normalize the embeddings to unit length.
        return_precision (bool): If True, return the precision of triplet selection.

    Returns:
        torch.Tensor: The computed triplet loss.
        (Optional) float: Precision of triplet selection.
    """
    if normalize_features:
        embeddings = F.normalize(embeddings)

    # Compute pairwise distances and similarity matrix
    mat_dist = euclidean_dist(embeddings, embeddings)
    mat_label = labels.unsqueeze(0).eq(labels.unsqueeze(1)).float()

    # Batch hard mining
    dist_ap, dist_an = batch_hard(mat_dist, mat_label)
    y = torch.ones_like(dist_ap)

    # Loss computation
    if margin is not None:
        loss_fn = torch.nn.MarginRankingLoss(margin=margin)
        loss = loss_fn(dist_an, dist_ap, y)
    else:
        loss_fn = torch.nn.SoftMarginLoss()
        loss = loss_fn(dist_an - dist_ap, y)

    if return_precision:
        precision = (dist_an > dist_ap).float().mean().item()
        return loss, precision
    return loss

