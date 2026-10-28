"""Response loss, SL margin loss and Bliss regularization."""

import numpy as np
import torch
from torch.nn import functional as F
from .model import predict_in_batches

MARGIN_SCALE = 0.05
HARD_POSITIVE_FRACTION = 0.1
HARD_POSITIVE_WEIGHT = 2.0
NEGATIVE_WEIGHT = 1.0


def response_loss(model, response, config, backward=False):
    pairs, contexts, targets, weights = response
    total = 0.0
    for start in range(0, len(pairs), config["batch_size"]):
        section = slice(start, start + config["batch_size"])
        logits = model(pairs[section], contexts[section])
        loss = (
            F.binary_cross_entropy_with_logits(
                logits, targets[section].nan_to_num(), reduction="none"
            )
            * weights[section]
        ).sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite response loss")
        if backward:
            (1.0 * loss).backward()
        total += float(loss.detach())
    return total


def margin_loss(model, positive, negative, config, backward=False):
    chunk = config["batch_size"]
    temperature = MARGIN_SCALE
    positive_target = config["m_pos"]
    negative_target = config["m_neg"]
    with torch.no_grad():
        margins = predict_in_batches(model, positive, chunk, "raw_difference")
        tail_count = max(1, int(np.ceil(len(positive) * HARD_POSITIVE_FRACTION)))
        weights = torch.full_like(margins, 1 / len(positive))
        weights[margins.topk(tail_count, largest=False).indices] += (
            HARD_POSITIVE_WEIGHT / tail_count
        )
    losses = []
    for pairs, is_positive in ((positive, True), (negative, False)):
        total = 0.0
        for start in range(0, len(pairs), chunk):
            raw = model.raw_difference(pairs[start : start + chunk])
            if is_positive:
                loss = (
                    F.relu((positive_target - raw) / temperature).square()
                    * weights[start : start + len(raw)]
                ).sum()
            else:
                loss = (
                    F.relu((raw - negative_target) / temperature).square().sum()
                    * NEGATIVE_WEIGHT
                    / len(negative)
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite SL margin loss")
            if backward:
                (config["lambda_margin"] * loss).backward()
            total += float(loss.detach())
        losses.append(total)
    return sum(losses)


def sample_bliss_pairs(n_genes, n_contexts, count, device, generator=None):
    u = torch.randint(n_genes, (count,), device=device, generator=generator)
    v = torch.randint(n_genes - 1, (count,), device=device, generator=generator)
    v = v + (v >= u).long()
    # Preserve the context draw in the reproducible sampling stream.
    contexts = torch.randint(n_contexts, (count,), device=device, generator=generator)
    return (torch.stack([u, v], dim=1).sort(dim=1).values, contexts)


def bliss_loss(model, sample, config, backward=False):
    pairs, contexts = sample
    total = 0.0
    for start in range(0, len(pairs), config["batch_size"]):
        section = slice(start, start + config["batch_size"])
        all_delta = model._raw_context_logits(pairs[section], contexts=None)[..., 0]
        maximum_positive = F.relu(all_delta).amax(dim=1)
        loss = maximum_positive.square().sum() / len(pairs)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite Bliss loss")
        if backward:
            (config["lambda_bliss"] * loss).backward()
        total += float(loss.detach())
    return total
