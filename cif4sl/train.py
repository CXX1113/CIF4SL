"""Train the response predictor with the three-term CIF4SL objective."""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn

from .graph import sl_adjacency, two_hop_operator, attach_support, export_evidence, propagate
from .io import save_json
from .losses import response_loss, margin_loss, bliss_loss, sample_bliss_pairs
from .model import CIF4SL
from .responses import prepare_responses, training_pairs
from .scoring import CIF4SLScorer, validation_ndcg


def _snapshot(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not value.is_sparse
    }


def _clock(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _tensors(data, parts, measured, settings, device):
    rows, contexts, context_index, weights, audit = prepare_responses(
        measured, parts, data.n_genes, settings
    )
    positive, negative = training_pairs(parts["train"], data.n_genes)
    response = (
        torch.as_tensor(rows[["u", "v"]].to_numpy(dtype=np.int64), device=device),
        torch.as_tensor(rows.context.map(context_index).to_numpy(dtype=np.int64), device=device),
        torch.as_tensor(rows[["y11", "y10", "y01"]].to_numpy(dtype=np.float32), device=device),
        torch.as_tensor(weights, device=device),
    )
    return (
        response,
        contexts,
        torch.as_tensor(positive, device=device),
        torch.as_tensor(negative, device=device),
        audit,
    )


@torch.no_grad()
def _evaluate_objective(model, positive, negative, response, sampled_pairs, settings):
    model.eval()
    margin = margin_loss(model, positive, negative, settings)
    measured = response_loss(model, response, settings)
    bliss = bliss_loss(model, sampled_pairs, settings)
    total = settings["lambda_margin"] * margin + measured + settings["lambda_bliss"] * bliss
    return {
        "response_loss": measured,
        "margin_loss": margin,
        "bliss_loss": bliss,
        "total_loss": total,
    }


def _step(model, optimizer, positive, negative, response, settings, generator):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    # Accumulate the three gradients in the same order for reproducibility.
    margin = margin_loss(model, positive, negative, settings, backward=True)
    measured = response_loss(model, response, settings, backward=True)
    pairs = sample_bliss_pairs(
        model.n,
        len(model.context_names),
        settings["bliss_pairs"],
        model.embedding.weight.device,
        generator,
    )
    bliss = bliss_loss(model, pairs, settings, backward=True)
    norm = nn.utils.clip_grad_norm_(
        model.parameters(), settings["grad_clip"], error_if_nonfinite=True
    )
    optimizer.step()
    return {
        "response_loss": measured,
        "margin_loss": margin,
        "bliss_loss": bliss,
        "gradient_norm": float(norm),
    }, pairs


def _warmup(data, parts, measured, settings, seed, fold, out, device):
    response, contexts, positive, negative, response_audit = _tensors(
        data, parts, measured, settings, device
    )
    out.mkdir(parents=True, exist_ok=True)
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed) + int(fold))
        model = CIF4SL(data.n_genes, contexts, settings, data.stable_nodes).to(device)
        generator = torch.Generator(device=device).manual_seed(int(seed) + int(fold) + 811)
        evaluation_pairs = sample_bliss_pairs(
            data.n_genes, len(contexts), settings["bliss_pairs"], device, generator
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=settings["warmup_lr"], weight_decay=settings["weight_decay"]
        )
        history = [
            {
                "epoch": 0,
                **_evaluate_objective(
                    model, positive, negative, response, evaluation_pairs, settings
                ),
            }
        ]
        best = None
        for epoch in range(1, settings["warmup_epochs"] + 1):
            stats, _ = _step(model, optimizer, positive, negative, response, settings, generator)
            record = {"epoch": epoch, "gradient_norm": stats["gradient_norm"]}
            if epoch % settings["warmup_eval_every"] == 0 or epoch in (
                1,
                settings["warmup_epochs"],
            ):
                record.update(
                    _evaluate_objective(
                        model, positive, negative, response, evaluation_pairs, settings
                    )
                )
                if best is None or record["total_loss"] < best["loss"]:
                    best = {"state": _snapshot(model), "epoch": epoch, "loss": record["total_loss"]}
                print(
                    f"[CIF4SL] Warm-up {epoch}/{settings['warmup_epochs']}: "
                    f"response={record['response_loss']:.5f}, margin={record['margin_loss']:.5f}, "
                    f"Bliss={record['bliss_loss']:.5f}",
                    flush=True,
                )
            history.append(record)
            if "response_loss" in record:
                pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        model.load_state_dict(best["state"], strict=True)
        model.eval()
        graph, graph_info = export_evidence(model, settings, out, capture_support=True)
        info = {
            "warmup_epochs": settings["warmup_epochs"],
            "warmup_selected_epoch": best["epoch"],
            "warmup_selection": "minimum training objective",
            "response_data": response_audit,
            "graph": graph_info,
        }
        _save_response(model, settings, info, out / "model.pt")
    return graph, model, info


def _save_response(model, settings, info, path):
    torch.save(
        {
            "format": "cif4sl_response_v4",
            "model_state": _snapshot(model),
            "n_genes": model.n,
            "contexts": list(model.context_names),
            "settings": settings,
            "training": info,
        },
        path,
    )


def _joint_train(data, parts, measured, model, topology, settings, info, seed, fold, out, device):
    support = sp.load_npz(out / "response" / "support.npz")
    attach_support(model, support)
    response, contexts, positive, negative, _ = _tensors(data, parts, measured, settings, device)
    if tuple(contexts) != tuple(model.context_names):
        raise RuntimeError("Joint training and warm-up context order differ")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings["joint_lr"], weight_decay=settings["weight_decay"]
    )
    generator = torch.Generator(device=device).manual_seed(int(seed) + int(fold) + 222)
    history, best = [], None

    def evaluate(epoch, stats):
        nonlocal best
        started = _clock(device)
        model.eval()
        graph, _ = export_evidence(model, settings, out / "validation_graph")
        representations = propagate(topology, graph).toarray().astype(np.float32)
        scorer = CIF4SLScorer(representations)
        ndcg = validation_ndcg(scorer, parts["train"], parts["valid"])
        record = {
            "epoch": epoch,
            "NDCG@10": ndcg,
            **stats,
            "validation_seconds": _clock(device) - started,
        }
        history.append(record)
        if best is None or ndcg > best["ndcg"]:
            best = {"epoch": epoch, "ndcg": ndcg, "state": _snapshot(model)}
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(f"[CIF4SL] Joint {epoch}/{settings['joint_epochs']}: NDCG@10={ndcg:.6f}", flush=True)

    evaluate(0, {})
    for epoch in range(1, settings["joint_epochs"] + 1):
        started = _clock(device)
        stats, _ = _step(model, optimizer, positive, negative, response, settings, generator)
        stats["training_seconds"] = _clock(device) - started
        if epoch % settings["joint_eval_every"] == 0 or epoch == settings["joint_epochs"]:
            evaluate(epoch, stats)
        else:
            history.append({"epoch": epoch, **stats})
            pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    model.load_state_dict(best["state"], strict=True)
    model.eval()
    graph, graph_info = export_evidence(model, settings, out / "response")
    representations = propagate(topology, graph).toarray().astype(np.float32, copy=False)
    scorer = CIF4SLScorer(representations)
    info = {
        **info,
        "joint_epochs": settings["joint_epochs"],
        "selected_epoch": best["epoch"],
        "joint_selection": "validation NDCG@10; strict improvement; epoch 0 included",
        "validation_NDCG@10": validation_ndcg(scorer, parts["train"], parts["valid"]),
        "graph": graph_info,
        "outer_test_used_for_selection": False,
    }
    _save_response(model, settings, info, out / "response" / "model.pt")
    np.save(out / "representations.npy", representations)
    torch.save(
        {
            "format": "cif4sl_scorer_v4",
            "representation_file": "representations.npy",
            "settings": settings,
            "seed": int(seed),
            "fold": int(fold),
            "training": info,
        },
        out / "model.pt",
    )
    save_json(info, out / "training.json")
    scorer.response_model = model
    scorer.training_info = info
    return scorer


def fit(data, parts, settings, seed, fold, out, device):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    torch.manual_seed(int(seed) + int(fold))
    measured = pd.read_csv(settings["response_file"])
    sl_graph = sl_adjacency(parts["train"], data.n_genes)
    topology = two_hop_operator(sl_graph)
    _, model, info = _warmup(data, parts, measured, settings, seed, fold, out / "response", device)
    save_json(settings, out / "resolved_settings.json")
    return _joint_train(
        data, parts, measured, model, topology, settings, info, seed, fold, out, device
    )
