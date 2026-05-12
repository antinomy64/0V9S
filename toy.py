"""
Toy test for the current Stage3GWLoss.

Goal
----
Construct two point sets with the same internal structure:
    V        : fixed visual prototypes
    T_fake   : V[perm] rotated by an orthogonal matrix R
Then train a tiny linear projector with the current Stage3GWLoss:
    dynamic GW assignment + pairwise weighted alignment + T-structure preservation
and check whether projector(T_fake_i) moves back to V[perm_i].

Run from the repository root:
    python toy_stage3_dynamic_gw.py

This script intentionally calls Stage3GWLoss.forward(). It does not reimplement
Stage3's loss.
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loss_stage3_gw import (
    Stage3GWLoss,
    entropic_gw,
    pairwise_cosine_distance,
    safe_normalize,
)


class ToyProjector(nn.Module):
    """Minimal model exposing the project_clip_txt API used by Stage3GWLoss."""

    def __init__(self, dim: int, init: str = "identity"):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            if init == "identity":
                self.proj.weight.copy_(torch.eye(dim))
            elif init == "orthogonal":
                q, _ = torch.linalg.qr(torch.randn(dim, dim))
                self.proj.weight.copy_(q)
            elif init == "random":
                nn.init.normal_(self.proj.weight, mean=0.0, std=1.0 / math.sqrt(dim))
            else:
                raise ValueError(f"Unknown init: {init}")

    def project_clip_txt(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def make_orthogonal_matrix(dim: int, device: torch.device) -> torch.Tensor:
    a = torch.randn(dim, dim, device=device)
    q, _ = torch.linalg.qr(a)
    return q


@torch.no_grad()
def evaluate(
    model: ToyProjector,
    part_text: torch.Tensor,
    visual: torch.Tensor,
    perm: torch.Tensor,
    criterion: Stage3GWLoss,
) -> Dict[str, float | List[int]]:
    z = safe_normalize(model.project_clip_txt(part_text), dim=-1)
    v = safe_normalize(visual, dim=-1)

    # Direct retrieval in the visual feature space.
    sim = z @ v.T
    retr_pred = sim.argmax(dim=1)
    retr_acc = (retr_pred == perm).float().mean().item()
    true_sim = (z * v[perm]).sum(dim=-1).mean().item()
    max_sim = sim.max(dim=1).values.mean().item()

    # Dynamic GW plan used by the current Stage3 loss.
    C_z = pairwise_cosine_distance(z.detach())
    C_v = pairwise_cosine_distance(v.detach())
    P = entropic_gw(
        C_z,
        C_v,
        epsilon=criterion.gw_epsilon,
        max_iter=criterion.gw_max_iter,
        sinkhorn_iter=criterion.sinkhorn_iter,
        init="uniform",
        hard=False,
    )
    gw_pred = P.argmax(dim=1)
    gw_acc = (gw_pred == perm).float().mean().item()

    # Structure preservation relative to the original T_fake structure.
    C_before = pairwise_cosine_distance(part_text.detach())
    C_after = pairwise_cosine_distance(z.detach())
    struct_mse = F.mse_loss(C_after, C_before).item()

    # How sharp the GW plan is. Row mass is 1/K; max row probability close to 1/K means sharp.
    k = part_text.shape[0]
    row_max = P.max(dim=1).values.mean().item()
    sharp_ratio = row_max / (1.0 / k)

    return {
        "retr_acc": retr_acc,
        "gw_acc": gw_acc,
        "true_sim": true_sim,
        "max_sim": max_sim,
        "struct_mse": struct_mse,
        "gw_row_max": row_max,
        "gw_sharp_ratio": sharp_ratio,
        "retr_pred": retr_pred.detach().cpu().tolist(),
        "gw_pred": gw_pred.detach().cpu().tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lambda_gw", type=float, default=1.0)
    parser.add_argument("--lambda_struct", type=float, default=0.1)
    parser.add_argument("--gw_epsilon", type=float, default=0.001)
    parser.add_argument("--gw_max_iter", type=int, default=100)
    parser.add_argument("--sinkhorn_iter", type=int, default=200)
    parser.add_argument("--projector_init", type=str, default="identity", choices=["identity", "orthogonal", "random"])
    parser.add_argument("--no_rotation", action="store_true")
    parser.add_argument("--print_every", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fixed visual prototypes.
    V = safe_normalize(torch.randn(args.k, args.dim, device=device), dim=-1)

    # T_fake has the same structure as V, but rows are permuted and optionally rotated.
    perm = torch.randperm(args.k, device=device)
    if args.no_rotation:
        R = torch.eye(args.dim, device=device)
    else:
        R = make_orthogonal_matrix(args.dim, device)
    T_fake = safe_normalize(V[perm] @ R, dim=-1)

    model = ToyProjector(args.dim, init=args.projector_init).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    part_ids = torch.arange(args.k, device=device, dtype=torch.long)
    class_blocks = [
        {
            "category_id": 0,
            "class_name": "toy_object",
            "part_ids": part_ids,
            "part_text": T_fake,
            "part_names": [f"p{i}" for i in range(args.k)],
        }
    ]
    proto_count = torch.ones(args.k, device=device)

    criterion = Stage3GWLoss(
        sim_model=model,
        visual_proto=V,
        class_blocks=class_blocks,
        lambda_obj=0.0,
        lambda_gw=args.lambda_gw,
        lambda_struct=args.lambda_struct,
        gw_epsilon=args.gw_epsilon,
        gw_max_iter=args.gw_max_iter,
        sinkhorn_iter=args.sinkhorn_iter,
        min_proto_count=1,
        proto_count=proto_count,
    ).to(device)

    print("device:", device)
    print("perm:", perm.detach().cpu().tolist())
    print(
        "settings:",
        f"k={args.k}",
        f"dim={args.dim}",
        f"steps={args.steps}",
        f"lr={args.lr}",
        f"lambda_gw={args.lambda_gw}",
        f"lambda_struct={args.lambda_struct}",
        f"gw_epsilon={args.gw_epsilon}",
        f"gw_max_iter={args.gw_max_iter}",
        f"sinkhorn_iter={args.sinkhorn_iter}",
        f"projector_init={args.projector_init}",
        f"rotation={not args.no_rotation}",
    )

    for step in range(args.steps + 1):
        losses = criterion(batch=None, do_structure_audit=False)
        loss = losses["total"]

        if step % args.print_every == 0 or step == args.steps:
            metrics = evaluate(model, T_fake, V, perm, criterion)
            print(
                f"step={step:04d} "
                f"total={float(losses['total'].detach()):.6f} "
                f"gw={float(losses['gw'].detach()):.6f} "
                f"struct={float(losses['struct'].detach()):.6f} "
                f"retr_acc={metrics['retr_acc']:.3f} "
                f"gw_acc={metrics['gw_acc']:.3f} "
                f"true_sim={metrics['true_sim']:.4f} "
                f"max_sim={metrics['max_sim']:.4f} "
                f"struct_mse={metrics['struct_mse']:.6e} "
                f"gw_sharp={metrics['gw_sharp_ratio']:.2f}x"
            )

        if step == args.steps:
            break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    final_metrics = evaluate(model, T_fake, V, perm, criterion)
    print("final_retr_pred:", final_metrics["retr_pred"])
    print("final_gw_pred:", final_metrics["gw_pred"])
    print("perm:", perm.detach().cpu().tolist())

    # Useful criterion for this toy:
    # - retr_acc should go up toward 1.0 if the projector has really rotated T_fake back.
    # - struct_mse should stay small if structure preservation is working.
    # - gw_acc shows whether the current dynamic entropic GW plan itself found the same permutation.


if __name__ == "__main__":
    main()
