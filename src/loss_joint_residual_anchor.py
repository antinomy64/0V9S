
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loss import ContrastiveLoss


class JointObjPartLoss(nn.Module):
    """
    Joint loss for:
      1) object-level branch: keep original contrastive objective
      2) part-level branch: object-inside part supervision on patch tokens

    New part logic:
      - build object-centered residuals on both text and vision sides
      - find unique anchors on residual relative scores
      - compute anchor hit metrics INSIDE forward (no extra audit recomputation)
      - use anchors only to initialize prototypes
      - run prototype EM in residual space (instead of region growing)
      - compute final part vision features by mean pooling original patch tokens
      - keep overlap disabled; optional global residual-space Spearman surrogate
    """

    def __init__(
        self,
        sim_model,
        obj_ltype: str = "infonce",
        obj_margin: float = 0.2,
        obj_max_violation: bool = True,
        lambda_obj: float = 1.0,
        lambda_inst: float = 0.2,
        lambda_overlap: float = 0.05,
        lambda_spear: float = 0.0,
        topk_ratio: float = 0.1,
        patch_temperature: float = 0.07,
        eps: float = 1e-6,
        em_iters: int = 3,
    ):
        super().__init__()
        self.sim_model = sim_model
        self.obj_criterion = ContrastiveLoss(
            sim_model,
            margin=obj_margin,
            max_violation=obj_max_violation,
            ltype=obj_ltype,
        )
        self.lambda_obj = lambda_obj
        self.lambda_inst = lambda_inst
        self.lambda_overlap = lambda_overlap
        self.lambda_spear = lambda_spear
        self.topk_ratio = topk_ratio
        self.patch_temperature = patch_temperature
        self.eps = eps
        self.em_iters = int(em_iters)

    def _safe_normalize(self, x, dim=-1):
        return x / x.norm(dim=dim, keepdim=True).clamp_min(self.eps)

    def forward(self, batch):
        obj_feat = batch["obj_feat"]
        patch_tokens = batch["patch_tokens"]
        obj_text_feat = batch["obj_text_feat"]
        part_text_feat = batch["part_text_feat"]
        obj_mask_patch = batch["obj_mask_patch"]
        part_valid_mask = batch["part_valid_mask"]
        part_gt_mask_patch = batch["part_gt_mask_patch"]

        obj_loss = self.obj_criterion(
            obj_feat,
            obj_text_feat,
            return_similarity_mat=False,
            self_attn_maps=None,
            cls=None,
            text_input_mask=None,
            text_argmax=None,
        )

        zero = obj_loss.new_tensor(0.0)

        if part_text_feat.shape[1] == 0 or not part_valid_mask.any():
            total = self.lambda_obj * obj_loss
            return {
                "total": total,
                "obj": obj_loss.detach(),
                "inst": zero.detach(),
                "overlap": zero.detach(),
                "spear": zero.detach(),
                "anchor_hit_rate": zero.detach(),
                "anchor_total_valid_parts": zero.detach(),
                "anchor_total_hits": zero.detach(),
            }

        # project object/part text into the same space as patch tokens
        obj_proj = self.sim_model.project_clip_txt(obj_text_feat.float())       # [B, D]
        part_proj = self.sim_model.project_clip_txt(part_text_feat.float())     # [B, K, D]

        obj_proj = self._safe_normalize(obj_proj, dim=-1)
        part_proj = self._safe_normalize(part_proj, dim=-1)
        patch_tokens = self._safe_normalize(patch_tokens.float(), dim=-1)

        obj_origin = obj_proj[:, None, :]                                       # [B, 1, D]

        # residuals on both sides
        text_residual = self._safe_normalize(part_proj - obj_origin, dim=-1)    # [B, K, D]
        vision_residual = self._safe_normalize(patch_tokens - obj_origin, dim=-1)  # [B, N, D]

        # residual score map
        residual_logits = torch.einsum("bkd,bnd->bkn", text_residual, vision_residual) / self.patch_temperature
        residual_logits = residual_logits.masked_fill(~obj_mask_patch[:, None, :], -1e4)

        z_part, proto_residual, anchor_metrics = self._anchor_proto_em_pool(
            patch_tokens=patch_tokens,
            patch_residual=vision_residual,
            residual_logits=residual_logits,
            obj_mask_patch=obj_mask_patch,
            part_valid_mask=part_valid_mask,
            part_gt_mask_patch=part_gt_mask_patch,
            num_iters=self.em_iters,
        )

        inst_loss = self._instance_consistency_loss(part_proj, z_part, part_valid_mask)

        # Keep overlap off; use a residual-space global Spearman-style surrogate.
        overlap_loss = zero
        spear_loss = self._residual_spearman_surrogate_loss(
            text_residual=text_residual,
            proto_residual=proto_residual,
            vision_residual=vision_residual,
            obj_mask_patch=obj_mask_patch,
            part_valid_mask=part_valid_mask,
        ) if self.lambda_spear > 0 else zero

        total = (
            self.lambda_obj * obj_loss
            + self.lambda_inst * inst_loss
            + self.lambda_overlap * overlap_loss
            + self.lambda_spear * spear_loss
        )

        return {
            "total": total,
            "obj": obj_loss.detach(),
            "inst": inst_loss.detach(),
            "overlap": overlap_loss.detach(),
            "spear": spear_loss.detach(),
            "anchor_hit_rate": anchor_metrics["anchor_hit_rate"].detach(),
            "anchor_total_valid_parts": anchor_metrics["anchor_total_valid_parts"].detach(),
            "anchor_total_hits": anchor_metrics["anchor_total_hits"].detach(),
        }

    def _compute_relative_scores(self, local_scores: torch.Tensor) -> torch.Tensor:
        """
        local_scores: [K_b, M_b]
        rel[p, n] = local_scores[p, n] - best_other[p, n]
        where best_other[p, n] is the highest score among other parts on patch n.
        """
        Kb, Mb = local_scores.shape
        if Kb <= 1:
            return local_scores

        top2_vals, top2_idx = torch.topk(local_scores, k=min(2, Kb), dim=0)
        best_vals = top2_vals[0]
        best_idx = top2_idx[0]
        second_vals = top2_vals[1]

        row_ids = torch.arange(Kb, device=local_scores.device)[:, None]
        is_top1 = row_ids == best_idx[None, :]
        best_other = torch.where(is_top1, second_vals[None, :], best_vals[None, :])

        rel_scores = local_scores - best_other
        return rel_scores

    def _anchor_proto_em_pool(
        self,
        patch_tokens,
        patch_residual,
        residual_logits,
        obj_mask_patch,
        part_valid_mask,
        part_gt_mask_patch,
        num_iters=3,
    ):
        """
        Residual anchor + prototype EM:
          1) for each object, choose one UNIQUE anchor patch per valid part from residual relative scores
          2) compute anchor hit metrics immediately from GT part masks
          3) use anchor residuals to initialize prototypes
          4) run a few rounds of prototype competition in residual space
          5) use final hard assignment to mean-pool ORIGINAL patch tokens as z_part
        """
        B, K, N = residual_logits.shape
        D = patch_tokens.shape[-1]
        z = patch_tokens.new_zeros((B, K, D))
        proto_residual = patch_tokens.new_zeros((B, K, D))

        total_valid_parts = patch_tokens.new_tensor(0.0)
        total_anchor_hits = patch_tokens.new_tensor(0.0)

        for b in range(B):
            valid_patch_mask = obj_mask_patch[b]
            valid_part_idx = torch.nonzero(part_valid_mask[b], as_tuple=False).squeeze(1)

            if valid_part_idx.numel() == 0 or valid_patch_mask.sum() == 0:
                continue

            valid_patch_tokens = patch_tokens[b][valid_patch_mask]       # [Mb, D]
            valid_patch_residual = patch_residual[b][valid_patch_mask]   # [Mb, D]
            local_scores = residual_logits[b][valid_part_idx][:, valid_patch_mask]  # [Kb, Mb]

            Kb, Mb = local_scores.shape
            if Mb == 0:
                continue

            # Step 1: unique anchors on relative residual scores
            rel_scores = self._compute_relative_scores(local_scores)

            flat_scores = rel_scores.reshape(-1)
            sorted_idx = torch.argsort(flat_scores, descending=True)

            anchor_idx_local = torch.full((Kb,), -1, dtype=torch.long, device=local_scores.device)
            patch_taken = torch.zeros((Mb,), dtype=torch.bool, device=local_scores.device)

            assigned_parts = 0
            for flat_id in sorted_idx:
                p_local = torch.div(flat_id, Mb, rounding_mode='floor')
                n_local = flat_id % Mb

                if anchor_idx_local[p_local] != -1:
                    continue
                if patch_taken[n_local]:
                    continue

                anchor_idx_local[p_local] = n_local
                patch_taken[n_local] = True
                assigned_parts += 1
                if assigned_parts == Kb:
                    break

            # Fallback when unique assignment cannot cover all parts (rare unless Mb < Kb)
            unassigned = torch.nonzero(anchor_idx_local < 0, as_tuple=False).squeeze(1)
            if unassigned.numel() > 0:
                local_best = rel_scores.argmax(dim=1)
                anchor_idx_local[unassigned] = local_best[unassigned]

            # Step 2: anchor hit metrics immediately (no extra audit recomputation later)
            valid_patch_idx_global = torch.nonzero(valid_patch_mask, as_tuple=False).squeeze(1)
            anchor_idx_global = valid_patch_idx_global[anchor_idx_local]  # [Kb]

            gt_masks = part_gt_mask_patch[b, valid_part_idx]              # [Kb, N]
            hit_vec = gt_masks[torch.arange(Kb, device=gt_masks.device), anchor_idx_global]  # [Kb]

            total_valid_parts += float(Kb)
            total_anchor_hits += float(hit_vec.long().sum().item())

            # Step 3: anchor initializes prototype in residual space
            C = valid_patch_residual[anchor_idx_local]                    # [Kb, D]

            # Step 4: prototype EM in residual space
            assign = None
            for _ in range(max(int(num_iters), 1)):
                assign_scores = valid_patch_residual @ C.T                # [Mb, Kb]
                assign = assign_scores.argmax(dim=1)                      # [Mb]

                # lock anchors so every valid part stays non-empty
                assign[anchor_idx_local] = torch.arange(Kb, device=assign.device)

                onehot = F.one_hot(assign, num_classes=Kb).float()        # [Mb, Kb]
                count = onehot.sum(dim=0).clamp_min(1.0)                  # [Kb]
                proto_sum = onehot.T @ valid_patch_residual               # [Kb, D]
                C = proto_sum / count[:, None]
                C = self._safe_normalize(C, dim=-1)

            # Step 5: final part visual feature from ORIGINAL patch tokens
            region_onehot = F.one_hot(assign, num_classes=Kb).float()     # [Mb, Kb]
            region_count = region_onehot.sum(dim=0).clamp_min(1.0)        # [Kb]
            region_sum = region_onehot.T @ valid_patch_tokens             # [Kb, D]
            z_local = region_sum / region_count[:, None]
            z_local = self._safe_normalize(z_local, dim=-1)

            z[b, valid_part_idx] = z_local
            proto_residual[b, valid_part_idx] = C

        hit_rate = total_anchor_hits / total_valid_parts.clamp_min(1.0)
        anchor_metrics = {
            "anchor_hit_rate": hit_rate,
            "anchor_total_valid_parts": total_valid_parts,
            "anchor_total_hits": total_anchor_hits,
        }
        return z, proto_residual, anchor_metrics

    def _instance_consistency_loss(self, part_proj, z_part, part_valid_mask):
        cos = F.cosine_similarity(part_proj, z_part.detach(), dim=-1)
        loss = 1.0 - cos
        return self._masked_mean(loss, part_valid_mask)

    def _residual_spearman_surrogate_loss(
        self,
        text_residual,
        proto_residual,
        vision_residual,
        obj_mask_patch,
        part_valid_mask,
    ):
        """
        Global residual-space Spearman-style surrogate.

        For each valid part:
          - text side scores:    <text_residual_k,  vision_residual_n>
          - pseudo/proto scores: <proto_residual_k, vision_residual_n>

        We compute the correlation on ALL object patches of that image, because
        training only knows the object's full part set, not which parts truly appear.
        To stabilize training, the pseudo/proto side is detached and only acts as a target.
        """
        txt_scores = torch.einsum("bkd,bnd->bkn", text_residual, vision_residual.detach())
        vis_scores = torch.einsum("bkd,bnd->bkn", proto_residual.detach(), vision_residual.detach())

        losses = []
        B, K, _ = txt_scores.shape
        for b in range(B):
            valid_patch = obj_mask_patch[b]
            if valid_patch.sum() <= 1:
                continue
            for p in range(K):
                if not part_valid_mask[b, p]:
                    continue
                t = txt_scores[b, p, valid_patch]
                v = vis_scores[b, p, valid_patch]

                t = t - t.mean()
                v = v - v.mean()

                denom = torch.sqrt((t ** 2).sum() + self.eps) * torch.sqrt((v ** 2).sum() + self.eps)
                corr = (t * v).sum() / (denom + self.eps)
                losses.append(1.0 - corr)

        if len(losses) == 0:
            return text_residual.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _spearman_surrogate_loss(self, patch_tokens, part_proj, z_part, obj_mask_patch, part_valid_mask):
        txt_scores = torch.einsum("bkd,bnd->bkn", part_proj, patch_tokens)
        vis_scores = torch.einsum("bkd,bnd->bkn", z_part, patch_tokens)

        losses = []
        B, K, _ = txt_scores.shape
        for b in range(B):
            valid_patch = obj_mask_patch[b]
            if valid_patch.sum() <= 1:
                continue
            for p in range(K):
                if not part_valid_mask[b, p]:
                    continue
                t = txt_scores[b, p, valid_patch]
                v = vis_scores[b, p, valid_patch]
                t = t - t.mean()
                v = v - v.mean()
                denom = torch.sqrt((t ** 2).sum() + self.eps) * torch.sqrt((v ** 2).sum() + self.eps)
                corr = (t * v).sum() / (denom + self.eps)
                losses.append(1.0 - corr)

        if len(losses) == 0:
            return patch_tokens.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _masked_mean(self, x, mask):
        if not mask.any():
            return x.new_tensor(0.0)
        x = x * mask.float()
        return x.sum() / (mask.float().sum() + self.eps)
