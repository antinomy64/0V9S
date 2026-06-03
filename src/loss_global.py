import torch
import torch.nn as nn
import torch.nn.functional as F

class PartLoss(nn.Module):
    def __init__(
        self,
        sim_model,
        lambda_inst: float = 0.2,
        lambda_overlap: float = 0.05,
        lambda_spear: float = 0.0,
        topk_ratio: float = 0.1,
        patch_temperature: float = 0.07,
        eps: float = 1e-6,
        em_iters: int = 3,
        present_only_anchor: bool = False,
    ):
        super().__init__()
        self.sim_model = sim_model
        self.lambda_inst = lambda_inst
        self.lambda_overlap = lambda_overlap
        self.lambda_spear = lambda_spear
        self.topk_ratio = topk_ratio
        self.patch_temperature = patch_temperature
        self.eps = eps
        self.em_iters = int(em_iters)
        self.present_only_anchor = bool(present_only_anchor)

    def _safe_normalize(self, x, dim=-1):
        return x / x.norm(dim=dim, keepdim=True).clamp_min(self.eps)

    def forward(self, batch):
        patch_tokens = batch["patch_tokens"]
        obj_text_feat = batch["obj_text_feat"]
        part_text_feat = batch["part_text_feat"]
        obj_mask_patch = batch["obj_mask_patch"].bool()
        part_valid_mask = batch["part_valid_mask"].bool()
        part_gt_mask_patch = batch["part_gt_mask_patch"].bool()

        if self.present_only_anchor:
            part_present_mask = (part_gt_mask_patch & obj_mask_patch[:, None, :]).sum(dim=-1) > 0
            part_anchor_mask = part_valid_mask & part_present_mask
        else:
            part_anchor_mask = part_valid_mask

        # Project text features into the same space as patch tokens.
        part_proj = self.sim_model.project_clip_txt(part_text_feat.float())   # [B, K, D]
        obj_proj = self.sim_model.project_clip_txt(obj_text_feat.float())     # [B, D]
        part_proj = self._safe_normalize(part_proj, dim=-1)
        obj_proj = self._safe_normalize(obj_proj, dim=-1)
        patch_tokens = self._safe_normalize(patch_tokens.float(), dim=-1)

        # Absolute part-patch score map inside the object.
        abs_logits = torch.einsum("bkd,bnd->bkn", part_proj, patch_tokens) / self.patch_temperature
        abs_logits = abs_logits.masked_fill(~obj_mask_patch[:, None, :], -1e4)

        z_proto, _, anchor_metrics = self._anchor_proto_em_pool(
            patch_tokens=patch_tokens,
            abs_logits=abs_logits,
            obj_mask_patch=obj_mask_patch,
            part_valid_mask=part_anchor_mask,
            part_gt_mask_patch=part_gt_mask_patch,
            num_iters=self.em_iters,
            return_anchor_tokens=False,
        )

        inst_loss = self._instance_consistency_loss(part_proj, z_proto, part_anchor_mask)

        overlap_loss = self._soft_part_overlap_loss(
            abs_logits=abs_logits,
            obj_mask_patch=obj_mask_patch,
            part_valid_mask=part_anchor_mask,
        )

        spear_loss = self._combined_structure_spearman_surrogate_loss(
                obj_text_feat=obj_text_feat,
                part_text_feat=part_text_feat,
                obj_proj=obj_proj,
                part_proj=part_proj,
                part_valid_mask=part_anchor_mask,
            )

        total = (
            + self.lambda_inst * inst_loss
            + self.lambda_overlap * overlap_loss
            + self.lambda_spear * spear_loss
        )

        return {
            "total": total,
            "inst": inst_loss.detach(),
            "overlap": overlap_loss.detach(),
            "spear": spear_loss.detach(),
            "anchor_hit_rate": anchor_metrics["anchor_hit_rate"].detach(),
            "anchor_total_valid_parts": anchor_metrics["anchor_total_valid_parts"].detach(),
            "anchor_total_hits": anchor_metrics["anchor_total_hits"].detach(),
        }

    def _compute_relative_scores(self, local_scores: torch.Tensor) -> torch.Tensor:
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
        abs_logits,
        obj_mask_patch,
        part_valid_mask,
        part_gt_mask_patch,
        num_iters=3,
        return_anchor_tokens: bool = False,
    ):
        B, K, N = abs_logits.shape
        D = patch_tokens.shape[-1]
        z_proto = patch_tokens.new_zeros((B, K, D))
        z_center = patch_tokens.new_zeros((B, K, D))

        total_valid_parts = patch_tokens.new_tensor(0.0)
        total_anchor_hits = patch_tokens.new_tensor(0.0)

        anchor_tokens = patch_tokens.new_zeros((B, K, D))
        anchor_valid = torch.zeros((B, K), dtype=torch.bool, device=patch_tokens.device)

        for b in range(B):
            valid_patch_mask = obj_mask_patch[b]
            valid_part_idx = torch.nonzero(part_valid_mask[b], as_tuple=False).squeeze(1)

            if valid_part_idx.numel() == 0 or valid_patch_mask.sum() == 0:
                continue

            if bool(valid_patch_mask.all()):
                valid_patch_tokens = patch_tokens[b]
                local_scores = abs_logits[b, valid_part_idx]
                gt_masks_local = part_gt_mask_patch[b, valid_part_idx]
            else:
                valid_patch_tokens = patch_tokens[b][valid_patch_mask]
                local_scores = abs_logits[b][valid_part_idx][:, valid_patch_mask]
                gt_masks_local = part_gt_mask_patch[b, valid_part_idx][:, valid_patch_mask]

            Kb, Mb = local_scores.shape
            if Mb == 0:
                continue

            rel_scores = self._compute_relative_scores(local_scores)
            neg_inf = torch.finfo(rel_scores.dtype).min
            masked_scores = rel_scores.clone()

            anchor_idx_local = torch.full((Kb,), -1, dtype=torch.long, device=local_scores.device)

            for _ in range(Kb):
                flat_id = masked_scores.reshape(-1).argmax()
                p_local = torch.div(flat_id, Mb, rounding_mode="floor")
                n_local = flat_id % Mb

                if masked_scores[p_local, n_local] == neg_inf:
                    break

                anchor_idx_local[p_local] = n_local

                # delete selected row and column
                masked_scores[p_local, :] = neg_inf
                masked_scores[:, n_local] = neg_inf

            unassigned = torch.nonzero(anchor_idx_local < 0, as_tuple=False).squeeze(1)
            if unassigned.numel() > 0:
                local_best = rel_scores.argmax(dim=1)
                anchor_idx_local[unassigned] = local_best[unassigned]

            hit_vec = gt_masks_local[
                torch.arange(Kb, device=gt_masks_local.device),
                anchor_idx_local,
            ]

            total_valid_parts += float(Kb)
            total_anchor_hits += float(hit_vec.long().sum().item())

            C = valid_patch_tokens[anchor_idx_local]
            anchor_tokens[b, valid_part_idx] = C
            anchor_valid[b, valid_part_idx] = True

            assign = None
            for _ in range(max(int(num_iters), 1)):
                assign_scores = valid_patch_tokens @ C.T
                assign = assign_scores.argmax(dim=1)
                assign[anchor_idx_local] = torch.arange(Kb, device=assign.device)

                proto_sum = valid_patch_tokens.new_zeros((Kb, valid_patch_tokens.shape[-1]))
                proto_sum.index_add_(0, assign, valid_patch_tokens)

                count = torch.bincount(assign, minlength=Kb).to(valid_patch_tokens.dtype).clamp_min(1.0)

                C = proto_sum / count[:, None]
                C = self._safe_normalize(C, dim=-1)

            z_local = C

            z_center[b, valid_part_idx] = C
            z_proto[b, valid_part_idx] = z_local

        hit_rate = total_anchor_hits / total_valid_parts.clamp_min(1.0)
        anchor_metrics = {
            "anchor_hit_rate": hit_rate,
            "anchor_total_valid_parts": total_valid_parts,
            "anchor_total_hits": total_anchor_hits,
        }
        if return_anchor_tokens:
            return z_proto, z_center, anchor_metrics, anchor_tokens, anchor_valid

        return z_proto, z_center, anchor_metrics

    def _instance_consistency_loss(self, part_proj, z_proto, part_valid_mask):
        cos = F.cosine_similarity(part_proj, z_proto.detach(), dim=-1)
        loss = 1.0 - cos
        return self._masked_mean(loss, part_valid_mask)

    def _corr_loss(self, x, y):
        x = x - x.mean()
        y = y - y.mean()
        denom = (
            torch.sqrt((x ** 2).sum() + self.eps)
            * torch.sqrt((y ** 2).sum() + self.eps)
        )
        corr = (x * y).sum() / (denom + self.eps)
        return 1.0 - corr

    def _part_graph_spearman_surrogate_loss(
        self,
        part_text_feat,
        part_proj,
        part_valid_mask,
    ):
        pre_part = self._safe_normalize(part_text_feat.float(), dim=-1)
        post_part = self._safe_normalize(part_proj.float(), dim=-1)

        losses = []
        B, _, _ = pre_part.shape
        for b in range(B):
            valid_idx = torch.nonzero(part_valid_mask[b], as_tuple=False).squeeze(1)
            Kb = int(valid_idx.numel())
            if Kb < 2:
                continue

            pre_b = pre_part[b, valid_idx]
            post_b = post_part[b, valid_idx]

            pre_sim = pre_b @ pre_b.T
            post_sim = post_b @ post_b.T

            tri = torch.triu_indices(Kb, Kb, offset=1, device=pre_sim.device)
            pre_vec = pre_sim[tri[0], tri[1]]
            post_vec = post_sim[tri[0], tri[1]]

            if pre_vec.numel() < 2:
                continue

            losses.append(self._corr_loss(pre_vec, post_vec))

        if len(losses) == 0:
            return part_proj.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _part_obj_relation_spearman_surrogate_loss(
        self,
        obj_text_feat,
        part_text_feat,
        obj_proj,
        part_proj,
        part_valid_mask,
    ):
        pre_obj = self._safe_normalize(obj_text_feat.float(), dim=-1)     # [B, D_t]
        pre_part = self._safe_normalize(part_text_feat.float(), dim=-1)   # [B, K, D_t]
        post_obj = self._safe_normalize(obj_proj.float(), dim=-1)         # [B, D_v]
        post_part = self._safe_normalize(part_proj.float(), dim=-1)       # [B, K, D_v]

        pre_scores = torch.einsum("bkd,bd->bk", pre_part, pre_obj)
        post_scores = torch.einsum("bkd,bd->bk", post_part, post_obj)

        losses = []
        B, _ = pre_scores.shape
        for b in range(B):
            valid_idx = torch.nonzero(part_valid_mask[b], as_tuple=False).squeeze(1)
            Kb = int(valid_idx.numel())
            if Kb < 2:
                continue

            pre_vec = pre_scores[b, valid_idx]
            post_vec = post_scores[b, valid_idx]

            if pre_vec.numel() < 2:
                continue

            losses.append(self._corr_loss(pre_vec, post_vec))

        if len(losses) == 0:
            return part_proj.new_tensor(0.0)
        return torch.stack(losses).mean()
    
    def _soft_part_overlap_loss(self, abs_logits, obj_mask_patch, part_valid_mask):
        if not part_valid_mask.any():
            return abs_logits.new_tensor(0.0)

        logits = abs_logits.masked_fill(~obj_mask_patch[:, None, :], -1e4)

        # Per-part soft patch distribution inside object mask.
        attn = F.softmax(logits, dim=-1)  # [B, K, N]
        attn = attn * part_valid_mask[:, :, None].float()

        # Pairwise overlap between part attention maps.
        # overlap[b, k, l] is high if part k and part l attend to same patches.
        overlap = torch.einsum("bkn,bln->bkl", attn, attn)  # [B, K, K]

        B, K, _ = overlap.shape
        valid_pair = part_valid_mask[:, :, None] & part_valid_mask[:, None, :]

        eye = torch.eye(K, device=overlap.device, dtype=torch.bool)[None, :, :]
        valid_pair = valid_pair & ~eye

        if not valid_pair.any():
            return abs_logits.new_tensor(0.0)

        return overlap[valid_pair].mean()

    def _combined_structure_spearman_surrogate_loss(
        self,
        obj_text_feat,
        part_text_feat,
        obj_proj,
        part_proj,
        part_valid_mask,
    ):
        graph_loss = self._part_graph_spearman_surrogate_loss(
            part_text_feat=part_text_feat,
            part_proj=part_proj,
            part_valid_mask=part_valid_mask,
        )
        objrel_loss = self._part_obj_relation_spearman_surrogate_loss(
            obj_text_feat=obj_text_feat,
            part_text_feat=part_text_feat,
            obj_proj=obj_proj,
            part_proj=part_proj,
            part_valid_mask=part_valid_mask,
        )

        # Equal-weight combination to keep overall spear scale stable.
        return 0.5 * (graph_loss + objrel_loss)

    def _masked_mean(self, x, mask):
        if not mask.any():
            return x.new_tensor(0.0)
        x = x * mask.float()
        return x.sum() / (mask.float().sum() + self.eps)
