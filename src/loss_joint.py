import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loss import ContrastiveLoss


class JointObjPartLoss(nn.Module):
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
        patch_temperature: float = 0.07,
        eps: float = 1e-6,
        em_iters: int = 3,
        present_only_anchor: bool = False,
        use_dustbin_gate: bool = False,
        dustbin_topk: int = 8,
        dustbin_tau: float = 0.04,
        dustbin_min_active_parts: int = 1,
    ):
        super().__init__()
        self.sim_model = sim_model
        self.obj_criterion = ContrastiveLoss(
            sim_model,
            margin=float(obj_margin),
            max_violation=obj_max_violation,
            ltype=obj_ltype,
        )
        self.lambda_obj = lambda_obj
        self.lambda_inst = lambda_inst
        self.lambda_overlap = lambda_overlap
        self.lambda_spear = lambda_spear
        self.patch_temperature = patch_temperature
        self.eps = eps
        self.em_iters = int(em_iters)
        self.present_only_anchor = bool(present_only_anchor)
        self.use_dustbin_gate = bool(use_dustbin_gate)
        self.dustbin_topk = int(dustbin_topk)
        self.dustbin_tau = float(dustbin_tau)
        self.dustbin_min_active_parts = int(dustbin_min_active_parts)

    def _safe_normalize(self, x, dim=-1):
        return x / x.norm(dim=dim, keepdim=True).clamp_min(self.eps)

    def _empty_dustbin_metrics(self, ref_tensor):
        zero = ref_tensor.new_tensor(0.0)
        return {
            "dustbin_enabled": ref_tensor.new_tensor(1.0 if self.use_dustbin_gate else 0.0),
            "dustbin_tau": ref_tensor.new_tensor(float(self.dustbin_tau)),
            "dustbin_topk": ref_tensor.new_tensor(float(self.dustbin_topk)),
            "dustbin_valid_parts": zero,
            "dustbin_active_parts": zero,
            "dustbin_dropped_parts": zero,
            "dustbin_active_ratio": zero,
            "dustbin_fallback_images": zero,
            "dustbin_score_count": zero,
            "dustbin_score_sum": zero,
            "dustbin_score_sq_sum": zero,
            "dustbin_score_mean": zero,
            "dustbin_score_std": zero,
            "dustbin_score_min": zero,
            "dustbin_score_max": zero,
            "dustbin_gt_present_total": zero,
            "dustbin_gt_present_active": zero,
            "dustbin_gt_present_dropped": zero,
            "dustbin_gt_present_keep_rate": zero,
            "dustbin_gt_absent_total": zero,
            "dustbin_gt_absent_dropped": zero,
            "dustbin_gt_absent_kept": zero,
            "dustbin_gt_absent_drop_rate": zero,
        }

    @torch.no_grad()
    def _compute_dustbin_gate(
        self,
        abs_logits,
        obj_mask_patch,
        part_valid_mask,
        part_gt_mask_patch=None,
    ):
        """Estimate which candidate parts have reliable visual support.

        This is a *training-time absent-part gate*, not an evaluation class.
        It does not introduce background or a new part label.

        The score is computed on cosine scale, not temperature-scaled logit scale:
            presence_score = topk_mean(cosine(part, object_patches)) - mean(cosine(...))

        A part is active iff presence_score > self.dustbin_tau.
        If no part survives for an object, keep the best `dustbin_min_active_parts` parts
        as a fallback so the part branch does not disappear completely.
        """
        B, K, _ = abs_logits.shape
        device = abs_logits.device
        active_part_mask = torch.zeros((B, K), dtype=torch.bool, device=device)

        valid_total = 0.0
        active_total = 0.0
        fallback_images = 0.0

        score_sum = 0.0
        score_sq_sum = 0.0
        score_count = 0.0
        score_min = None
        score_max = None

        present_total = 0.0
        present_active = 0.0
        absent_total = 0.0
        absent_dropped = 0.0

        for b in range(B):
            patch_idx = torch.nonzero(obj_mask_patch[b], as_tuple=False).squeeze(1)
            valid_idx = torch.nonzero(part_valid_mask[b], as_tuple=False).squeeze(1)

            if patch_idx.numel() == 0 or valid_idx.numel() == 0:
                continue

            # abs_logits is cosine / temperature. Convert back to cosine-like scale
            # so tau values such as 0.02/0.04/0.06 are interpretable.
            local_scores = (abs_logits[b, valid_idx][:, patch_idx] * self.patch_temperature).float()

            k_eff = min(max(int(self.dustbin_topk), 1), local_scores.shape[1])
            topk_vals = torch.topk(local_scores, k=k_eff, dim=-1).values
            topk_mean = topk_vals.mean(dim=-1)
            mean_score = local_scores.mean(dim=-1)
            score = topk_mean - mean_score

            keep = score > float(self.dustbin_tau)
            if keep.sum().item() < int(self.dustbin_min_active_parts):
                num_keep = min(int(self.dustbin_min_active_parts), int(valid_idx.numel()))
                if num_keep > 0:
                    best_local = torch.topk(score, k=num_keep, dim=0).indices
                    keep = torch.zeros_like(keep, dtype=torch.bool)
                    keep[best_local] = True
                    fallback_images += 1.0

            active_part_mask[b, valid_idx] = keep

            valid_total += float(valid_idx.numel())
            active_total += float(keep.long().sum().item())

            score_sum += float(score.sum().item())
            score_sq_sum += float((score ** 2).sum().item())
            score_count += float(score.numel())
            cur_min = float(score.min().item())
            cur_max = float(score.max().item())
            score_min = cur_min if score_min is None else min(score_min, cur_min)
            score_max = cur_max if score_max is None else max(score_max, cur_max)

            # GT part masks are used only for audit/debugging, not for gate decisions.
            if part_gt_mask_patch is not None:
                gt_present = (part_gt_mask_patch[b, valid_idx] & obj_mask_patch[b][None, :]).sum(dim=-1) > 0
                present_total += float(gt_present.long().sum().item())
                present_active += float((gt_present & keep).long().sum().item())

                gt_absent = ~gt_present
                absent_total += float(gt_absent.long().sum().item())
                absent_dropped += float((gt_absent & ~keep).long().sum().item())

        ref = abs_logits
        metrics = self._empty_dustbin_metrics(ref)
        metrics.update({
            "dustbin_valid_parts": ref.new_tensor(valid_total),
            "dustbin_active_parts": ref.new_tensor(active_total),
            "dustbin_dropped_parts": ref.new_tensor(max(valid_total - active_total, 0.0)),
            "dustbin_active_ratio": ref.new_tensor(0.0 if valid_total <= 0 else active_total / valid_total),
            "dustbin_fallback_images": ref.new_tensor(fallback_images),
            "dustbin_score_count": ref.new_tensor(score_count),
            "dustbin_score_sum": ref.new_tensor(score_sum),
            "dustbin_score_sq_sum": ref.new_tensor(score_sq_sum),
            "dustbin_score_mean": ref.new_tensor(0.0 if score_count <= 0 else score_sum / score_count),
            "dustbin_score_std": ref.new_tensor(
                0.0 if score_count <= 0 else max(score_sq_sum / score_count - (score_sum / score_count) ** 2, 0.0) ** 0.5
            ),
            "dustbin_score_min": ref.new_tensor(0.0 if score_min is None else score_min),
            "dustbin_score_max": ref.new_tensor(0.0 if score_max is None else score_max),
            "dustbin_gt_present_total": ref.new_tensor(present_total),
            "dustbin_gt_present_active": ref.new_tensor(present_active),
            "dustbin_gt_present_dropped": ref.new_tensor(max(present_total - present_active, 0.0)),
            "dustbin_gt_present_keep_rate": ref.new_tensor(0.0 if present_total <= 0 else present_active / present_total),
            "dustbin_gt_absent_total": ref.new_tensor(absent_total),
            "dustbin_gt_absent_dropped": ref.new_tensor(absent_dropped),
            "dustbin_gt_absent_kept": ref.new_tensor(max(absent_total - absent_dropped, 0.0)),
            "dustbin_gt_absent_drop_rate": ref.new_tensor(0.0 if absent_total <= 0 else absent_dropped / absent_total),
        })
        return active_part_mask, metrics

    def forward(self, batch):
        obj_feat = batch["obj_feat"]
        patch_tokens = batch["patch_tokens"]
        obj_text_feat = batch["obj_text_feat"]
        part_text_feat = batch["part_text_feat"]
        obj_mask_patch = batch["obj_mask_patch"].bool()
        part_valid_mask = batch["part_valid_mask"].bool()
        part_gt_mask_patch = batch["part_gt_mask_patch"].bool()

        if getattr(self, "present_only_anchor", False):
            part_present_mask = (part_gt_mask_patch & obj_mask_patch[:, None, :]).sum(dim=-1) > 0
            part_anchor_mask = part_valid_mask & part_present_mask
        else:
            part_anchor_mask = part_valid_mask

        has_obj_patch = obj_mask_patch.any(dim=-1, keepdim=True)  # [B, 1]
        part_train_mask = part_anchor_mask & has_obj_patch        # [B, K]

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

        if part_text_feat.shape[1] == 0 or not part_train_mask.any():
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
                **{k: v.detach() for k, v in self._empty_dustbin_metrics(zero).items()},
            }

        # Project text features into the same space as patch tokens.
        part_proj = self.sim_model.project_clip_txt(part_text_feat.float())   # [B, K, D]
        obj_proj = self.sim_model.project_clip_txt(obj_text_feat.float())     # [B, D]
        part_proj = self._safe_normalize(part_proj, dim=-1)
        obj_proj = self._safe_normalize(obj_proj, dim=-1)
        patch_tokens = self._safe_normalize(patch_tokens.float(), dim=-1)

        # Absolute part-patch score map inside the object.
        abs_logits = torch.einsum("bkd,bnd->bkn", part_proj, patch_tokens) / self.patch_temperature
        abs_logits = abs_logits.masked_fill(~obj_mask_patch[:, None, :], -1e4)

        dustbin_metrics = self._empty_dustbin_metrics(abs_logits)
        if getattr(self, "use_dustbin_gate", False):
            active_part_mask, dustbin_metrics = self._compute_dustbin_gate(
                abs_logits=abs_logits.detach(),
                obj_mask_patch=obj_mask_patch,
                part_valid_mask=part_train_mask,
                part_gt_mask_patch=part_gt_mask_patch,
            )
            part_train_mask = part_train_mask & active_part_mask

        if not part_train_mask.any():
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
                **{k: v.detach() for k, v in dustbin_metrics.items()},
            }

        z_part, _, anchor_metrics = self._anchor_proto_em_pool(
            patch_tokens=patch_tokens,
            abs_logits=abs_logits,
            obj_mask_patch=obj_mask_patch,
            part_valid_mask=part_train_mask,
            part_gt_mask_patch=part_gt_mask_patch,
            num_iters=self.em_iters,
        )

        inst_loss = self._instance_consistency_loss(part_proj, z_part, part_train_mask)

        overlap_loss = (
            self._soft_part_overlap_loss(
                abs_logits=abs_logits,
                obj_mask_patch=obj_mask_patch,
                part_valid_mask=part_train_mask,
            )
            if self.lambda_overlap > 0
            else zero
        )

        # New structure-preserving "Spearman-style" loss:
        #   1) keep the part-part text graph stable before/after projection
        #   2) keep the part-obj relation stable before/after projection
        spear_loss = (
            self._combined_structure_spearman_surrogate_loss(
                obj_text_feat=obj_text_feat,
                part_text_feat=part_text_feat,
                obj_proj=obj_proj,
                part_proj=part_proj,
                part_valid_mask=part_train_mask,
            )
            if self.lambda_spear > 0
            else zero
        )

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
            **{k: v.detach() for k, v in dustbin_metrics.items()},
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
        z = patch_tokens.new_zeros((B, K, D))
        proto_part = patch_tokens.new_zeros((B, K, D))

        total_valid_parts = patch_tokens.new_tensor(0.0)
        total_anchor_hits = patch_tokens.new_tensor(0.0)

        anchor_tokens = patch_tokens.new_zeros((B, K, D))
        anchor_valid = torch.zeros((B, K), dtype=torch.bool, device=patch_tokens.device)

        for b in range(B):
            valid_patch_mask = obj_mask_patch[b]
            valid_part_idx = torch.nonzero(part_valid_mask[b], as_tuple=False).squeeze(1)

            if valid_part_idx.numel() == 0 or valid_patch_mask.sum() == 0:
                continue

            valid_patch_tokens = patch_tokens[b][valid_patch_mask]
            local_scores = abs_logits[b][valid_part_idx][:, valid_patch_mask]

            Kb, Mb = local_scores.shape
            if Mb == 0:
                continue

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

            unassigned = torch.nonzero(anchor_idx_local < 0, as_tuple=False).squeeze(1)
            if unassigned.numel() > 0:
                local_best = rel_scores.argmax(dim=1)
                anchor_idx_local[unassigned] = local_best[unassigned]

            valid_patch_idx_global = torch.nonzero(valid_patch_mask, as_tuple=False).squeeze(1)
            anchor_idx_global = valid_patch_idx_global[anchor_idx_local]

            gt_masks = part_gt_mask_patch[b, valid_part_idx]
            hit_vec = gt_masks[torch.arange(Kb, device=gt_masks.device), anchor_idx_global]

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

                onehot = F.one_hot(assign, num_classes=Kb).float()
                count = onehot.sum(dim=0).clamp_min(1.0)
                proto_sum = onehot.T @ valid_patch_tokens
                C = proto_sum / count[:, None]
                C = self._safe_normalize(C, dim=-1)

            region_onehot = F.one_hot(assign, num_classes=Kb).float()
            region_count = region_onehot.sum(dim=0).clamp_min(1.0)
            region_sum = region_onehot.T @ valid_patch_tokens
            z_local = region_sum / region_count[:, None]
            z_local = self._safe_normalize(z_local, dim=-1)

            z[b, valid_part_idx] = z_local
            proto_part[b, valid_part_idx] = C

        hit_rate = total_anchor_hits / total_valid_parts.clamp_min(1.0)
        anchor_metrics = {
            "anchor_hit_rate": hit_rate,
            "anchor_total_valid_parts": total_valid_parts,
            "anchor_total_hits": total_anchor_hits,
        }
        if return_anchor_tokens:
            return z, proto_part, anchor_metrics, anchor_tokens, anchor_valid

        return z, proto_part, anchor_metrics

    def _instance_consistency_loss(self, part_proj, z_part, part_valid_mask):
        cos = F.cosine_similarity(part_proj, z_part.detach(), dim=-1)
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
