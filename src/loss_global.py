import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment


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
        anchor_matcher: str = "greedy",
        anchor_score_type: str = "relative",
    ):
        super().__init__()
        self.sim_model = sim_model
        self.lambda_inst = float(lambda_inst)
        self.lambda_overlap = float(lambda_overlap)
        self.lambda_spear = float(lambda_spear)
        self.topk_ratio = float(topk_ratio)
        self.patch_temperature = float(patch_temperature)
        self.eps = float(eps)
        self.em_iters = int(em_iters)
        self.present_only_anchor = bool(present_only_anchor)

        self.anchor_matcher = str(anchor_matcher).lower()
        self.anchor_score_type = str(anchor_score_type).lower()
        self._validate_anchor_options()

    def _validate_anchor_options(self):
        if self.anchor_matcher not in {"hungarian", "greedy"}:
            raise ValueError(
                f"anchor_matcher must be 'hungarian' or 'greedy', got {self.anchor_matcher!r}"
            )
        if self.anchor_score_type not in {"absolute", "relative"}:
            raise ValueError(
                f"anchor_score_type must be 'absolute' or 'relative', got {self.anchor_score_type!r}"
            )
        if self.anchor_matcher == "hungarian" and linear_sum_assignment is None:
            raise ImportError(
                "scipy is required for anchor_matcher='hungarian'. "
                "Install it with: pip install scipy"
            )

    def _safe_normalize(self, x, dim=-1):
        return x / x.norm(dim=dim, keepdim=True).clamp_min(self.eps)

    def _build_part_anchor_mask(self, part_valid_mask, part_gt_mask_patch, obj_mask_patch):
        if self.present_only_anchor:
            part_present_mask = (part_gt_mask_patch & obj_mask_patch[:, None, :]).sum(dim=-1) > 0
            return part_valid_mask & part_present_mask
        return part_valid_mask

    def _empty_output(self, obj_text_feat):
        zero = self.sim_model.project_clip_txt(obj_text_feat.float()).sum() * 0.0
        return {
            "total": zero,
            "inst": zero.detach(),
            "overlap": zero.detach(),
            "spear": zero.detach(),
            "anchor_hit_rate": zero.detach(),
            "anchor_total_valid_parts": zero.detach(),
            "anchor_total_hits": zero.detach(),
        }

    def forward(self, batch):
        patch_tokens = batch["patch_tokens"]
        obj_text_feat = batch["obj_text_feat"]
        part_text_feat = batch["part_text_feat"]
        obj_mask_patch = batch["obj_mask_patch"].bool()
        part_valid_mask = batch["part_valid_mask"].bool()
        part_gt_mask_patch = batch["part_gt_mask_patch"].bool()

        part_anchor_mask = self._build_part_anchor_mask(
            part_valid_mask=part_valid_mask,
            part_gt_mask_patch=part_gt_mask_patch,
            obj_mask_patch=obj_mask_patch,
        )

        # Robustness guard for empty categories / empty sampled pools.
        if part_text_feat.shape[1] == 0 or not part_anchor_mask.any() or not obj_mask_patch.any():
            return self._empty_output(obj_text_feat)

        part_proj = self.sim_model.project_clip_txt(part_text_feat.float())  # [B, K, D]
        obj_proj = self.sim_model.project_clip_txt(obj_text_feat.float())    # [B, D]
        part_proj = self._safe_normalize(part_proj, dim=-1)
        obj_proj = self._safe_normalize(obj_proj, dim=-1)
        patch_tokens = self._safe_normalize(patch_tokens.float(), dim=-1)

        abs_logits = torch.einsum("bkd,bnd->bkn", part_proj, patch_tokens)
        abs_logits = abs_logits / self.patch_temperature
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
            self.lambda_inst * inst_loss
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

    @torch.no_grad()
    def audit_anchor_and_proto(self, batch):
        """
        Audit anchor and EM pseudo prototype semantics.

        GT masks are used only here for reporting:
          - which GT part covers the selected anchor patch;
          - which GT part prototype is nearest to the EM pseudo prototype.
        """
        patch_tokens = batch["patch_tokens"]
        part_text_feat = batch["part_text_feat"]
        obj_mask_patch = batch["obj_mask_patch"].bool()
        part_valid_mask = batch["part_valid_mask"].bool()
        part_gt_mask_patch = batch["part_gt_mask_patch"].bool()
        part_category_id = batch["part_category_id"].long()
        category_id = batch.get("category_id", None)

        if part_text_feat.shape[1] == 0 or not part_valid_mask.any():
            return []

        part_anchor_mask = self._build_part_anchor_mask(
            part_valid_mask=part_valid_mask,
            part_gt_mask_patch=part_gt_mask_patch,
            obj_mask_patch=obj_mask_patch,
        )
        if not part_anchor_mask.any():
            return []

        part_proj = self.sim_model.project_clip_txt(part_text_feat.float())
        part_proj = self._safe_normalize(part_proj, dim=-1)
        patch_tokens = self._safe_normalize(patch_tokens.float(), dim=-1)

        abs_logits = torch.einsum("bkd,bnd->bkn", part_proj, patch_tokens)
        abs_logits = abs_logits / self.patch_temperature
        abs_logits = abs_logits.masked_fill(~obj_mask_patch[:, None, :], -1e4)

        z_proto, _, _, anchor_indices, anchor_valid = self._anchor_proto_em_pool(
            patch_tokens=patch_tokens,
            abs_logits=abs_logits,
            obj_mask_patch=obj_mask_patch,
            part_valid_mask=part_anchor_mask,
            part_gt_mask_patch=part_gt_mask_patch,
            num_iters=self.em_iters,
            return_anchor_indices=True,
        )

        records = []
        B, K, _ = z_proto.shape

        for b in range(B):
            valid_gt_slots = []
            valid_gt_ids = []
            valid_gt_protos = []

            for j in range(K):
                if not bool(part_valid_mask[b, j].item()):
                    continue

                gt_mask = part_gt_mask_patch[b, j] & obj_mask_patch[b]
                if not bool(gt_mask.any().item()):
                    continue

                gt_proto = patch_tokens[b, gt_mask].mean(dim=0)
                gt_proto = self._safe_normalize(gt_proto, dim=-1)

                valid_gt_slots.append(j)
                valid_gt_ids.append(int(part_category_id[b, j].item()))
                valid_gt_protos.append(gt_proto)

            gt_proto_mat = torch.stack(valid_gt_protos, dim=0) if valid_gt_protos else None
            cat_id = -1 if category_id is None else int(category_id[b].item())

            for k in range(K):
                if not bool(anchor_valid[b, k].item()):
                    continue

                anchor_idx = int(anchor_indices[b, k].item())
                if anchor_idx < 0:
                    continue

                anchor_hit_ids = []
                for j in valid_gt_slots:
                    if bool(part_gt_mask_patch[b, j, anchor_idx].item()):
                        anchor_hit_ids.append(int(part_category_id[b, j].item()))

                nearest_gt_id = -1
                nearest_gt_cos = float("nan")
                if gt_proto_mat is not None:
                    sims = z_proto[b, k] @ gt_proto_mat.T
                    nearest_local = int(sims.argmax().item())
                    nearest_gt_id = valid_gt_ids[nearest_local]
                    nearest_gt_cos = float(sims[nearest_local].item())

                records.append(
                    {
                        "category_id": cat_id,
                        "target_part_id": int(part_category_id[b, k].item()),
                        "anchor_gt_part_ids": sorted(set(anchor_hit_ids)),
                        "pseudo_proto_nearest_gt_part_id": nearest_gt_id,
                        "pseudo_proto_nearest_gt_cos": nearest_gt_cos,
                    }
                )

        return records

    def _compute_relative_scores(self, local_scores: torch.Tensor) -> torch.Tensor:
        """
        Relative score used by the old anchor selector.

        rel_score(part_i, patch_n) =
            score(part_i, patch_n) - best score of any other part on patch_n
        """
        Kb, _ = local_scores.shape
        if Kb <= 1:
            return local_scores

        top2_vals, top2_idx = torch.topk(local_scores, k=min(2, Kb), dim=0)
        best_vals = top2_vals[0]
        best_idx = top2_idx[0]
        second_vals = top2_vals[1]

        row_ids = torch.arange(Kb, device=local_scores.device)[:, None]
        is_top1 = row_ids == best_idx[None, :]
        best_other = torch.where(is_top1, second_vals[None, :], best_vals[None, :])

        return local_scores - best_other

    def _anchor_match_scores(self, local_scores: torch.Tensor) -> torch.Tensor:
        if self.anchor_score_type == "absolute":
            return local_scores
        if self.anchor_score_type == "relative":
            return self._compute_relative_scores(local_scores)
        raise RuntimeError(f"Invalid anchor_score_type: {self.anchor_score_type}")

    def _select_anchor_indices(self, match_scores: torch.Tensor) -> torch.Tensor:
        if self.anchor_matcher == "hungarian":
            return self._select_anchor_indices_hungarian(match_scores)
        if self.anchor_matcher == "greedy":
            return self._select_anchor_indices_greedy(match_scores)
        raise RuntimeError(f"Invalid anchor_matcher: {self.anchor_matcher}")

    def _select_anchor_indices_hungarian(self, match_scores: torch.Tensor) -> torch.Tensor:
        """
        One-to-one part-anchor assignment.

        Args:
            match_scores: [Kb, Mb], higher is better.

        Returns:
            anchor_idx_local: [Kb], local patch index for each local part.
        """
        Kb, Mb = match_scores.shape
        device = match_scores.device

        anchor_idx_local = torch.full((Kb,), -1, dtype=torch.long, device=device)
        if Kb == 0 or Mb == 0:
            return anchor_idx_local

        cost = -match_scores.detach().float().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)

        row_ind = torch.as_tensor(row_ind, dtype=torch.long, device=device)
        col_ind = torch.as_tensor(col_ind, dtype=torch.long, device=device)
        anchor_idx_local[row_ind] = col_ind

        # If Mb < Kb, rectangular Hungarian can only assign Mb rows.
        # Fill the remaining parts with their individual best patch, allowing duplicates.
        unassigned = torch.nonzero(anchor_idx_local < 0, as_tuple=False).squeeze(1)
        if unassigned.numel() > 0:
            local_best = match_scores.argmax(dim=1)
            anchor_idx_local[unassigned] = local_best[unassigned]

        return anchor_idx_local

    def _select_anchor_indices_greedy(self, match_scores: torch.Tensor) -> torch.Tensor:
        """
        Old greedy row/column deletion, kept for ablation.
        """
        Kb, Mb = match_scores.shape
        device = match_scores.device

        anchor_idx_local = torch.full((Kb,), -1, dtype=torch.long, device=device)
        if Kb == 0 or Mb == 0:
            return anchor_idx_local

        neg_inf = torch.finfo(match_scores.dtype).min
        masked_scores = match_scores.clone()

        for _ in range(min(Kb, Mb)):
            flat_id = masked_scores.reshape(-1).argmax()
            p_local = torch.div(flat_id, Mb, rounding_mode="floor")
            n_local = flat_id % Mb

            if masked_scores[p_local, n_local] == neg_inf:
                break

            anchor_idx_local[p_local] = n_local
            masked_scores[p_local, :] = neg_inf
            masked_scores[:, n_local] = neg_inf

        unassigned = torch.nonzero(anchor_idx_local < 0, as_tuple=False).squeeze(1)
        if unassigned.numel() > 0:
            local_best = match_scores.argmax(dim=1)
            anchor_idx_local[unassigned] = local_best[unassigned]

        return anchor_idx_local

    def _slice_valid_pool(
        self,
        patch_tokens_b,
        abs_logits_b,
        obj_mask_patch_b,
        part_gt_mask_patch_b,
        valid_part_idx,
    ):
        """
        Return the valid object-pool tokens and corresponding local score/mask tensors.
        """
        if bool(obj_mask_patch_b.all()):
            valid_patch_tokens = patch_tokens_b
            local_scores = abs_logits_b[valid_part_idx]
            gt_masks_local = part_gt_mask_patch_b[valid_part_idx]
            valid_patch_idx_global = torch.arange(
                patch_tokens_b.shape[0],
                device=patch_tokens_b.device,
            )
        else:
            valid_patch_tokens = patch_tokens_b[obj_mask_patch_b]
            local_scores = abs_logits_b[valid_part_idx][:, obj_mask_patch_b]
            gt_masks_local = part_gt_mask_patch_b[valid_part_idx][:, obj_mask_patch_b]
            valid_patch_idx_global = torch.nonzero(
                obj_mask_patch_b,
                as_tuple=False,
            ).squeeze(1)

        return valid_patch_tokens, local_scores, gt_masks_local, valid_patch_idx_global

    def _run_hard_em(self, valid_patch_tokens, init_centers, anchor_idx_local, num_iters):
        C = init_centers
        Kb = int(C.shape[0])

        for _ in range(max(int(num_iters), 1)):
            assign_scores = valid_patch_tokens @ C.T
            assign = assign_scores.argmax(dim=1)

            # Keep each anchor patch assigned to its own center.
            assign[anchor_idx_local] = torch.arange(Kb, device=assign.device)

            proto_sum = valid_patch_tokens.new_zeros((Kb, valid_patch_tokens.shape[-1]))
            proto_sum.index_add_(0, assign, valid_patch_tokens)

            count = torch.bincount(
                assign,
                minlength=Kb,
            ).to(valid_patch_tokens.dtype).clamp_min(1.0)

            C = proto_sum / count[:, None]
            C = self._safe_normalize(C, dim=-1)

        return C

    def _anchor_proto_em_pool(
        self,
        patch_tokens,
        abs_logits,
        obj_mask_patch,
        part_valid_mask,
        part_gt_mask_patch,
        num_iters=3,
        return_anchor_tokens: bool = False,
        return_anchor_indices: bool = False,
    ):
        B, K, _ = abs_logits.shape
        D = patch_tokens.shape[-1]

        z_proto = patch_tokens.new_zeros((B, K, D))
        z_center = patch_tokens.new_zeros((B, K, D))

        total_valid_parts = patch_tokens.new_tensor(0.0)
        total_anchor_hits = patch_tokens.new_tensor(0.0)

        anchor_tokens = patch_tokens.new_zeros((B, K, D))
        anchor_indices = torch.full(
            (B, K),
            -1,
            dtype=torch.long,
            device=patch_tokens.device,
        )
        anchor_valid = torch.zeros((B, K), dtype=torch.bool, device=patch_tokens.device)

        for b in range(B):
            valid_part_idx = torch.nonzero(part_valid_mask[b], as_tuple=False).squeeze(1)
            if valid_part_idx.numel() == 0 or valid_part_idx.numel() > int(obj_mask_patch[b].sum().item()):
                # The second condition is only a safety fallback:
                # if there are fewer valid patches than parts, Hungarian cannot
                # provide a unique anchor for every part. The selector itself
                # still handles this, so we do not skip.
                pass

            if valid_part_idx.numel() == 0 or obj_mask_patch[b].sum() == 0:
                continue

            (
                valid_patch_tokens,
                local_scores,
                gt_masks_local,
                valid_patch_idx_global,
            ) = self._slice_valid_pool(
                patch_tokens_b=patch_tokens[b],
                abs_logits_b=abs_logits[b],
                obj_mask_patch_b=obj_mask_patch[b],
                part_gt_mask_patch_b=part_gt_mask_patch[b],
                valid_part_idx=valid_part_idx,
            )

            Kb, Mb = local_scores.shape
            if Kb == 0 or Mb == 0:
                continue

            match_scores = self._anchor_match_scores(local_scores)
            anchor_idx_local = self._select_anchor_indices(match_scores)

            hit_vec = gt_masks_local[
                torch.arange(Kb, device=gt_masks_local.device),
                anchor_idx_local,
            ]

            total_valid_parts += float(Kb)
            total_anchor_hits += float(hit_vec.long().sum().item())

            anchor_idx_global = valid_patch_idx_global[anchor_idx_local]

            C0 = valid_patch_tokens[anchor_idx_local]
            C = self._run_hard_em(
                valid_patch_tokens=valid_patch_tokens,
                init_centers=C0,
                anchor_idx_local=anchor_idx_local,
                num_iters=num_iters,
            )

            z_center[b, valid_part_idx] = C
            z_proto[b, valid_part_idx] = C

            anchor_tokens[b, valid_part_idx] = C0
            anchor_indices[b, valid_part_idx] = anchor_idx_global
            anchor_valid[b, valid_part_idx] = True

        hit_rate = total_anchor_hits / total_valid_parts.clamp_min(1.0)
        anchor_metrics = {
            "anchor_hit_rate": hit_rate,
            "anchor_total_valid_parts": total_valid_parts,
            "anchor_total_hits": total_anchor_hits,
        }

        if return_anchor_tokens and return_anchor_indices:
            return (
                z_proto,
                z_center,
                anchor_metrics,
                anchor_tokens,
                anchor_indices,
                anchor_valid,
            )
        if return_anchor_tokens:
            return z_proto, z_center, anchor_metrics, anchor_tokens, anchor_valid
        if return_anchor_indices:
            return z_proto, z_center, anchor_metrics, anchor_indices, anchor_valid
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
        pre_obj = self._safe_normalize(obj_text_feat.float(), dim=-1)
        pre_part = self._safe_normalize(part_text_feat.float(), dim=-1)
        post_obj = self._safe_normalize(obj_proj.float(), dim=-1)
        post_part = self._safe_normalize(part_proj.float(), dim=-1)

        pre_scores = torch.einsum("bkd,bd->bk", pre_part, pre_obj)
        post_scores = torch.einsum("bkd,bd->bk", post_part, post_obj)

        losses = []
        B, _ = pre_scores.shape
        for b in range(B):
            valid_idx = torch.nonzero(part_valid_mask[b], as_tuple=False).squeeze(1)
            if valid_idx.numel() < 2:
                continue

            losses.append(self._corr_loss(pre_scores[b, valid_idx], post_scores[b, valid_idx]))

        if len(losses) == 0:
            return part_proj.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _soft_part_overlap_loss(self, abs_logits, obj_mask_patch, part_valid_mask):
        if not part_valid_mask.any():
            return abs_logits.new_tensor(0.0)

        logits = abs_logits.masked_fill(~obj_mask_patch[:, None, :], -1e4)

        # Per-part soft patch distribution inside object mask.
        attn = F.softmax(logits, dim=-1)
        attn = attn * part_valid_mask[:, :, None].float()

        # Pairwise overlap between part attention maps.
        overlap = torch.einsum("bkn,bln->bkl", attn, attn)

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
        return 0.5 * (graph_loss + objrel_loss)

    def _masked_mean(self, x, mask):
        if not mask.any():
            return x.new_tensor(0.0)
        return (x * mask.float()).sum() / (mask.float().sum() + self.eps)
