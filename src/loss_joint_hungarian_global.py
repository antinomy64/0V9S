
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from src.loss import ContrastiveLoss


class JointObjPartLoss(nn.Module):
    """
    Joint loss for:
      1) object-level branch: keep original contrastive objective
      2) class-global part branch inside a batch

    User-requested pseudo-label logic:
      - for each class in the batch, collect all object patches from all images
      - use ALL class part text features from dataset-level class_part_bank
      - build global similarity matrix S = T_c P_c^T / tau
      - Hungarian matching first finds UNIQUE anchor patches
      - then every patch is assigned to its nearest anchor (no duplicate patch ownership)
      - prototype of each matched part = mean(anchor patch + its assigned patches)
      - keep obj loss / global spear / total loss structure unchanged

    Current tweak:
      - anchor selection uses RELATIVE scores instead of raw absolute scores
      - region growing / patch-to-anchor assignment remains unchanged
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
        class_part_bank=None,
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
        self.class_part_bank = class_part_bank if class_part_bank is not None else {}

    def forward(self, batch):
        obj_feat = batch["obj_feat"]
        patch_tokens = batch["patch_tokens"]
        obj_text_feat = batch["obj_text_feat"]
        obj_mask_patch = batch["obj_mask_patch"]
        category_id = batch["category_id"]

        obj_loss = self.obj_criterion(
            obj_feat,
            obj_text_feat,
            return_similarity_mat=False,
            self_attn_maps=None,
            cls=None,
            text_input_mask=None,
            text_argmax=None,
        )

        if not self.class_part_bank:
            zero = obj_loss.new_tensor(0.0)
            total = self.lambda_obj * obj_loss
            return {
                "total": total,
                "obj": obj_loss.detach(),
                "inst": zero.detach(),
                "overlap": zero.detach(),
                "spear": zero.detach(),
            }

        patch_tokens = F.normalize(patch_tokens.float(), dim=-1)
        obj_mask_patch = obj_mask_patch.bool()

        inst_loss, overlap_loss, spear_loss = self._class_global_part_losses(
            patch_tokens=patch_tokens,
            obj_mask_patch=obj_mask_patch,
            category_id=category_id,
            device=patch_tokens.device,
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
        }

    def _project_bank_text(self, raw_part_text_feat: torch.Tensor) -> torch.Tensor:
        proj = self.sim_model.project_clip_txt(raw_part_text_feat.float())
        return F.normalize(proj.float(), dim=-1)

    def _compute_relative_scores(self, S: torch.Tensor) -> torch.Tensor:
        """
        S: [K, M] absolute similarity
        relative score for (k, m):
            S[k, m] - best competing part score on the same patch m

        This favors patches that are not only high-score for part k,
        but also uniquely high compared with other parts.
        """
        K, M = S.shape
        if K <= 1:
            return S

        top2_vals, top2_idx = torch.topk(S, k=min(2, K), dim=0)  # over parts, for each patch
        best_vals = top2_vals[0]     # [M]
        best_idx = top2_idx[0]       # [M]
        second_vals = top2_vals[1]   # [M]

        row_ids = torch.arange(K, device=S.device)[:, None]      # [K, 1]
        is_top1 = row_ids == best_idx[None, :]                   # [K, M]

        # If current part is the top-1 part on patch m, subtract the second-best score.
        # Otherwise subtract the top-1 score.
        best_other = torch.where(is_top1, second_vals[None, :], best_vals[None, :])  # [K, M]
        return S - best_other

    def _class_global_part_losses(self, patch_tokens, obj_mask_patch, category_id, device):
        unique_classes = torch.unique(category_id)

        inst_terms = []
        overlap_terms = []
        spear_terms = []

        for cls in unique_classes.tolist():
            sample_idx = torch.nonzero(category_id == cls, as_tuple=False).squeeze(1)
            if sample_idx.numel() == 0:
                continue

            # global patch pool from all object patches of this class within current batch
            patch_list = []
            for b in sample_idx.tolist():
                valid_patch = obj_mask_patch[b]
                if valid_patch.any():
                    patch_list.append(patch_tokens[b][valid_patch])
            if len(patch_list) == 0:
                continue

            P = torch.cat(patch_list, dim=0)  # [M, D]
            M = P.shape[0]
            if M == 0:
                continue

            # use ALL part text features of this class from dataset-level bank
            cls_int = int(cls)
            bank_entry = self.class_part_bank.get(cls_int, None)
            if bank_entry is None:
                continue

            raw_T = bank_entry["part_text_feat"].to(device).float()
            if raw_T.numel() == 0:
                continue
            T = self._project_bank_text(raw_T)  # [K, D]
            K = T.shape[0]
            if K == 0:
                continue

            # similarity matrix S = T P^T / tau
            S = (T @ P.T) / self.patch_temperature  # [K, M]

            Z, assign_global, matched_mask = self._hungarian_anchor_region_pool(P, S)

            if matched_mask.sum() == 0:
                continue

            T_used = T[matched_mask]
            Z_used = Z[matched_mask]

            inst = 1.0 - F.cosine_similarity(T_used, Z_used, dim=-1)
            inst_terms.append(inst.mean())

            if matched_mask.sum() > 1:
                q = F.one_hot(assign_global, num_classes=K).float().T  # [K, M]
                q_used = q[matched_mask]
                overlap_terms.append(self._assignment_overlap_loss(q_used))

            if self.lambda_spear > 0:
                spear_terms.append(self._global_spearman_surrogate_loss(T_used, Z_used.detach(), P))

        zero = patch_tokens.new_tensor(0.0)
        inst_loss = torch.stack(inst_terms).mean() if len(inst_terms) > 0 else zero
        overlap_loss = torch.stack(overlap_terms).mean() if len(overlap_terms) > 0 else zero
        spear_loss = torch.stack(spear_terms).mean() if len(spear_terms) > 0 else zero
        return inst_loss, overlap_loss, spear_loss

    def _hungarian_anchor_region_pool(self, P: torch.Tensor, S: torch.Tensor):
        """
        P: [M, D] global patch pool for one class in current batch
        S: [K, M] text-patch similarity

        Steps:
          1) Hungarian matching on RELATIVE scores chooses UNIQUE anchor patch for each matched part
          2) every patch is assigned to its nearest anchor (strict one-owner partition, no duplicate patches)
          3) prototype of each matched part = mean of anchor patch + its assigned patches

        Returns:
          Z: [K, D] prototypes, zeros for unmatched parts
          assign_global: [M] global part index for each patch
          matched_mask: [K] whether this part obtained an anchor from Hungarian
        """
        K, M = S.shape
        D = P.shape[-1]

        Z = P.new_zeros((K, D))
        if K == 0 or M == 0:
            assign_global = torch.zeros((M,), dtype=torch.long, device=P.device)
            matched_mask = torch.zeros((K,), dtype=torch.bool, device=P.device)
            return Z, assign_global, matched_mask

        # Only change: use relative scores for anchor selection
        S_rel = self._compute_relative_scores(S)  # [K, M]

        # Hungarian on relative similarity: maximize S_rel <=> minimize -S_rel
        cost = (-S_rel).detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)

        matched_part_idx = torch.as_tensor(row_ind, device=P.device, dtype=torch.long)   # [L]
        anchor_patch_idx = torch.as_tensor(col_ind, device=P.device, dtype=torch.long)    # [L]
        L = matched_part_idx.numel()

        matched_mask = torch.zeros((K,), dtype=torch.bool, device=P.device)
        matched_mask[matched_part_idx] = True

        if L == 0:
            assign_global = torch.zeros((M,), dtype=torch.long, device=P.device)
            return Z, assign_global, matched_mask

        anchor_feats = P[anchor_patch_idx]  # [L, D]

        # Assign every patch to nearest anchor (keep the rest unchanged)
        region_scores = P @ anchor_feats.T                 # [M, L]
        region_assign_local = region_scores.argmax(dim=1)  # [M]

        # Force each anchor patch to remain in its own anchor region
        region_assign_local[anchor_patch_idx] = torch.arange(L, device=P.device)

        # Map local anchor-region ids back to global part ids
        assign_global = matched_part_idx[region_assign_local]  # [M]

        # Mean pooling per matched part region
        region_onehot = F.one_hot(region_assign_local, num_classes=L).float()  # [M, L]
        region_count = region_onehot.sum(dim=0).clamp_min(1.0)                 # [L]
        region_sum = region_onehot.T @ P                                       # [L, D]
        z_matched = region_sum / region_count[:, None]                         # [L, D]
        z_matched = F.normalize(z_matched, dim=-1)

        Z[matched_part_idx] = z_matched
        return Z, assign_global, matched_mask

    def _assignment_overlap_loss(self, q: torch.Tensor) -> torch.Tensor:
        """
        q: [K_used, M] one-hot assignment matrix over matched parts.
        Under strict partition, overlap is naturally very small / zero.
        """
        K = q.shape[0]
        pair_losses = []
        for p in range(K):
            for r in range(p + 1, K):
                qp = q[p]
                qr = q[r]
                num = (qp * qr).sum()
                den = torch.sqrt((qp ** 2).sum() + self.eps) * torch.sqrt((qr ** 2).sum() + self.eps)
                pair_losses.append(num / (den + self.eps))
        if len(pair_losses) == 0:
            return q.new_tensor(0.0)
        return torch.stack(pair_losses).mean()

    def _global_spearman_surrogate_loss(self, T: torch.Tensor, Z: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        txt_scores = T @ P.T
        vis_scores = Z @ P.T

        losses = []
        K = txt_scores.shape[0]
        for k in range(K):
            t = txt_scores[k]
            v = vis_scores[k]
            if t.numel() <= 1:
                continue
            t = t - t.mean()
            v = v - v.mean()
            denom = torch.sqrt((t ** 2).sum() + self.eps) * torch.sqrt((v ** 2).sum() + self.eps)
            corr = (t * v).sum() / (denom + self.eps)
            losses.append(1.0 - corr)

        if len(losses) == 0:
            return P.new_tensor(0.0)
        return torch.stack(losses).mean()
