"""
Label-aware contrastive loss for multi-label CLIP fine-tuning on MIMIC-CXR.

WHY THIS EXISTS
---------------
Standard CLIP uses a diagonal target: in a batch of B (image, text) pairs,
sample i is the one and only positive for itself, and every other sample is
treated as a negative.  That is wrong for multi-label medical data.

Example: image A and image B both show "Edema and Pleural Effusion".  Standard
CLIP tries to PUSH image A away from text B — exactly the opposite of what we
want.  Label-aware loss fixes this by identifying which pairs in the batch share
pathology labels and treating them as additional positives.

LABEL ENCODING (from CXRLabelDataset)
--------------------------------------
  Tensor value  |  CSV value  |  Meaning
  --------------|-------------|----------------------------------
      1.0       |      1      |  pathology positively mentioned
     -1.0       |      0      |  pathology explicitly ruled out
      0.0       |     -1      |  uncertain language  } ignored:
     -1.0       |     NaN     |  not mentioned       } 

MATCHING MODES  (--match-mode)
-------------------------------
  single_label    Two samples are a positive pair if they share ≥1 label with
                  value 1.0.  Finds the most positives per batch.

  two_label       Same, but threshold is ≥2 shared positive labels.  Reduces
                  noise from coincidentally shared single labels.  Works best
                  with --caption-mode pair (captions list two labels).

  negative_aware  Builds on single_label (threshold=1) and adds an explicit
                  REPULSION term for conflict pairs.  A conflict means one
                  sample has label=1 and the other has label=-1 for the SAME
                  pathology (one report says "Edema present", the other says
                  "No edema").  Those pairs are pushed apart with a hinge loss.

LOSS VARIANTS  (--loss)
-----------------------
  clip    Multi-positive softmax cross-entropy (LabelAwareClipLoss).
          Each row of the similarity matrix is normalised via softmax; the
          target is a soft distribution that sums to 1 across all in-batch
          positives.  Equivalent to vanilla CLIP when the only positive is
          the diagonal.

  siglip  Sigmoid binary cross-entropy per pair (LabelAwareSigLipLoss).
          Every (image, text) pair is treated INDEPENDENTLY — positive pairs
          get target +1, negative / conflicting pairs get target −1.
          Loss = -sum over all pairs of logsigmoid(target * logit) / B.
          Key difference from clip: no softmax normalisation across the row,
          so the loss for each pair is decoupled from the rest of the batch.
          In negative_aware mode an extra hinge repulsion is added on top,
          but it is less critical than in the CLIP variant because SigLIP's
          −1 targets already provide per-pair repulsion for conflicting pairs.

KNOBS
-----
  --negative-weight  (λ)      Weight of the repulsion term relative to the main
                              contrastive loss.  Start at 0.5; lower if unstable.
  --negative-margin  (τ)      Pairs with cosine_sim < τ are NOT penalised — they
                              are already far apart.  Default 0.0 means any
                              positive similarity between conflicting pairs is
                              penalised.  Raise τ to be more aggressive, lower
                              (e.g. -0.2) to be more lenient.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelAwareClipLoss(nn.Module):
    """
    Multi-positive contrastive loss driven by in-batch label overlap.

    The loss has two components:

    1. ATTRACTION (all label-aware modes)
       Multi-positive cross-entropy: like vanilla CLIP's cross-entropy but the
       target is a soft matrix where EVERY in-batch pair that shares enough
       positive labels gets a non-zero target weight.  The weights in each row
       sum to 1.

    2. REPULSION (negative_aware mode only)
       Hinge loss that pushes CONFLICTING pairs (one says "Edema", other says
       "No edema") to have cosine similarity below --negative-margin.

    Args:
        match_mode:  'single_label', 'two_label', or 'negative_aware'.
        neg_weight:  λ — weight for the repulsion term (negative_aware only).
        neg_margin:  τ — cosine-sim threshold for repulsion hinge (negative_aware only).
    """

    def __init__(
        self,
        match_mode: str = "single_label",
        neg_weight: float = 0.5,
        neg_margin: float = 0.0,
    ):
        super().__init__()
        assert match_mode in ("single_label", "two_label", "negative_aware", "graded"), match_mode
        self.match_mode = match_mode
        self.neg_weight = neg_weight
        self.neg_margin = neg_margin
        # How many shared POSITIVE labels are required for a pair to count as positive.
        # two_label requires ≥2; everything else (single_label, negative_aware) requires ≥1.
        # graded does not use a threshold — it uses raw shared counts as soft weights.
        self._pos_threshold = 2 if match_mode == "two_label" else 1

    # ------------------------------------------------------------------
    # Core: build the soft target matrix and conflict mask
    # ------------------------------------------------------------------

    def _build_target(self, labels: torch.Tensor):
        """
        Build the (B, B) soft target matrix and conflict mask for one batch.

        STEP 1 — Extract positive and explicit-negative indicators
        -----------------------------------------------------------
        `pos[i, l]  = 1`  if sample i has label l == 1.0  (pathology confirmed)
        `neg[i, l]  = 1`  if sample i has label l == -1.0 (pathology ruled out)
        Uncertain (0.0) entries contribute nothing to either matrix.

        STEP 2 — Count shared positive labels between every pair (i, j)
        ----------------------------------------------------------------
        shared_pos[i, j]  = number of labels where BOTH sample i AND sample j
                            have value 1.0.

        This is computed as the matrix product  pos @ pos.T:
          (pos @ pos.T)[i, j]  =  Σ_l  pos[i, l] * pos[j, l]
        which is exactly the count of labels that are simultaneously positive
        for both samples.

        STEP 3 — Apply the matching threshold
        ----------------------------------------
        single_label  (threshold=1):  positive[i,j] = True  iff  shared_pos[i,j] >= 1
        two_label     (threshold=2):  positive[i,j] = True  iff  shared_pos[i,j] >= 2
        negative_aware              : same as single_label (threshold=1)

        The diagonal is always set to True: each sample is always a positive
        with itself (standard CLIP identity pairing is preserved).

        STEP 4 — Detect conflict pairs  (used by negative_aware repulsion only)
        -------------------------------------------------------------------------
        A pair (i, j) CONFLICTS if there exists any label l such that:
          sample i says "label l is present" (pos[i,l]=1) AND
          sample j says "label l is absent"  (neg[j,l]=1)
          — OR vice versa.

        Computed as:
          (pos @ neg.T)[i, j]  = Σ_l  pos[i,l] * neg[j,l]   — i=pos, j=neg
          (neg @ pos.T)[i, j]  = Σ_l  neg[i,l] * pos[j,l]   — i=neg, j=pos
          conflict[i,j] = True  iff  either sum > 0

        The diagonal is cleared (a sample never conflicts with itself).

        NaN labels are encoded as -1.0 (absent), so the conflict rate is
        typically ~78 % of off-diagonal pairs.  Conflict pairs are NOT
        zeroed out of the positive target — if two samples share positive
        labels they attract each other regardless of any conflicts on other
        labels.  The hinge term in negative_aware forward() provides the
        counter-signal; keep its weight λ small (≤ 0.1) so attraction
        dominates.

        STEP 5 — Build the soft target and normalise
        ---------------------------------------------
        target[i,j] = positive[i,j] / row_sum_i
          (conflicting pairs are NOT excluded — shared positive labels attract
           even when the pair also conflicts on a different label)

        Each row is divided by its sum (clamped to ≥1 so we never divide by
        zero for samples that have no in-batch positive partner).  After
        normalisation, each row sums to 1, just like a one-hot target in
        standard CLIP — but the mass may be spread across multiple positives.

        Args:
            labels: (B, L) float tensor with 1.0 / -1.0 / 0.0 encoding.
        Returns:
            target:        (B, B) float — soft positive target, rows sum to 1.
            conflict_mask: (B, B) bool  — True where pair is a known conflict.
        """
        # Step 1: binary masks for confirmed positives and explicit negatives
        pos = (labels == 1.0).float()   # (B, L)
        neg = (labels == -1.0).float()  # (B, L)

        # Step 2: count of shared positive labels between every pair (i, j)
        shared_pos = (pos @ pos.T).long()                    # (B, B)

        if self.match_mode == "graded":
            # Graded: soft target weight proportional to number of shared positive labels.
            # Pairs sharing 2 labels attract more strongly than pairs sharing 1.
            # Diagonal is clamped to ≥1 to keep the identity pairing for label-free samples.
            target = shared_pos.float()
            B = labels.shape[0]
            diag_idx = torch.arange(B, device=labels.device)
            target[diag_idx, diag_idx] = target[diag_idx, diag_idx].clamp(min=1.0)
            # Conflict: pairs with no shared positives but at least one contradiction
            conflict = ((pos @ neg.T) + (neg @ pos.T)) > 0
            conflict = conflict & (shared_pos == 0)
            conflict.fill_diagonal_(False)
            row_sums = target.sum(dim=1, keepdim=True).clamp(min=1.0)
            return target / row_sums, conflict

        # Step 3: threshold → boolean positive matrix; self is always positive
        positive = shared_pos >= self._pos_threshold         # (B, B) bool
        positive.fill_diagonal_(True)

        # Step 4: conflict detection — any label where one has 1 and other has -1,
        # but only for pairs that do NOT already share a positive label.
        # If two samples share a positive, attraction wins and we don't also repel.
        conflict = ((pos @ neg.T) + (neg @ pos.T)) > 0      # (B, B) bool
        conflict = conflict & ~positive
        conflict.fill_diagonal_(False)

        # Step 5: build target and normalise rows to sum=1.
        # Attraction is based purely on shared positive labels in all modes.
        # In negative_aware, conflicting pairs are repelled via the hinge term
        # in the forward pass — no need to zero them out here too.
        target = positive.float()
        row_sums = target.sum(dim=1, keepdim=True).clamp(min=1.0)
        return target / row_sums, conflict

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def batch_stats(self, labels: torch.Tensor) -> dict:
        """
        Diagnostic statistics for one batch — logged at the first step of every
        epoch in train_lora.py so you can sanity-check training signal quality.

        avg_positives_per_sample
            Mean number of in-batch positives per sample (including self).
            Should be > 1.0 for label-aware modes to provide benefit over
            standard CLIP.  If it hovers around 1.0, the batch is too small
            or labels are too sparse — increase --batch-size or drop to
            single_label mode.

        pct_diagonal_only
            Percentage of samples whose only in-batch positive is themselves.
            Keep below ~20 %; above that the label-aware loss degenerates to
            vanilla CLIP for most samples.

        pct_conflict_pairs
            Percentage of off-diagonal pairs flagged as conflicting.  Only
            meaningful in negative_aware mode; tells you how much repulsion
            signal is present.  Very low (<1 %) means conflicting labels are
            rare in the batch — the repulsion term won't do much.
        """
        with torch.no_grad():
            target, conflict = self._build_target(labels)
            B = labels.shape[0]
            n_pairs = B * (B - 1)

            pos_per_row = (target > 0).float().sum(dim=1)
            avg_pos = pos_per_row.mean().item()

            conflict_pairs = conflict.float().sum().item()
            pct_conflict = 100.0 * conflict_pairs / max(n_pairs, 1)

            diagonal_only = (pos_per_row == 1).float().mean().item() * 100.0

        return {
            "avg_positives_per_sample": avg_pos,
            "pct_conflict_pairs": pct_conflict,
            "pct_diagonal_only": diagonal_only,
        }

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the label-aware contrastive loss.

        PART 1 — LOGIT MATRIX
        ----------------------
        logits[i, j]  =  logit_scale * cosine_sim(image_i, text_j)

        Both image_features and text_features are already L2-normalised, so
        their dot product IS the cosine similarity.  logit_scale is the
        exponentiated learned temperature (same as in vanilla CLIP).

        The resulting (B, B) matrix is:
          - diagonal: similarity of the paired (image_i, text_i) — should be high
          - off-diagonal [i,j]: similarity of image_i to a DIFFERENT text — can be
            high if they share pathology labels (label-aware modes exploit this)

        PART 2 — MULTI-POSITIVE CROSS-ENTROPY  (attraction)
        ----------------------------------------------------
        Standard CLIP minimises cross-entropy with a one-hot target (diagonal).
        Here the target is the soft matrix from _build_target, where each row
        distributes probability mass across ALL in-batch positives.

        For each sample i, the loss is:
          L_i2t[i]  =  -Σ_j  target[i,j] * log_softmax(logits[i, :])[j]

        This says: "distribute attention across all positive texts for image i,
        not just the paired one."  If sample i has two in-batch positives (itself
        and sample j), each gets target weight 0.5 and both are pulled up equally.

        The loss is computed SYMMETRICALLY:
          L_i2t: image → text direction  (each row of logits is a distribution over texts)
          L_t2i: text → image direction  (transpose — each row is a distribution over images)
        Final loss = (L_i2t + L_t2i) / 2

        PART 3 — REPULSION HINGE  (negative_aware mode only)
        ------------------------------------------------------
        For every pair (i, j) flagged as conflicting in the conflict_mask, we
        add a hinge penalty:

          repulsion[i,j]  =  relu(cosine_sim[i,j]  -  neg_margin)

        This is zero when the pair is already separated below neg_margin (τ),
        and grows linearly with cosine similarity above τ.

        The mean over all conflicting pairs is weighted by neg_weight (λ):
          L_total  =  L_clip  +  λ * L_repulsion

        Note: the repulsion uses RAW cosine similarity (image_features @ text_features.T),
        not the scaled logits — the hinge acts in [−1, 1] space independently
        of the temperature.

        Args:
            image_features: (B, D) L2-normalised image embeddings.
            text_features:  (B, D) L2-normalised text embeddings.
            logit_scale:    scalar — exp of the learned log-temperature.
            labels:         (B, L) float with 1.0 / -1.0 / 0.0 encoding.
        Returns:
            Scalar loss tensor.
        """
        # Part 1: scaled cosine-similarity matrix
        logits = logit_scale * (image_features @ text_features.T)  # (B, B)

        # Build soft target and conflict mask from batch labels
        target, conflict = self._build_target(labels)

        # Part 2: symmetric multi-positive cross-entropy
        lp_i2t = F.log_softmax(logits, dim=1)    # log-prob over texts   for each image
        lp_t2i = F.log_softmax(logits.T, dim=1)  # log-prob over images  for each text
        clip_loss = (
            -(target * lp_i2t).sum(1).mean()   # image→text direction
            + -(target * lp_t2i).sum(1).mean() # text→image direction
        ) / 2

        self._last_clip_loss = clip_loss.detach()
        self._last_neg_loss = None

        # Part 3: repulsion hinge for conflicting pairs (negative_aware only)
        # Uses scaled logits (not raw cosine sim) so the hinge gradient is
        # ~logit_scale× larger and comparable to the CLIP loss gradient.
        # neg_margin is now in logit space; 0.0 still means "penalise any
        # positive logit between conflicting pairs".
        if self.match_mode == "negative_aware" and conflict.any():
            neg_loss = F.relu(logits[conflict] - self.neg_margin).mean()
            self._last_neg_loss = neg_loss.detach()
            return clip_loss + self.neg_weight * neg_loss

        return clip_loss


# ── Sigmoid variant (SigLIP loss) ─────────────────────────────────────────────

class ImagePairAwareLoss(nn.Module):
    """
    Three-component loss for label-supervised image retrieval training:

    1. IMAGE-TEXT (SigLIP diagonal): each (image_i, text_i) pair is pulled
       together; all off-diagonal image-text pairs are pushed apart.

    2. IMAGE-IMAGE ATTRACTION: image pairs that share the same label value
       (both 1.0 or both -1.0 for any label) are pulled closer in image-
       embedding space only.

    3. IMAGE-IMAGE REPULSION: image pairs with a conflicting label (one 1.0,
       one -1.0 for any label) are pushed apart in image-embedding space only.

    Missing labels (0.0 in the tensor) are ignored in all pairings.
    Use --nan-mode ignore when creating the dataset so that only CSV 0
    (explicitly ruled out) encodes as -1.0, and NaN (unmentioned) is 0.0.

    Args:
        attract_weight: λ_a — weight for the image-image attraction term.
        repel_weight:   λ_r — weight for the image-image repulsion term.
        repel_margin:   τ   — cosine-sim threshold: pairs already below τ are
                        not penalised by the repulsion hinge. Default 0.0.
    """

    def __init__(
        self,
        attract_weight: float = 0.1,
        repel_weight: float = 0.1,
        repel_margin: float = 0.0,
    ):
        super().__init__()
        self.attract_weight = attract_weight
        self.repel_weight = repel_weight
        self.repel_margin = repel_margin

    def _build_image_pair_masks(
        self, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build attract and conflict boolean masks from the label matrix.

        attract[i,j]: True if i and j share ≥1 label with the same non-zero
                      value (both 1.0 or both -1.0).
        conflict[i,j]: True if i and j have ≥1 label where one is 1.0 and
                       the other is -1.0, AND they do not attract on any label.
                       (attraction wins when labels also coincide.)

        Both masks are False on the diagonal.
        """
        pos = (labels == 1.0).float()   # (B, L)
        neg = (labels == -1.0).float()  # (B, L)

        attract = ((pos @ pos.T) > 0) | ((neg @ neg.T) > 0)
        attract.fill_diagonal_(False)

        conflict = ((pos @ neg.T) + (neg @ pos.T)) > 0
        conflict = conflict & ~attract
        conflict.fill_diagonal_(False)

        return attract, conflict

    def batch_stats(self, labels: torch.Tensor) -> dict:
        with torch.no_grad():
            attract, conflict = self._build_image_pair_masks(labels)
            B = labels.shape[0]
            n_pairs = B * (B - 1)
            return {
                "avg_positives_per_sample": 1.0,
                "pct_diagonal_only": 0.0,
                "pct_conflict_pairs": 100.0 * conflict.float().sum().item() / max(n_pairs, 1),
                "pct_attract_pairs": 100.0 * attract.float().sum().item() / max(n_pairs, 1),
            }

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        labels: torch.Tensor,
        logit_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = image_features.shape[0]

        # 1. Image-text SigLIP loss (diagonal ±1 target)
        logits = logit_scale * (image_features @ text_features.T)
        if logit_bias is not None:
            logits = logits + logit_bias
        it_target = 2.0 * torch.eye(B, device=image_features.device) - 1.0
        clip_loss = -F.logsigmoid(it_target * logits).sum() / B

        self._last_clip_loss = clip_loss.detach()
        self._last_attract_loss = None
        self._last_repel_loss = None

        attract, conflict = self._build_image_pair_masks(labels)
        img_sim = image_features @ image_features.T  # cosine sim, L2-normalised inputs

        total_loss = clip_loss

        # 2. Image-image attraction: MSE towards cosine-sim = 1.0
        if attract.any() and self.attract_weight > 0:
            attract_loss = (1.0 - img_sim[attract]).mean()
            self._last_attract_loss = attract_loss.detach()
            total_loss = total_loss + self.attract_weight * attract_loss

        # 3. Image-image repulsion: hinge on cosine-sim above repel_margin
        if conflict.any() and self.repel_weight > 0:
            repel_loss = F.relu(img_sim[conflict] - self.repel_margin).mean()
            self._last_repel_loss = repel_loss.detach()
            total_loss = total_loss + self.repel_weight * repel_loss

        return total_loss


# ── Sigmoid variant (SigLIP loss) ─────────────────────────────────────────────

class LabelAwareSigLipLoss(nn.Module):
    """
    Multi-positive sigmoid binary cross-entropy driven by in-batch label overlap.

    WHY SIGLIP INSTEAD OF CLIP LOSS?
    ---------------------------------
    Standard CLIP softmax cross-entropy normalises each row over the whole batch:
    the loss for sample i depends on how "hard" all the other negatives are.
    SigLIP (arxiv.org/abs/2303.15343) treats every (image_i, text_j) pair as an
    INDEPENDENT binary classification:
      - positive pair  → predict sigmoid(logit) ≈ 1
      - negative pair  → predict sigmoid(logit) ≈ 0

    No softmax denominator means the loss for each pair is decoupled from the
    rest of the batch, which can work better at smaller batch sizes.

    For the LABEL-AWARE extension, the same positive/conflict detection logic
    as LabelAwareClipLoss is used.  The only difference is the target encoding:
      - CLIP:    soft target ∈ [0, 1], rows sum to 1 (probability distribution)
      - SigLIP:  hard ±1 target; +1 for true positives, −1 for everything else

    REPULSION IN SIGLIP
    --------------------
    Conflicting pairs automatically receive target −1, so the SigLIP loss already
    pushes them apart without a separate hinge.  The optional hinge repulsion term
    (negative_aware mode) is kept for additional emphasis but is less critical here
    than in the CLIP variant.

    Args:
        match_mode:  'single_label', 'two_label', or 'negative_aware'.
        neg_weight:  λ — weight for the optional extra repulsion hinge.
        neg_margin:  τ — cosine-sim threshold for repulsion hinge.
    """

    def __init__(
        self,
        match_mode: str = "single_label",
        neg_weight: float = 0.5,
        neg_margin: float = 0.0,
    ):
        super().__init__()
        assert match_mode in ("single_label", "two_label", "negative_aware", "graded"), match_mode
        self.match_mode = match_mode
        self.neg_weight = neg_weight
        self.neg_margin = neg_margin
        self._pos_threshold = 2 if match_mode == "two_label" else 1

    def _build_siglip_target(self, labels: torch.Tensor):
        """
        Build the ±1 SigLIP target and conflict mask for one batch.

        Uses the same four-step mask logic as LabelAwareClipLoss._build_target
        (pos/neg indicators → shared_pos count → threshold → conflict detection),
        but instead of a soft normalised target it produces a hard ±1 matrix:

          +1  →  positive pair (shares ≥k positive labels, regardless of conflicts)
          −1  →  negative pair (insufficient label overlap)

        Conflicting pairs that also share positive labels receive +1 (attracted).
        This matches the CLIP variant: both losses rely on the hinge repulsion
        term in negative_aware forward() for the push signal rather than
        forcing conflicting pairs to 0/−1 in the target.  Because NaN is
        encoded as -1.0 (absent) the conflict rate is high (~78 %); keep
        λ small (≤ 0.1) so attraction dominates.

        Args:
            labels: (B, L) float tensor with 1.0 / -1.0 / 0.0 encoding.
        Returns:
            target:  (B, B) float — +1.0 for positives, -1.0 for negatives.
            conflict: (B, B) bool — True where pair is a known conflict.
        """
        pos = (labels == 1.0).float()
        neg = (labels == -1.0).float()

        shared_pos = (pos @ pos.T).long()

        if self.match_mode == "graded":
            # Graded: soft target in [0, 1] proportional to shared label count.
            # Row-normalised so the strongest match in each row gets the largest weight.
            # Uses soft BCE in forward() instead of the ±1 SigLIP formula.
            target = shared_pos.float()
            B = labels.shape[0]
            diag_idx = torch.arange(B, device=labels.device)
            target[diag_idx, diag_idx] = target[diag_idx, diag_idx].clamp(min=1.0)
            row_sums = target.sum(dim=1, keepdim=True).clamp(min=1.0)
            target = target / row_sums  # [0, 1]
            conflict = ((pos @ neg.T) + (neg @ pos.T)) > 0
            conflict = conflict & (shared_pos == 0)
            conflict.fill_diagonal_(False)
            return target, conflict

        positive = shared_pos >= self._pos_threshold
        positive.fill_diagonal_(True)

        conflict = ((pos @ neg.T) + (neg @ pos.T)) > 0
        conflict = conflict & ~positive
        conflict.fill_diagonal_(False)

        # +1 for shared-label pairs, −1 for everything else.
        # Attraction is based purely on shared positive labels in all modes.
        # Conflicting pairs are repelled via the hinge term in the forward pass.
        target = torch.where(
            positive,
            torch.ones_like(positive, dtype=torch.float),
            -torch.ones_like(positive, dtype=torch.float),
        )
        return target, conflict

    def batch_stats(self, labels: torch.Tensor) -> dict:
        """Same diagnostics as LabelAwareClipLoss.batch_stats — see that docstring."""
        with torch.no_grad():
            target, conflict = self._build_siglip_target(labels)
            B = labels.shape[0]
            n_pairs = B * (B - 1)

            pos_per_row = (target > 0).float().sum(dim=1)  # +1 cells are positives
            avg_pos = pos_per_row.mean().item()

            conflict_pairs = conflict.float().sum().item()
            pct_conflict = 100.0 * conflict_pairs / max(n_pairs, 1)

            diagonal_only = (pos_per_row == 1).float().mean().item() * 100.0

        return {
            "avg_positives_per_sample": avg_pos,
            "pct_conflict_pairs": pct_conflict,
            "pct_diagonal_only": diagonal_only,
        }

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        labels: torch.Tensor,
        logit_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the label-aware sigmoid contrastive loss.

        PART 1 — LOGIT MATRIX
        ----------------------
        logits[i, j]  =  logit_scale * cosine_sim(image_i, text_j)  [+ logit_bias]

        logit_bias is an optional learned scalar present in SigLIP-specific model
        checkpoints (e.g. ViT-B-16-SigLIP).  Vanilla CLIP models (ViT-B-32,
        ViT-S-32-alt) do not have it; train_lora.py detects its presence via
        getattr(model, 'logit_bias', None) and passes None when absent.

        PART 2 — SIGMOID BINARY CROSS-ENTROPY
        ----------------------------------------
        Every (i, j) pair is treated as an independent binary problem:

          loss[i,j]  =  -logsigmoid(target[i,j] * logits[i,j])

        For positive pairs  (target=+1):  minimised when logit >> 0 (high similarity)
        For negative pairs  (target=−1):  minimised when logit << 0 (low similarity)

        Total loss = sum over all B² pairs / B  (matches open_clip SigLipLoss normalisation).
        Conflicting pairs get target −1, so they are already repelled by this term.

        PART 3 — OPTIONAL EXTRA REPULSION HINGE  (negative_aware mode only)
        ---------------------------------------------------------------------
        Same hinge as LabelAwareClipLoss for consistency and stronger emphasis:

          repulsion[i,j]  =  relu(cosine_sim[i,j]  -  neg_margin)

          L_total  =  L_siglip  +  λ * mean(repulsion over conflict pairs)

        Less impactful than in the CLIP variant (SigLIP already repels via −1 target)
        but keeps negative_aware semantics consistent across both loss variants.

        Args:
            image_features: (B, D) L2-normalised image embeddings.
            text_features:  (B, D) L2-normalised text embeddings.
            logit_scale:    scalar — exp of the learned log-temperature.
            labels:         (B, L) float with 1.0 / -1.0 / 0.0 encoding.
            logit_bias:     optional scalar bias (from SigLIP model checkpoints).
        Returns:
            Scalar loss tensor.
        """
        # Part 1: scaled (and optionally biased) cosine-similarity matrix
        logits = logit_scale * (image_features @ text_features.T)  # (B, B)
        if logit_bias is not None:
            logits = logits + logit_bias

        # Build target and conflict mask
        target, conflict = self._build_siglip_target(labels)

        # Part 2: loss over all B² pairs, normalised by B.
        # Graded mode: target ∈ [0,1] → soft BCE: -(t·log σ(l) + (1-t)·log σ(-l))
        # Other modes: target ∈ {-1,+1} → standard SigLIP: -log σ(t·l)
        B = image_features.shape[0]
        if self.match_mode == "graded":
            siglip_loss = -(
                target * F.logsigmoid(logits) + (1 - target) * F.logsigmoid(-logits)
            ).sum() / B
        else:
            siglip_loss = -F.logsigmoid(target * logits).sum() / B

        self._last_clip_loss = siglip_loss.detach()
        self._last_neg_loss = None

        # Part 3: optional extra repulsion hinge (negative_aware only)
        # Uses scaled logits so gradient magnitude matches the siglip loss term.
        if self.match_mode == "negative_aware" and conflict.any():
            neg_loss = F.relu(logits[conflict] - self.neg_margin).mean()
            self._last_neg_loss = neg_loss.detach()
            return siglip_loss + self.neg_weight * neg_loss

        return siglip_loss
