import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ClipMetrics(nn.Module):
    """
    Lightweight CLIP wrapper used during training.

    forward(image_0, image_1, text_0, text_1) returns:
      - sim_0: image_0 vs text_0 cosine similarity
      - sim_1: image_1 vs text_1 cosine similarity
      - sim_direction: directional similarity between (image_1 - image_0) and (text_1 - text_0)
      - sim_image: image_0 vs image_1 cosine similarity
    """

    def __init__(self, name: str = "ViT-L/14"):
        super().__init__()
        assert name in (
            "RN50",
            "RN101",
            "RN50x4",
            "RN50x16",
            "RN50x64",
            "ViT-B/32",
            "ViT-B/16",
            "ViT-L/14",
            "ViT-L/14@336px",
        )  # fmt: skip
        self.size = {
            "RN50x4": 288,
            "RN50x16": 384,
            "RN50x64": 448,
            "ViT-L/14@336px": 336,
        }.get(name, 224)

        self.model, _ = clip.load(name, device="cpu", download_root="./")
        self.model.eval().requires_grad_(False)

        self.register_buffer("mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)))
        self.register_buffer("std", torch.tensor((0.26862954, 0.26130258, 0.27577711)))

    def encode_text(self, text: str):
        text = clip.tokenize(text, truncate=True).to(next(self.parameters()).device)
        text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode images in range [0, 1].
        image: (B, 3, H, W)
        """
        image = F.interpolate(
            image.float(), size=self.size, mode="bicubic", align_corners=False
        )
        image = image - rearrange(self.mean, "c -> 1 c 1 1")
        image = image / rearrange(self.std, "c -> 1 c 1 1")
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def forward(
        self,
        image_0: torch.Tensor,
        image_1: torch.Tensor,
        text_0: str,
        text_1: str,
    ):
        image_features_0 = self.encode_image(image_0)
        image_features_1 = self.encode_image(image_1)
        text_features_0 = self.encode_text(text_0)
        text_features_1 = self.encode_text(text_1)

        # CLIP score terms: cosine similarities between image and text
        sim_0 = F.cosine_similarity(image_features_0, text_features_0)
        sim_1 = F.cosine_similarity(image_features_1, text_features_1)

        # Directional similarity between image and text edits
        # (image_1 - image_0) vs (text_1 - text_0), as in CLIPDirSim
        sim_direction = F.cosine_similarity(
            image_features_1 - image_features_0,
            text_features_1 - text_features_0,
        )

        # Image-image similarity (used as a simple consistency term)
        sim_image = F.cosine_similarity(image_features_0, image_features_1)

        return sim_0, sim_1, sim_direction, sim_image

    def clip_score(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CLIP score between image and text embeddings.

        This follows the same cosine-similarity formulation used above, but
        operates directly on pre-computed features:

            score = cos( image_feat, text_feat )

        Args:
            image_features: (B, D) image embeddings.
            text_features:  (B, D) text embeddings.
        """
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return F.cosine_similarity(image_features, text_features)

    def clip_directional_consistency(
        self,
        gt_source_features: torch.Tensor,
        gt_target_features: torch.Tensor,
        render_source_features: torch.Tensor,
        render_target_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CLIP directional consistency between ground-truth and rendered
        edits in **source → target** space, following:

            gt_dir     = gt_target     - gt_source
            render_dir = render_target - render_source
            consistency = cos( gt_dir, render_dir )

        This matches the directional-consistency idea used in CLIPDirCons,
        but operates on already-computed embeddings instead of file paths.

        Args:
            gt_source_features:     (B, D) CLIP embedding of GT source images.
            gt_target_features:     (B, D) CLIP embedding of GT target images.
            render_source_features: (B, D) CLIP embedding of rendered source images.
            render_target_features: (B, D) CLIP embedding of rendered target images.
        """
        gt_dir = gt_target_features - gt_source_features
        render_dir = render_target_features - render_source_features

        gt_dir = gt_dir / gt_dir.norm(dim=1, keepdim=True)
        render_dir = render_dir / render_dir.norm(dim=1, keepdim=True)

        return F.cosine_similarity(gt_dir, render_dir)

