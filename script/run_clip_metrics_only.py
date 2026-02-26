#!/usr/bin/env python3
"""
Run CLIP-based metrics on VcEdit (GaussianEditor-style) outputs.

This mirrors the behavior of the GaussianEditor `run_clip_metrics_only.py`
script: given a trial directory that contains a `save/` folder with renders
(`itXXXX-test`), it runs the external `clip_metrics_eval.py` script to compute
CLIP-based metrics between GT images and the renders, and writes `eval_clip.txt`
into the trial directory.
"""

import argparse
import os
from pathlib import Path
from typing import List

import re
import random

from tqdm import tqdm
from PIL import Image
import torch
import clip
from torchvision import transforms
import numpy as np


# === DEFAULT SETTINGS (can be overridden by CLI) ===
# These are just placeholders; override them from the command line.
DATA_TYPE = "3d-ovs"
DATA_NAME = "covered_desk"

# Example GT path used in the original GaussianEditor project.
# Override via --gt_dir when running this script.
GT_DIR = f"/data/users/jaeyeonpark/3dgs-trained/{DATA_TYPE}/{DATA_NAME}/colmap_render_full"

# Trial directory produced by VcEdit (the directory that contains `save/`)
# Example:
# OUTPUT_DIR = "/working/style-transfer/VcEdit/outputs/edit-inf/clown@20260226-144858"
OUTPUT_DIR = ""

# GPU index or "cpu"
GPU = "0"

# Style / edit description
STYLE_IMAGE = ""
STYLE_SOURCE_PROMPT = ""
STYLE_TARGET_PROMPT = ""

# Training step used for test renders (e.g. "800" -> it800-test)
MAX_STEPS = "800"
DEVICE = "cuda"

# Metrics options
INTERVAL = 1  # CLIP directional consistency (CLIPDirCons) frame interval k


# === CLIP helper components (ported from GaussianEditor) ===
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

cropper = transforms.Compose(
    [
        transforms.RandomCrop(128),
    ]
)

augment = transforms.Compose(
    [
        transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5),
        transforms.Resize(224),
    ]
)


CLIP_TEMPLATES: List[str] = [
    "a photo of {}",
    "a rendering of {}",
    "a cropped photo of {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a close-up photo of {}",
    "a bright photo of {}",
    "a dark photo of {}",
]


def compose_text_with_templates(text: str) -> List[str]:
    return [template.format(text) for template in CLIP_TEMPLATES]


def encode_image(
    model,
    preprocess,
    image_path,
    patch: bool = False,
    device: str = "cuda",
):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        img_proc = []
        if patch:
            for _ in range(128):
                target_crop = cropper(image)
                target_crop = augment(target_crop)
                img_proc.append(target_crop)
        else:
            img_proc.append(image)
        img_proc = torch.cat(img_proc, dim=0)
        image_feat = model.encode_image(img_proc)
        image_feat /= image_feat.clone().norm(dim=-1, keepdim=True)
    return image_feat


def encode_text(model, text: str, device: str):
    composed_text = compose_text_with_templates(text)
    tokens = clip.tokenize(composed_text).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(tokens)
        text_feat = text_feat.mean(axis=0, keepdim=True)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
    return text_feat


def get_direction(emb1, emb2):
    return emb1 - emb2


def convert_render_filename_to_gt(render_filename: str, gt_dir: str) -> str:
    match = re.search(r"(\d+)", render_filename)
    if match:
        num = int(match.group(1))
        # In this project, GT frames are saved as "<index>.png" (no zero padding),
        # e.g. "0.png", "1.png", ..., so we map directly.
        gt_filename = f"{num}.png"
        gt_path = os.path.join(gt_dir, gt_filename)
        if os.path.exists(gt_path):
            return gt_filename
    return render_filename


class CLIPDirSim:
    def __init__(
        self,
        model,
        preprocess,
        style_target_prompt: str = None,
        style_image: str = None,
        style_source_prompt: str = "a Photo",
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.preprocess = preprocess
        self.device = device
        with torch.no_grad():
            if style_target_prompt is not None:
                self.style_feat = encode_text(model, style_target_prompt, device)
            if style_image is not None:
                self.style_feat = encode_image(
                    model, preprocess, style_image, device=device
                )
            src_feat = encode_text(model, style_source_prompt, device=device)
            self.style_dir = get_direction(self.style_feat, src_feat)

    def __call__(self, gt_image_path: str, render_image_path: str, patch: bool = False):
        files = [
            f
            for f in os.listdir(render_image_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        files.sort(
            key=lambda f: int("".join(filter(str.isdigit, f)))
            if "".join(filter(str.isdigit, f))
            else 0
        )
        scores_sum = 0.0
        images = 0
        with torch.no_grad():
            for filename in tqdm(
                files, desc="CLIP directional similarity", unit="img"
            ):
                gt_filename = convert_render_filename_to_gt(filename, gt_image_path)
                gt_path = os.path.join(gt_image_path, gt_filename)
                render_path = os.path.join(render_image_path, filename)
                gt_feat = encode_image(
                    self.model, self.preprocess, gt_path, device=self.device
                )
                render_feat = encode_image(
                    self.model,
                    self.preprocess,
                    render_path,
                    patch=patch,
                    device=self.device,
                )
                img_dir = get_direction(render_feat, gt_feat)
                style_dir = self.style_dir
                scores_sum += (
                    torch.cosine_similarity(img_dir, style_dir, dim=1)
                    .cpu()
                    .numpy()[0]
                )
                images += 1
        return 100 * scores_sum / max(images, 1)


class CLIPDirCons:
    def __init__(self, model, preprocess, device: str = "cuda"):
        self.model = model.to(device)
        self.preprocess = preprocess
        self.device = device

    def __call__(
        self,
        gt_image_path: str,
        render_image_path: str,
        k: int = 1,
        patch: bool = False,
    ):
        files = [
            f
            for f in os.listdir(render_image_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        files.sort(
            key=lambda f: int("".join(filter(str.isdigit, f)))
            if "".join(filter(str.isdigit, f))
            else 0
        )
        scores_sum = 0.0
        images = 0
        with torch.no_grad():
            for i, _ in tqdm(
                enumerate(files),
                total=len(files),
                desc="CLIP directional consistency",
                unit="img",
            ):
                if i < len(files) - k:
                    gt_filename = convert_render_filename_to_gt(files[i], gt_image_path)
                    gt_filename_next = convert_render_filename_to_gt(
                        files[i + k], gt_image_path
                    )
                    gt_path = os.path.join(gt_image_path, gt_filename)
                    gt_path_next = os.path.join(gt_image_path, gt_filename_next)
                    render_path = os.path.join(render_image_path, files[i])
                    render_path_next = os.path.join(
                        render_image_path, files[i + k]
                    )
                    render_feat = encode_image(
                        self.model,
                        self.preprocess,
                        render_path,
                        patch,
                        device=self.device,
                    )
                    render_feat_next = encode_image(
                        self.model,
                        self.preprocess,
                        render_path_next,
                        patch,
                        device=self.device,
                    )
                    gt_feat = encode_image(
                        self.model,
                        self.preprocess,
                        gt_path,
                        device=self.device,
                    )
                    gt_feat_next = encode_image(
                        self.model,
                        self.preprocess,
                        gt_path_next,
                        device=self.device,
                    )
                    gt_dir = get_direction(gt_feat_next, gt_feat)
                    render_dir = get_direction(render_feat_next, render_feat)
                    scores_sum += (
                        torch.cosine_similarity(gt_dir, render_dir, dim=1)
                        .cpu()
                        .numpy()[0]
                    )
                    images += 1
        return 100 * scores_sum / max(images, 1)


class CLIPScore:
    def __init__(
        self,
        model,
        preprocess,
        style_target_prompt: str = None,
        style_image: str = None,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.preprocess = preprocess
        self.device = device
        with torch.no_grad():
            if style_target_prompt is not None:
                self.style_feat = encode_text(
                    model, style_target_prompt, device
                )
            if style_image is not None:
                self.style_feat = encode_image(
                    model, preprocess, style_image, device=device
                )

    def __call__(self, render_image_path: str, patch: bool = False):
        files = [
            f
            for f in os.listdir(render_image_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        files.sort(
            key=lambda f: int("".join(filter(str.isdigit, f)))
            if "".join(filter(str.isdigit, f))
            else 0
        )
        scores_sum = 0.0
        images = 0
        with torch.no_grad():
            for filename in tqdm(files, desc="CLIP Score", unit="img"):
                render_path = os.path.join(render_image_path, filename)
                render_feat = encode_image(
                    self.model,
                    self.preprocess,
                    render_path,
                    patch,
                    device=self.device,
                )
                scores_sum += (
                    torch.cosine_similarity(render_feat, self.style_feat, dim=1)
                    .cpu()
                    .numpy()[0]
                )
                images += 1
        return 100 * scores_sum / max(images, 1)


class CLIPF:
    def __init__(self, model, preprocess, device: str = "cuda"):
        self.model = model.to(device)
        self.preprocess = preprocess
        self.device = device

    def __call__(
        self,
        gt_image_path: str,
        render_image_path: str,
        patch: bool = False,
    ):
        files = [
            f
            for f in os.listdir(render_image_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        files.sort(
            key=lambda f: int("".join(filter(str.isdigit, f)))
            if "".join(filter(str.isdigit, f))
            else 0
        )
        scores_sum_render = 0.0
        scores_sum_gt = 0.0
        images = 0
        with torch.no_grad():
            for i, _ in tqdm(
                enumerate(files), total=len(files), desc="CLIP F", unit="img"
            ):
                if i < len(files) - 1:
                    gt_filename = convert_render_filename_to_gt(
                        files[i], gt_image_path
                    )
                    gt_filename_next = convert_render_filename_to_gt(
                        files[i + 1], gt_image_path
                    )
                    gt_path = os.path.join(gt_image_path, gt_filename)
                    gt_path_next = os.path.join(gt_image_path, gt_filename_next)
                    render_path = os.path.join(render_image_path, files[i])
                    render_path_next = os.path.join(
                        render_image_path, files[i + 1]
                    )
                    render_feat = encode_image(
                        self.model,
                        self.preprocess,
                        render_path,
                        patch,
                        device=self.device,
                    )
                    render_feat_next = encode_image(
                        self.model,
                        self.preprocess,
                        render_path_next,
                        patch,
                        device=self.device,
                    )
                    gt_feat = encode_image(
                        self.model,
                        self.preprocess,
                        gt_path,
                        device=self.device,
                    )
                    gt_feat_next = encode_image(
                        self.model,
                        self.preprocess,
                        gt_path_next,
                        device=self.device,
                    )
                    scores_sum_render += (
                        torch.cosine_similarity(
                            render_feat, render_feat_next, dim=1
                        )
                        .cpu()
                        .numpy()[0]
                    )
                    scores_sum_gt += (
                        torch.cosine_similarity(gt_feat, gt_feat_next, dim=1)
                        .cpu()
                        .numpy()[0]
                    )
                    images += 1
        clip_f_gt = scores_sum_gt / max(images, 1)
        clip_f_render = scores_sum_render / max(images, 1)
        return 100 * clip_f_render / max(clip_f_gt, 1e-6)

def get_script_dir() -> Path:
    return Path(__file__).parent.absolute()


def get_root_dir() -> Path:
    # This script lives in <root>/script/, so parent is project root.
    return get_script_dir().parent


def main() -> int:
    global OUTPUT_DIR, GT_DIR, GPU, DEVICE, STYLE_IMAGE, STYLE_SOURCE_PROMPT, STYLE_TARGET_PROMPT, MAX_STEPS, INTERVAL

    parser = argparse.ArgumentParser(description="Run CLIP metrics for a VcEdit trial")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Trial directory that contains `save/`")
    parser.add_argument("--gt_dir", type=str, default=GT_DIR, help="GT image directory")
    parser.add_argument("--gpu", type=str, default=GPU, help="GPU index (e.g. '0') or 'cpu'")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device string, e.g. 'cuda' or 'cpu'")
    parser.add_argument("--style_image", type=str, default=STYLE_IMAGE, help="Optional path to style image")
    parser.add_argument("--style_source_prompt", type=str, default=STYLE_SOURCE_PROMPT, help="Source (before edit) text prompt")
    parser.add_argument("--style_target_prompt", type=str, default=STYLE_TARGET_PROMPT, help="Target (after edit) text prompt")
    parser.add_argument("--max_steps", type=str, default=MAX_STEPS, help="Training max_steps used (for it{N}-test)")
    parser.add_argument("--interval", type=int, default=INTERVAL, help="Frame interval k for directional consistency")
    args = parser.parse_args()

    # Override globals with CLI values for this run
    OUTPUT_DIR = args.output_dir
    GT_DIR = args.gt_dir
    GPU = args.gpu
    DEVICE = args.device
    STYLE_IMAGE = args.style_image
    STYLE_SOURCE_PROMPT = args.style_source_prompt
    STYLE_TARGET_PROMPT = args.style_target_prompt
    MAX_STEPS = args.max_steps
    INTERVAL = args.interval

    if not OUTPUT_DIR:
        print("Error: OUTPUT_DIR is not set (pass --output_dir).")
        return 1
    if not GT_DIR:
        print("Error: GT_DIR is not set (pass --gt_dir).")
        return 1

    root_dir = get_root_dir()
    os.chdir(root_dir)

    save_dir = Path(OUTPUT_DIR.rstrip("/")) / "save"
    render_dir = save_dir / f"it{MAX_STEPS}-test"
    if not render_dir.exists():
        test_dirs = list(save_dir.glob("it*-test"))
        render_dir = test_dirs[0] if test_dirs else render_dir
    render_dir = render_dir.resolve()
    if not render_dir.exists():
        print(f"Error: render dir not found: {render_dir} (OUTPUT_DIR/save/it{{N}}-test)")
        return 1
    if not list(render_dir.glob("*.png")):
        print(f"Warning: No .png files in {render_dir}")

    if not Path(GT_DIR).exists():
        print(f"Error: GT_DIR not found: {GT_DIR}")
        return 1

    # Resolve device string (e.g. "cuda:0" or "cpu")
    device = f"cuda:{GPU}" if DEVICE == "cuda" and GPU != "cpu" else DEVICE

    print("=" * 50)
    print("Run CLIP metrics only (VcEdit)")
    print("=" * 50)
    print(f"Render: {render_dir}")
    print(f"GT:     {GT_DIR}")
    print("=" * 50)

    # Load CLIP model
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)

    clip_similarity = CLIPDirSim(
        clip_model,
        clip_preprocess,
        style_target_prompt=STYLE_TARGET_PROMPT if STYLE_TARGET_PROMPT else None,
        style_image=STYLE_IMAGE if STYLE_IMAGE else None,
        style_source_prompt=STYLE_SOURCE_PROMPT if STYLE_SOURCE_PROMPT else "a Photo",
        device=device,
    )
    clip_consistency = CLIPDirCons(
        clip_model, clip_preprocess, device=device
    )
    clip_f = CLIPF(clip_model, clip_preprocess, device=device)
    clip_score_metric = CLIPScore(
        clip_model,
        clip_preprocess,
        style_target_prompt=STYLE_TARGET_PROMPT if STYLE_TARGET_PROMPT else None,
        style_image=STYLE_IMAGE if STYLE_IMAGE else None,
        device=device,
    )

    clip_dir_consistency = clip_consistency(
        GT_DIR, str(render_dir), k=INTERVAL
    )
    clip_f_scaled = clip_f(GT_DIR, str(render_dir))
    clip_score_value = clip_score_metric(str(render_dir))
    clip_dir_similarity = clip_similarity(GT_DIR, str(render_dir))

    print(f"CLIP directional consistency: {clip_dir_consistency}")
    print(f"CLIP_F (scaled): {clip_f_scaled}")
    print(f"CLIP Score: {clip_score_value}")
    print(f"CLIP directional similarity: {clip_dir_similarity}")

    render_path = os.path.abspath(str(render_dir))
    trial_dir = Path(OUTPUT_DIR.rstrip("/")).resolve()
    eval_clip_path = trial_dir / "eval_clip.txt"

    with open(eval_clip_path, "w") as f:
        f.write("CLIP Evaluation Metrics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"GT Directory: {GT_DIR}\n")
        f.write(f"Render Directory: {render_dir}\n")
        if STYLE_TARGET_PROMPT:
            f.write(f"Style Target Prompt: {STYLE_TARGET_PROMPT}\n")
        if STYLE_IMAGE:
            f.write(f"Style Image: {STYLE_IMAGE}\n")
        f.write(f"Interval: {INTERVAL}\n")
        f.write("\n")
        f.write("-" * 50 + "\n")
        f.write("Results:\n")
        f.write("-" * 50 + "\n")
        f.write(f"CLIP directional consistency: {clip_dir_consistency}\n")
        f.write(f"CLIP_F (scaled): {clip_f_scaled}\n")
        f.write(f"CLIP Score: {clip_score_value}\n")
        f.write(f"CLIP directional similarity: {clip_dir_similarity}\n")

    print(f"\nMetrics saved to: {eval_clip_path}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

