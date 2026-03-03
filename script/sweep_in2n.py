#!/usr/bin/env python3
"""
WandB Sweep for VcEdit edit-inf experiments on 3D-OVS-style datasets.

- Sweep target metric: CLIP directional similarity (maximize)
- Search space: only the "task" (prompt / seg_prompt combo), no hyperparameter tuning.

Usage:
  # 1) Create sweep (run once)
  python script/sweep_in2n.py --create

  # 2) Run agent(s) – each agent picks a task and runs train+metrics
  python script/sweep_in2n.py --agent --sweep_id <SWEEP_ID>

  # Use specific GPU(s): single GPU or comma-separated for multi-GPU per run
  python script/sweep_in2n.py --agent --sweep_id <ID> --gpus 0,1,2
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import wandb


# ==========================
# Fixed configuration
# ==========================

CONFIG = "configs/edit-inf.yaml"
GPU_DEFAULT = "0"  # default; override with --gpu or SWEEP_GPU

DATA_TYPE = "in2n-GSEditor"
DATA_SOURCE_ROOT = "/data/users/jaeyeonpark/dataset"
GS_SOURCE_ROOT = "/data/users/jaeyeonpark/3dgs-trained"
RENDER_SUBDIR = "colmap_render_full"

MAX_STEPS = "800"
PER_EDITING_STEP = "400"

WANDB_PROJECT = "vcedit-edit-inf"
WANDB_SWEEP_NAME = f"edit-inf/{DATA_TYPE}"

# CLIP metrics options
INTERVAL = 1
DEVICE = "cuda"


# ==========================
# Task definitions (prompt / seg_prompt / style prompts)
# Reuse the 3d-ovs TASKS from GaussianEditor edit-n2n.
# ==========================

TASKS = [
 # 1) Change the man's fleece color to dark green
    {
        "name": "fleece_dark_green",
        "DATA_NAME": "face",
        "PROMPT": "Change the color of his fleece jacket to dark green",
        "SEG_PROMPT": "grey fleece jacket",
        "TARGET_PROMPT": "dark green fleece jacket",
        "STYLE_SOURCE_PROMPT": "man wearing jacket with grey speckled fleece fabric",
        "STYLE_TARGET_PROMPT": "man wearing jacket with dark green textured fleece fabric",
    },
    # 2) Change the man's hair color to dark brown
    {
        "name": "hair_dark_brown",
        "DATA_NAME": "face",
        "PROMPT": "Change his hair color to dark brown",
        "SEG_PROMPT": "the man's head",
        "TARGET_PROMPT": "the man's dark brown hair",
        "STYLE_SOURCE_PROMPT": "man with natural light brown hair",
        "STYLE_TARGET_PROMPT": "man with dark brown hair",
    },
    # 3) Make the man's mouth smile
    {
        "name": "mouth_smile",
        "DATA_NAME": "face",
        "PROMPT": "Make his mouth smile",
        "SEG_PROMPT": "the man's mouth",
        "TARGET_PROMPT": "the man's mouth in a smiling pose",
        "STYLE_SOURCE_PROMPT": "man with neutral, closed mouth",
        "STYLE_TARGET_PROMPT": "man with open, smiling mouth",
    },
    # 4) Make him wear sunglasses
    {
        "name": "wear_sunglasses",
        "DATA_NAME": "face",
        "PROMPT": "Make him wear sunglasses",
        "SEG_PROMPT": "the man's eyes and nose bridge",
        "TARGET_PROMPT": "the man's eyes covered by dark sunglasses",
        "STYLE_SOURCE_PROMPT": "man with eyes, open and visible",
        "STYLE_TARGET_PROMPT": "man with dark sunglasses over the man's visible eyes",
    },
    # 5) Make his ear like an elf's ear
    {
        "name": "ear_elf",
        "DATA_NAME": "face",
        "PROMPT": "Make his ear like an elf's ear",
        "SEG_PROMPT": "man's visible right ear",
        "TARGET_PROMPT": "man's pointed, elf-like right ear",
        "STYLE_SOURCE_PROMPT": "man's rounded, normal human ear",
        "STYLE_TARGET_PROMPT": "man with pointed, elf-like human ear",
    },
    # 6) Change the Patagonia logo to a simple tree graphic
    {
        "name": "logo_change_tree",
        "DATA_NAME": "face",
        "PROMPT": "Change the Patagonia patch to a simple tree graphic patch",
        "SEG_PROMPT": "black Patagonia logo patch on the fleece pocket",
        "TARGET_PROMPT": "a patch with a simple tree graphic on the fleece pocket",
        "STYLE_SOURCE_PROMPT": "man wearing jacket with a patch with the text 'PATAGONIA' and a mountain range",
        "STYLE_TARGET_PROMPT": "man wearing jacket with a patch with a simple, stylized tree graphic and no text",
    },
    # 7) Add a small silver nose stud piercing
    {
        "name": "nose_stud_silver",
        "DATA_NAME": "face",
        "PROMPT": "Add a small silver stud piercing to his nose",
        "SEG_PROMPT": "right side of the man's nostril",
        "TARGET_PROMPT": "a nostril with a small silver stud piercing",
        "STYLE_SOURCE_PROMPT": "man with plain skin of the man's nose",
        "STYLE_TARGET_PROMPT": "man with skin with a small, glinting silver stud piercing",
    },
    # 8) Change the zipper pull to a bright red color
    {
        "name": "zipper_pull_red",
        "DATA_NAME": "face",
        "PROMPT": "Change the zipper pull to a bright red color",
        "SEG_PROMPT": "metallic zipper pull of the main zipper",
        "TARGET_PROMPT": "bright red colored zipper pull",
        "STYLE_SOURCE_PROMPT": "man wearing jacket with dull, metallic zipper pull",
        "STYLE_TARGET_PROMPT": "man wearing jacket with vibrant, bright red zipper pull",
    },
    # 9) Change the material of the jacket to a denim jacket
    {
        "name": "jacket_denim",
        "DATA_NAME": "face",
        "PROMPT": "Change the jacket material to blue denim",
        "SEG_PROMPT": "entire fleece jacket",
        "TARGET_PROMPT": "a blue denim jacket",
        "STYLE_SOURCE_PROMPT": "man wearing jacket with textured grey speckled fleece fabric",
        "STYLE_TARGET_PROMPT": "man wearing jacket with classic blue denim twill fabric",
    },
    # 10) Add a graphic of a compass to the sleeve
    {
        "name": "sleeve_compass",
        "DATA_NAME": "face",
        "PROMPT": "Add a graphic of a compass to the left sleeve of his fleece",
        "SEG_PROMPT": "fleece fabric of the left sleeve",
        "TARGET_PROMPT": "fleece sleeve with a small black compass graphic added",
        "STYLE_SOURCE_PROMPT": "man wearing jacket with plain grey speckled fleece fabric",
        "STYLE_TARGET_PROMPT": "man wearing jacket with plain grey fleece fabric with a detailed compass graphic",
    },
    # 11) Change his hair style to a short, cropped style
    {
        "name": "hair_style_short",
        "DATA_NAME": "face",
        "PROMPT": "Change his hair style to a short, cropped look",
        "SEG_PROMPT": "the man's entire hair",
        "TARGET_PROMPT": "the man with a short, cropped haircut",
        "STYLE_SOURCE_PROMPT": "man with natural curly, wavy light brown hair",
        "STYLE_TARGET_PROMPT": "man with short, closely cropped light brown hair",
    },
    # 12) Change the color of his eyes to blue
    {
        "name": "eyes_blue",
        "DATA_NAME": "face",
        "PROMPT": "Change his eye color to blue",
        "SEG_PROMPT": "irises of the man's visible eyes",
        "TARGET_PROMPT": "eyes with blue irises",
        "STYLE_SOURCE_PROMPT": "man with brown eye irises",
        "STYLE_TARGET_PROMPT": "man with blue eye irises",
    },
    # 13) Add text to the chest pocket saying 'STAFF'
    {
        "name": "chest_text_staff",
        "DATA_NAME": "face",
        "PROMPT": "Add text that says 'STAFF' to the fleece chest pocket patch, below the logo",
        "SEG_PROMPT": "bottom area of the fleece chest pocket patch",
        "TARGET_PROMPT": "chest pocket patch with text 'STAFF' in block letters added",
        "STYLE_SOURCE_PROMPT": "man wearing jacket with plain patch surface",
        "STYLE_TARGET_PROMPT": "man wearing jacket with patch surface with detailed block text 'STAFF'",
    },
    # 14) Add a realistic-looking tattoo of an anchor to his neck
    {
        "name": "neck_tattoo_anchor",
        "DATA_NAME": "face",
        "PROMPT": "Add a realistic-looking anchor tattoo to his neck",
        "SEG_PROMPT": "man's neck skin",
        "TARGET_PROMPT": "a neck with a detailed, small black anchor tattoo",
        "STYLE_SOURCE_PROMPT": "man with plain skin of the man's neck",
        "STYLE_TARGET_PROMPT": "man with skin with a detailed, black anchor tattoo that looks realistic",
    },
    # 15) Replace the white wall on the right with a large, city-view window
    {
        "name": "background_window_city",
        "DATA_NAME": "face",
        "PROMPT": "Replace the white wall on the right with a large window looking out at a city",
        "SEG_PROMPT": "white wall surface on the far right",
        "TARGET_PROMPT": "a large window with a detailed city skyline view",
        "STYLE_SOURCE_PROMPT": "man standing in front of a flat, plain white painted surface",
        "STYLE_TARGET_PROMPT": "man standing in front of a detailed window view with city buildings and sky",
    },

    ## bear
    # 1) Change the yellow face markings to red
    {
        "name": "markings_red",
        "DATA_NAME": "bear",
        "PROMPT": "Change the yellow face markings to red",
        "SEG_PROMPT": "markings on the bear's face",
        "TARGET_PROMPT": "red markings on the bear's face",
        "STYLE_SOURCE_PROMPT": "bear with yellow markings",
        "STYLE_TARGET_PROMPT": "bear with red markings",
    },
    # 2) Add a small plaid scarf around the neck area
    {
        "name": "neck_scarf",
        "DATA_NAME": "bear",
        "PROMPT": "Add a small plaid scarf around the bear's neck",
        "SEG_PROMPT": "neck area of the bear statue",
        "TARGET_PROMPT": "bear statue wearing a plaid scarf",
        "STYLE_SOURCE_PROMPT": "bear statue with plain stone neck",
        "STYLE_TARGET_PROMPT": "bear statue with stone neck with a woven plaid scarf",
    },
    # 3) Put a pair of sunglasses on the bear's head
    {
        "name": "wear_sunglasses",
        "DATA_NAME": "bear",
        "PROMPT": "Put a pair of sunglasses on the bear's head",
        "SEG_PROMPT": "head of the bear statue",
        "TARGET_PROMPT": "bear statue wearing sunglasses",
        "STYLE_SOURCE_PROMPT": "bare head of the bear statue",
        "STYLE_TARGET_PROMPT": "bear statue with head with a pair of dark sunglasses",
    },
    # 4) Change the material of the entire statue to bronze
    {
        "name": "statue_bronze",
        "DATA_NAME": "bear",
        "PROMPT": "Change the material of the bear statue to bronze",
        "SEG_PROMPT": "entire bear statue",
        "TARGET_PROMPT": "bronze bear statue",
        "STYLE_SOURCE_PROMPT": "bear statue with grey stone texture",
        "STYLE_TARGET_PROMPT": "bear statue with aged bronze metal texture",
    },
    # 5) Change the material of the entire statue to clear ice
    {
        "name": "statue_ice",
        "DATA_NAME": "bear",
        "PROMPT": "Change the material of the bear statue to clear ice",
        "SEG_PROMPT": "entire bear statue",
        "TARGET_PROMPT": "clear ice bear statue",
        "STYLE_SOURCE_PROMPT": "bear statue with grey stone texture",
        "STYLE_TARGET_PROMPT": "bear statue with transparent ice texture with reflections",
    },
    # 6) Make the facial expression of the head look much angrier
    {
        "name": "expression_angry",
        "DATA_NAME": "bear",
        "PROMPT": "Make the facial expression of the bear much angrier",
        "SEG_PROMPT": "head of the bear statue",
        "TARGET_PROMPT": "angry-faced bear statue",
        "STYLE_SOURCE_PROMPT": "bear statue with neutral, open-mouthed expression",
        "STYLE_TARGET_PROMPT": "bear statue with furrowed brow and snarling teeth",
    },
    # 7) Replace the rock pedestal under the statue with a pile of gold bars
    {
        "name": "pedestal_gold_bars",
        "DATA_NAME": "bear",
        "PROMPT": "Replace the rock pedestal with a pile of gold bars",
        "SEG_PROMPT": "rock pedestal under the bear statue",
        "TARGET_PROMPT": "pile of gold bars under the bear statue",
        "STYLE_SOURCE_PROMPT": "bear statue with grey rock texture",
        "STYLE_TARGET_PROMPT": "bear statue with stack of shiny gold bars",
    },
    # 8) Cover the entire statue with a layer of fuzzy green moss
    {
        "name": "statue_mossy",
        "DATA_NAME": "bear",
        "PROMPT": "Cover the bear statue with a layer of fuzzy green moss",
        "SEG_PROMPT": "entire bear statue",
        "TARGET_PROMPT": "bear statue covered in green moss",
        "STYLE_SOURCE_PROMPT": "bear statue with clean stone texture",
        "STYLE_TARGET_PROMPT": "bear statue with mossy green fuzzy texture",
    },
    # 9) Add a small, fabric backpack to the back area of the bear
    {
        "name": "back_backpack",
        "DATA_NAME": "bear",
        "PROMPT": "Add a small backpack to the bear's back",
        "SEG_PROMPT": "back area of the bear statue",
        "TARGET_PROMPT": "bear statue with a small backpack",
        "STYLE_SOURCE_PROMPT": "bear statue with plain stone back",
        "STYLE_TARGET_PROMPT": "bear statue with fleece back with a small fabric backpack",
    },
    # 10) Replace the background plants with a detailed city skyline
    {
        "name": "background_city",
        "DATA_NAME": "bear",
        "PROMPT": "Replace the background plants with a city skyline",
        "SEG_PROMPT": "plants and dirt background ground",
        "TARGET_PROMPT": "background showing a city skyline and concrete pavement",
        "STYLE_SOURCE_PROMPT": "background with soil, leaves, and green foliage",
        "STYLE_TARGET_PROMPT": "background with pavement, buildings, and clear city view",
    },
    # 11) Change the color of the front paws to solid gold
    {
        "name": "paws_gold",
        "DATA_NAME": "bear",
        "PROMPT": "Change the color of the bear's front paws to solid gold",
        "SEG_PROMPT": "front paws of the bear statue",
        "TARGET_PROMPT": "front paws colored in solid gold",
        "STYLE_SOURCE_PROMPT": "bear statue with grey stone color",
        "STYLE_TARGET_PROMPT": "bear statue with metallic gold color",
    },
    # 12) Add clear text that says 'GRIZZLY' below the statue on the rock face
    {
        "name": "text_grizzly",
        "DATA_NAME": "bear",
        "PROMPT": "Add text that says 'GRIZZLY' below the statue on the rock",
        "SEG_PROMPT": "flat surface of the rock pedestal",
        "TARGET_PROMPT": "rock pedestal with 'GRIZZLY' text",
        "STYLE_SOURCE_PROMPT": "rock pedestal with plain rock surface",
        "STYLE_TARGET_PROMPT": "rock pedestal with rock surface with detailed black block text",
    },
]

TASKS_BY_NAME = {t["name"]: t for t in TASKS}


# ==========================
# Sweep configuration
# ==========================

SWEEP_CONFIG = {
    "name": WANDB_SWEEP_NAME,
    "method": "grid",
    "metric": {
        "name": "clip_dir_similarity",
        "goal": "maximize",
    },
    "parameters": {
        "task": {
            "values": [t["name"] for t in TASKS],
        },
    },
}


# ==========================
# Helpers
# ==========================


def get_root_dir() -> Path:
    # This script lives in <root>/script/, so parent is project root.
    return Path(__file__).parent.parent.absolute()


def _strip_ansi(s: str) -> str:
    """Remove ANSI color codes from a string."""
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def find_save_directory(launch_output: str) -> Optional[Path]:
    """
    Find save directory from VcEdit launch.py stdout.

    We look for lines like:
        Validation results will be saved to outputs/edit-inf/.../save
    """
    clean_output = _strip_ansi(launch_output)
    pattern = r"Validation results will be saved to (.+)"
    matches = re.findall(pattern, clean_output)
    if matches:
        p = Path(matches[-1].strip())
        if p.exists():
            return p
    return None


def find_render_directory(save_dir: Path) -> Optional[Path]:
    render_dir = save_dir / f"it{MAX_STEPS}-test"
    if render_dir.exists():
        return render_dir
    test_dirs = list(save_dir.glob("it*-test"))
    return test_dirs[0] if test_dirs else None


def parse_metrics(output: str) -> dict:
    """Parse run_clip_metrics_only stdout into a dict."""
    result = {}
    patterns = {
        "clip_dir_consistency": r"^(?:\r)?CLIP directional consistency:\s*([-\d.]+)\s*$",
        "clip_f_scaled": r"^(?:\r)?CLIP_F \(scaled\):\s*([-\d.]+)\s*$",
        "clip_score": r"^(?:\r)?CLIP Score:\s*([-\d.]+)\s*$",
        "clip_dir_similarity": r"^(?:\r)?CLIP directional similarity:\s*([-\d.]+)\s*$",
    }
    for key, pat in patterns.items():
        m = re.search(pat, output, re.MULTILINE)
        if m:
            result[key] = float(m.group(1))
    return result


# ==========================
# Agent function (called per sweep run)
# ==========================


def train_and_evaluate():
    """Single sweep run: pick task from wandb, train, evaluate, log metrics."""
    run = wandb.init()
    cfg = run.config

    gpu = os.environ.get("SWEEP_GPU", GPU_DEFAULT)

    task_name = getattr(cfg, "task", TASKS[0]["name"])
    task = TASKS_BY_NAME[task_name]
    prompt = task["PROMPT"]
    seg_prompt = task["SEG_PROMPT"]
    target_prompt = task["TARGET_PROMPT"]
    style_target_prompt = task.get("STYLE_TARGET_PROMPT", "")
    style_source_prompt = task.get("STYLE_SOURCE_PROMPT", "a Photo")
    data_name = task.get("DATA_NAME", "covered_desk")

    data_source = f"{DATA_SOURCE_ROOT}/{DATA_TYPE}/{data_name}"
    gs_source = (
        f"{GS_SOURCE_ROOT}/{DATA_TYPE}/{data_name}/point_cloud/iteration_30000/point_cloud.ply"
    )
    gt_dir = f"{GS_SOURCE_ROOT}/{DATA_TYPE}/{data_name}/{RENDER_SUBDIR}"

    name = f"edit-inf/{DATA_TYPE}/{data_name}/{task_name}"

    root_dir = get_root_dir()
    os.chdir(root_dir)

    # ---- Build launch command ----
    launch_cmd = [
        sys.executable,
        "launch.py",
        "--config",
        CONFIG,
        "--train",
        "--gpu",
        gpu,
        f"trainer.max_steps={MAX_STEPS}",
        f"data.source={data_source}",
        f"system.gs_source={gs_source}",
        f"system.prompt_processor.prompt={prompt}",
        f"system.seg_prompt={seg_prompt}",
        f"system.guidance.src_prompt={style_source_prompt}",
        f"system.guidance.tgt_prompt={target_prompt}",
        # reasonable defaults copied from 3d-ovs/covered_desk.sh
        "system.max_densify_percent=0.01",
        "system.anchor_weight_init_g0=0.05",
        "system.anchor_weight_init=0.1",
        "system.anchor_weight_multiplier=1.3",
        "system.loss.lambda_anchor_color=0",
        "system.loss.lambda_anchor_geo=50",
        "system.loss.lambda_anchor_scale=50",
        "system.loss.lambda_anchor_opacity=50",
        "system.densify_from_iter=100",
        "system.densify_until_iter=1501",
        "system.densification_interval=100",
        f"system.per_editing_step={PER_EDITING_STEP}",
        # optional: disable wandb inside VcEdit run (we log externally)
        "system.loggers.wandb.enable=false",
        f"name={name}",
    ]

    print(f"\n[Sweep] Running task={task_name}")
    print(f"[Sweep] name={name}\n")

    # ---- Run training ----
    proc = subprocess.Popen(
        launch_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=root_dir,
        env=os.environ.copy(),
    )
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    launch_output = "".join(lines)

    if proc.returncode != 0:
        print("[Sweep] Training failed.")
        wandb.log({"clip_dir_similarity": float("nan")})
        run.finish(exit_code=1)
        return

    # ---- Find render directory ----
    save_dir = find_save_directory(launch_output)
    if not save_dir:
        print("[Sweep] Could not find save directory from stdout.")
        wandb.log({"clip_dir_similarity": float("nan")})
        run.finish(exit_code=1)
        return

    render_dir = find_render_directory(save_dir)
    if not render_dir:
        print(f"[Sweep] Could not find render dir in {save_dir}")
        wandb.log({"clip_dir_similarity": float("nan")})
        run.finish(exit_code=1)
        return

    trial_dir = save_dir.parent
    print(f"[Sweep] Render dir: {render_dir}")
    print(f"[Sweep] Trial dir:  {trial_dir}")

    # ---- Run CLIP metrics via run_clip_metrics_only.py ----
    gpu_id = gpu.split(",")[0].strip()
    metrics_cmd = [
        sys.executable,
        "script/run_clip_metrics_only.py",
        "--output_dir",
        str(trial_dir),
        "--gt_dir",
        gt_dir,
        "--gpu",
        gpu_id,
        "--device",
        DEVICE,
        "--interval",
        str(INTERVAL),
        "--style_source_prompt",
        style_source_prompt,
        "--style_target_prompt",
        style_target_prompt,
        "--max_steps",
        MAX_STEPS,
    ]

    metrics_proc = subprocess.Popen(
        metrics_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=root_dir,
        env=os.environ.copy(),
    )
    metrics_lines = []
    for line in metrics_proc.stdout:
        print(line, end="", flush=True)
        metrics_lines.append(line)
    metrics_proc.wait()
    metrics_output = "".join(metrics_lines)

    if metrics_proc.returncode != 0:
        print("[Sweep] Metrics failed.")
        wandb.log({"clip_dir_similarity": float("nan")})
        run.finish(exit_code=1)
        return

    # ---- Parse and log metrics ----
    metrics = parse_metrics(metrics_output)
    print(f"\n[Sweep] Metrics: {metrics}")

    if not metrics:
        print("[Sweep] Could not parse any metrics from output.")
        wandb.log({"clip_dir_similarity": float("nan")})
    else:
        wandb.log(metrics)

    run.finish()


# ==========================
# Entry point
# ==========================


def main():
    parser = argparse.ArgumentParser(
        description="WandB Sweep for VcEdit edit-inf (3D-OVS)"
    )
    parser.add_argument("--create", action="store_true", help="Create a new sweep")
    parser.add_argument("--agent", action="store_true", help="Start a sweep agent")
    parser.add_argument("--sweep_id", type=str, default=None, help="Sweep ID")
    parser.add_argument(
        "--count", type=int, default=None, help="Max number of runs per agent"
    )
    parser.add_argument(
        "--gpu", type=str, default=None, help="GPU id for this specific agent"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPUs (e.g., '0,1,2,3') to run parallel agents",
    )
    args = parser.parse_args()

    sweep_id = args.sweep_id

    if args.create:
        sweep_id = wandb.sweep(SWEEP_CONFIG, project=WANDB_PROJECT)
        print(f"\n[Created] Sweep ID: {sweep_id}")

    if args.agent:
        if not sweep_id:
            print("Error: --sweep_id is required.")
            sys.exit(1)

        # Multiple GPUs: spawn one agent process per GPU
        if args.gpus:
            gpu_list = [s.strip() for s in args.gpus.split(",") if s.strip()]
            print(f"Launching {len(gpu_list)} agents on GPUs: {gpu_list}")

            processes = []
            for g in gpu_list:
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = g
                env["SWEEP_GPU"] = "0"  # inside each process, GPU is index 0

                cmd = [
                    sys.executable,
                    "-u",
                    sys.argv[0],
                    "--agent",
                    "--sweep_id",
                    sweep_id,
                    "--gpu",
                    "0",
                ]
                if args.count:
                    cmd += ["--count", str(args.count)]

                p = subprocess.Popen(cmd, env=env)
                processes.append(p)

            for p in processes:
                p.wait()
            sys.exit(0)

        # Single agent (possibly on a single physical GPU)
        gpu_to_use = args.gpu if args.gpu else "0"
        os.environ["SWEEP_GPU"] = gpu_to_use

        print(
            f"Agent started on Physical GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'Unknown')}"
        )

        wandb.agent(
            sweep_id,
            function=train_and_evaluate,
            project=WANDB_PROJECT,
            count=args.count,
        )


if __name__ == "__main__":
    main()

