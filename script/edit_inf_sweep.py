#!/usr/bin/env python3
"""
WandB Sweep for VcEdit edit-inf experiments on 3D-OVS-style datasets.

- Sweep target metric: CLIP directional similarity (maximize)
- Search space: only the "task" (prompt / seg_prompt combo), no hyperparameter tuning.

Usage:
  # 1) Create sweep (run once)
  python script/edit_inf_sweep.py --create

  # 2) Run agent(s) – each agent picks a task and runs train+metrics
  python script/edit_inf_sweep.py --agent --sweep_id <SWEEP_ID>

  # Use specific GPU(s): single GPU or comma-separated for multi-GPU per run
  python script/edit_inf_sweep.py --agent --sweep_id <ID> --gpu 0
  python script/edit_inf_sweep.py --agent --sweep_id <ID> --gpus 0,1,2
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

DATA_TYPE = "3d-ovs"
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
    ## covered_desk
    # 1) Change the shaving razor into an apple
    {
        "name": "razor_to_apple",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Change the shaving razor into an apple",
        "SEG_PROMPT": "shaving razor",
        "TARGET_PROMPT": "apple",
        "STYLE_SOURCE_PROMPT": "shaving razor",
        "STYLE_TARGET_PROMPT": "apple",
    },
    # 2) Change the shampoo bottle into an apple
    {
        "name": "bottle_to_apple",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Change the shampoo bottle into an apple",
        "SEG_PROMPT": "shampoo bottle",
        "TARGET_PROMPT": "apple",
        "STYLE_SOURCE_PROMPT": "shampoo bottle",
        "STYLE_TARGET_PROMPT": "apple",
    },
    # 3) Give the pooh a pair of pants
    {
        "name": "pooh_pants",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Give the pooh a pair of pants",
        "SEG_PROMPT": "pooh",
        "TARGET_PROMPT": "pants",
        "STYLE_SOURCE_PROMPT": "pooh",
        "STYLE_TARGET_PROMPT": "pooh wearing pants",
    },
    # 4) Make the pooh look like a penguin
    {
        "name": "pooh_penguin",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Make the pooh look like a penguin",
        "SEG_PROMPT": "pooh",
        "TARGET_PROMPT": "penguin",
        "STYLE_SOURCE_PROMPT": "pooh",
        "STYLE_TARGET_PROMPT": "penguin",
    },
    # 5) Change the red sweater into a leather jacket
    {
        "name": "sweater_to_leather_jacket",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Change the red sweater into a leather jacket",
        "SEG_PROMPT": "red sweater of the pooh",
        "TARGET_PROMPT": "leather jacket of the pooh",
        "STYLE_SOURCE_PROMPT": "red sweater of the pooh",
        "STYLE_TARGET_PROMPT": "leather jacket of the pooh",
    },
    # 6) Make the pooh wear sunglasses on his face
    {
        "name": "pooh_sunglasses",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Make the pooh wear sunglasses on his face",
        "SEG_PROMPT": "head of the pooh",
        "TARGET_PROMPT": "sunglasses",
        "STYLE_SOURCE_PROMPT": "head of the pooh",
        "STYLE_TARGET_PROMPT": "pooh wearing sunglasses",
    },
    # 7) Change the pooh's sweater color to blue
    {
        "name": "sweater_blue",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Change the pooh's sweater color to blue",
        "SEG_PROMPT": "red sweater of the pooh",
        "TARGET_PROMPT": "blue sweater of the pooh",
        "STYLE_SOURCE_PROMPT": "red sweater of the pooh",
        "STYLE_TARGET_PROMPT": "blue sweater of the pooh",
    },
    # 8) Add flower pattern to the pooh's sweater
    {
        "name": "sweater_flower",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Add flower pattern to the pooh's sweater",
        "SEG_PROMPT": "red sweater of the pooh",
        "TARGET_PROMPT": "sweater of the pooh with flower pattern",
        "STYLE_SOURCE_PROMPT": "red sweater of the pooh",
        "STYLE_TARGET_PROMPT": "sweater of the pooh with flower pattern",
    },
    # 9) Make the pooh look like a panda
    {
        "name": "pooh_panda",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Make the pooh look like a panda",
        "SEG_PROMPT": "pooh",
        "TARGET_PROMPT": "panda",
        "STYLE_SOURCE_PROMPT": "pooh",
        "STYLE_TARGET_PROMPT": "panda",
    },
    # 10) Make the pooh look like a robot
    {
        "name": "pooh_robot",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Make the pooh look like a robot",
        "SEG_PROMPT": "pooh",
        "TARGET_PROMPT": "robot",
        "STYLE_SOURCE_PROMPT": "pooh",
        "STYLE_TARGET_PROMPT": "robot",
    },


    ## blue_sofa
    # 1) Change the plush toy's color to pink
    {
        "name": "plush_pink",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Change the plush toy's color to pink",
        "SEG_PROMPT": "yellow plush toy",
        "TARGET_PROMPT": "pink plush toy",
        "STYLE_SOURCE_PROMPT": "yellow plush toy",
        "STYLE_TARGET_PROMPT": "pink plush toy",
    },
    # 2) Make the plush toy wear a tiny hat
    {
        "name": "plush_hat",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Make the plush toy wear a tiny hat",
        "SEG_PROMPT": "head of the plush toy",
        "TARGET_PROMPT": "head of the plush toy wearing a tiny party hat",
        "STYLE_SOURCE_PROMPT": "head of the plush toy",
        "STYLE_TARGET_PROMPT": "head of the plush toy wearing a tiny party hat",
    },
    # 3) Change the sunglasses to red frames
    {
        "name": "glasses_red",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Change the sunglasses to red frames",
        "SEG_PROMPT": "black frames of the sunglasses",
        "TARGET_PROMPT": "bright red frames of the sunglasses",
        "STYLE_SOURCE_PROMPT": "black frames of the sunglasses",
        "STYLE_TARGET_PROMPT": "bright red frames of the sunglasses",
    },
    # 4) Replace the JBL speaker with a vintage radio
    {
        "name": "speaker_vintage",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Replace the JBL speaker with a vintage radio",
        "SEG_PROMPT": "grey JBL speaker",
        "TARGET_PROMPT": "vintage wooden radio",
        "STYLE_SOURCE_PROMPT": "grey JBL speaker",
        "STYLE_TARGET_PROMPT": "vintage wooden radio",
    },
    # 5) Change the perfume liquid color to blue
    {
        "name": "perfume_blue",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Change the perfume liquid color to blue",
        "SEG_PROMPT": "yellow liquid inside the perfume bottle",
        "TARGET_PROMPT": "ocean blue liquid inside the perfume bottle",
        "STYLE_SOURCE_PROMPT": "yellow liquid inside the perfume bottle",
        "STYLE_TARGET_PROMPT": "ocean blue liquid inside the perfume bottle",
    },
    # 6) Add a digital clock display to the remote controller
    {
        "name": "remote_digital",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Make the remote controller screen glow neon green",
        "SEG_PROMPT": "screen of the white remote controller",
        "TARGET_PROMPT": "glowing neon green screen of the remote controller",
        "STYLE_SOURCE_PROMPT": "screen of the white remote controller",
        "STYLE_TARGET_PROMPT": "glowing neon green screen of the remote controller",
    },
    # 7) Turn the plush toy into a tiger
    {
        "name": "plush_tiger",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Change the plush toy's pattern to tiger stripes",
        "SEG_PROMPT": "plush toy",
        "TARGET_PROMPT": "tiger striped plush toy",
        "STYLE_SOURCE_PROMPT": "plush toy",
        "STYLE_TARGET_PROMPT": "tiger striped plush toy",
    },
    # 8) Change the speaker color to gold
    {
        "name": "speaker_gold",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Change the speaker to a shiny gold texture",
        "SEG_PROMPT": "grey speaker body",
        "TARGET_PROMPT": "shiny metallic gold speaker body",
        "STYLE_SOURCE_PROMPT": "grey speaker body",
        "STYLE_TARGET_PROMPT": "shiny metallic gold speaker body",
    },
    # 9) Make the sunglasses look like aviator glasses
    {
        "name": "glasses_aviator",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Change the sunglasses style to gold-rimmed aviators",
        "SEG_PROMPT": "black sunglasses",
        "TARGET_PROMPT": "gold-rimmed aviator sunglasses",
        "STYLE_SOURCE_PROMPT": "black sunglasses",
        "STYLE_TARGET_PROMPT": "gold-rimmed aviator sunglasses",
    },
    # 10) Change the perfume bottle cap to silver
    {
        "name": "perfume_silver_cap",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Change the perfume bottle cap to silver",
        "SEG_PROMPT": "black cap of the perfume bottle",
        "TARGET_PROMPT": "polished silver cap of the perfume bottle",
        "STYLE_SOURCE_PROMPT": "black cap of the perfume bottle",
        "STYLE_TARGET_PROMPT": "polished silver cap of the perfume bottle",
    },

    ## room
    # 1) Change the rubber chicken's color to red
    {
        "name": "chicken_red",
        "DATA_NAME": "room",
        "PROMPT": "Change the rubber chicken's color to red",
        "SEG_PROMPT": "yellow rubber chicken",
        "TARGET_PROMPT": "red rubber chicken",
        "STYLE_SOURCE_PROMPT": "yellow rubber chicken",
        "STYLE_TARGET_PROMPT": "red rubber chicken",
    },
    # 2) Make the rabbit figure white
    {
        "name": "rabbit_white",
        "DATA_NAME": "room",
        "PROMPT": "Make the rabbit figure white",
        "SEG_PROMPT": "grey rabbit figure",
        "TARGET_PROMPT": "white rabbit figure",
        "STYLE_SOURCE_PROMPT": "grey rabbit figure",
        "STYLE_TARGET_PROMPT": "white rabbit figure",
    },
    # 3) Change the dinosaur figure to green
    {
        "name": "dino_green",
        "DATA_NAME": "room",
        "PROMPT": "Change the dinosaur figure to green",
        "SEG_PROMPT": "brown dinosaur figure",
        "TARGET_PROMPT": "green dinosaur figure",
        "STYLE_SOURCE_PROMPT": "brown dinosaur figure",
        "STYLE_TARGET_PROMPT": "green dinosaur figure",
    },
    # 4) Replace the baseball with a tennis ball
    {
        "name": "ball_tennis",
        "DATA_NAME": "room",
        "PROMPT": "Replace the baseball with a tennis ball",
        "SEG_PROMPT": "white baseball",
        "TARGET_PROMPT": "yellow tennis ball",
        "STYLE_SOURCE_PROMPT": "white baseball",
        "STYLE_TARGET_PROMPT": "yellow tennis ball",
    },
    # 5) Change the basket material to wood
    {
        "name": "basket_wood",
        "DATA_NAME": "room",
        "PROMPT": "Change the basket material to dark wood",
        "SEG_PROMPT": "woven basket",
        "TARGET_PROMPT": "dark wooden basket",
        "STYLE_SOURCE_PROMPT": "woven basket",
        "STYLE_TARGET_PROMPT": "dark wooden basket",
    },
    # 6) Add a tiny bow tie to the rubber chicken
    {
        "name": "chicken_bow_tie",
        "DATA_NAME": "room",
        "PROMPT": "Add a tiny blue bow tie to the rubber chicken's neck",
        "SEG_PROMPT": "neck of the yellow rubber chicken",
        "TARGET_PROMPT": "yellow rubber chicken wearing a tiny blue bow tie",
        "STYLE_SOURCE_PROMPT": "neck of the yellow rubber chicken",
        "STYLE_TARGET_PROMPT": "yellow rubber chicken wearing a tiny blue bow tie",
    },
    # 7) Give the rabbit figure sunglasses
    {
        "name": "rabbit_sunglasses",
        "DATA_NAME": "room",
        "PROMPT": "Give the rabbit figure a pair of small sunglasses",
        "SEG_PROMPT": "face of the grey rabbit figure",
        "TARGET_PROMPT": "grey rabbit figure wearing small sunglasses",
        "STYLE_SOURCE_PROMPT": "face of the grey rabbit figure",
        "STYLE_TARGET_PROMPT": "grey rabbit figure wearing small sunglasses",
    },
    # 8) Make the dinosaur figure look like it's made of metal
    {
        "name": "dino_metal",
        "DATA_NAME": "room",
        "PROMPT": "Make the dinosaur figure look like it's made of shiny metal",
        "SEG_PROMPT": "brown dinosaur figure",
        "TARGET_PROMPT": "shiny metallic dinosaur figure",
        "STYLE_SOURCE_PROMPT": "brown dinosaur figure",
        "STYLE_TARGET_PROMPT": "shiny metallic dinosaur figure",
    },
    # 9) Change the baseball to a golden ball
    {
        "name": "ball_gold",
        "DATA_NAME": "room",
        "PROMPT": "Change the baseball to a solid golden ball",
        "SEG_PROMPT": "white baseball",
        "TARGET_PROMPT": "solid golden ball",
        "STYLE_SOURCE_PROMPT": "white baseball",
        "STYLE_TARGET_PROMPT": "solid golden ball",
    },
    # 10) Fill the basket with apples
    {
        "name": "basket_apples",
        "DATA_NAME": "room",
        "PROMPT": "Fill the empty space in the basket with red apples",
        "SEG_PROMPT": "inside of the woven basket",
        "TARGET_PROMPT": "woven basket filled with red apples",
        "STYLE_SOURCE_PROMPT": "inside of the woven basket",
        "STYLE_TARGET_PROMPT": "woven basket filled with red apples",
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

