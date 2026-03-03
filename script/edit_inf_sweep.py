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
    # ## covered_desk
    # # 1) Change the shaving razor into an apple
    # {
    #     "name": "razor_to_apple",
    #     "DATA_NAME": "covered_desk",
    #     "PROMPT": "Change the shaving razor into an apple",
    #     "SEG_PROMPT": "shaving razor",
    #     "TARGET_PROMPT": "apple",
    #     "STYLE_SOURCE_PROMPT": "shaving razor",
    #     "STYLE_TARGET_PROMPT": "apple",
    # },
    # # 2) Change the shampoo bottle into an apple
    # {
    #     "name": "bottle_to_apple",
    #     "DATA_NAME": "covered_desk",
    #     "PROMPT": "Change the shampoo bottle into an apple",
    #     "SEG_PROMPT": "shampoo bottle",
    #     "TARGET_PROMPT": "apple",
    #     "STYLE_SOURCE_PROMPT": "shampoo bottle",
    #     "STYLE_TARGET_PROMPT": "apple",
    # },
    # # 3) Give the pooh a pair of pants
    # {
    #     "name": "pooh_pants",
    #     "DATA_NAME": "covered_desk",
    #     "PROMPT": "Give the pooh a pair of pants",
    #     "SEG_PROMPT": "pooh",
    #     "TARGET_PROMPT": "pants",
    #     "STYLE_SOURCE_PROMPT": "pooh",
    #     "STYLE_TARGET_PROMPT": "pooh wearing pants",
    # },
    # # 4) Make the pooh look like a penguin
    # {
    #     "name": "pooh_penguin",
    #     "DATA_NAME": "covered_desk",
    #     "PROMPT": "Make the pooh look like a penguin",
    #     "SEG_PROMPT": "pooh",
    #     "TARGET_PROMPT": "penguin",
    #     "STYLE_SOURCE_PROMPT": "pooh",
    #     "STYLE_TARGET_PROMPT": "penguin",
    # },
    # # 5) Change the red sweater into a leather jacket
    # {
    #     "name": "sweater_to_leather_jacket",
    #     "DATA_NAME": "covered_desk",
    #     "PROMPT": "Change the red sweater into a leather jacket",
    #     "SEG_PROMPT": "red sweater of the pooh",
    #     "TARGET_PROMPT": "leather jacket of the pooh",
    #     "STYLE_SOURCE_PROMPT": "red sweater of the pooh",
    #     "STYLE_TARGET_PROMPT": "leather jacket of the pooh",
    # },
    # # 6) Make the pooh wear sunglasses on his face
    # {
    #     "name": "pooh_sunglasses",
    #     "DATA_NAME": "covered_desk",
    #     "PROMPT": "Make the pooh wear sunglasses on his face",
    #     "SEG_PROMPT": "head of the pooh",
    #     "TARGET_PROMPT": "sunglasses",
    #     "STYLE_SOURCE_PROMPT": "head of the pooh",
    #     "STYLE_TARGET_PROMPT": "pooh wearing sunglasses",
    # },
    # # 7) Change the pooh's sweater color to blue
    # {
    #     "name": "sweater_blue",
    #     "DATA_NAME": "covered_desk",
    #     "PROMPT": "Change the pooh's sweater color to blue",
    #     "SEG_PROMPT": "red sweater of the pooh",
    #     "TARGET_PROMPT": "blue sweater of the pooh",
    #     "STYLE_SOURCE_PROMPT": "red sweater of the pooh",
    #     "STYLE_TARGET_PROMPT": "blue sweater of the pooh",
    # },
    # # 8) Add flower pattern to the pooh's sweater
    # {
    #     "name": "sweater_flower",
    #     "DATA_NAME": "covered_desk",
    #     "PROMPT": "Add flower pattern to the pooh's sweater",
    #     "SEG_PROMPT": "red sweater of the pooh",
    #     "TARGET_PROMPT": "sweater of the pooh with flower pattern",
    #     "STYLE_SOURCE_PROMPT": "red sweater of the pooh",
    #     "STYLE_TARGET_PROMPT": "sweater of the pooh with flower pattern",
    # },
    # # 9) Make the pooh look like a panda
    # {
    #     "name": "pooh_panda",
    #     "DATA_NAME": "covered_desk",
    #     "PROMPT": "Make the pooh look like a panda",
    #     "SEG_PROMPT": "pooh",
    #     "TARGET_PROMPT": "panda",
    #     "STYLE_SOURCE_PROMPT": "pooh",
    #     "STYLE_TARGET_PROMPT": "panda",
    # },
    # # 10) Make the pooh look like a robot
    # {
    #     "name": "pooh_robot",
    #     "DATA_NAME": "covered_desk",
    #     "PROMPT": "Make the pooh look like a robot",
    #     "SEG_PROMPT": "pooh",
    #     "TARGET_PROMPT": "robot",
    #     "STYLE_SOURCE_PROMPT": "pooh",
    #     "STYLE_TARGET_PROMPT": "robot",
    # },


    # ## blue_sofa
    # # 1) Change the plush toy's color to pink
    # {
    #     "name": "plush_pink",
    #     "DATA_NAME": "blue_sofa",
    #     "PROMPT": "Change the plush toy's color to pink",
    #     "SEG_PROMPT": "yellow plush toy",
    #     "TARGET_PROMPT": "pink plush toy",
    #     "STYLE_SOURCE_PROMPT": "yellow plush toy",
    #     "STYLE_TARGET_PROMPT": "pink plush toy",
    # },
    # # 2) Make the plush toy wear a tiny hat
    # {
    #     "name": "plush_hat",
    #     "DATA_NAME": "blue_sofa",
    #     "PROMPT": "Make the plush toy wear a tiny hat",
    #     "SEG_PROMPT": "head of the plush toy",
    #     "TARGET_PROMPT": "head of the plush toy wearing a tiny party hat",
    #     "STYLE_SOURCE_PROMPT": "head of the plush toy",
    #     "STYLE_TARGET_PROMPT": "head of the plush toy wearing a tiny party hat",
    # },
    # # 3) Change the sunglasses to red frames
    # {
    #     "name": "glasses_red",
    #     "DATA_NAME": "blue_sofa",
    #     "PROMPT": "Change the sunglasses to red frames",
    #     "SEG_PROMPT": "black frames of the sunglasses",
    #     "TARGET_PROMPT": "bright red frames of the sunglasses",
    #     "STYLE_SOURCE_PROMPT": "black frames of the sunglasses",
    #     "STYLE_TARGET_PROMPT": "bright red frames of the sunglasses",
    # },
    # # 4) Replace the JBL speaker with a vintage radio
    # {
    #     "name": "speaker_vintage",
    #     "DATA_NAME": "blue_sofa",
    #     "PROMPT": "Replace the JBL speaker with a vintage radio",
    #     "SEG_PROMPT": "grey JBL speaker",
    #     "TARGET_PROMPT": "vintage wooden radio",
    #     "STYLE_SOURCE_PROMPT": "grey JBL speaker",
    #     "STYLE_TARGET_PROMPT": "vintage wooden radio",
    # },
    # # 5) Change the perfume liquid color to blue
    # {
    #     "name": "perfume_blue",
    #     "DATA_NAME": "blue_sofa",
    #     "PROMPT": "Change the perfume liquid color to blue",
    #     "SEG_PROMPT": "yellow liquid inside the perfume bottle",
    #     "TARGET_PROMPT": "ocean blue liquid inside the perfume bottle",
    #     "STYLE_SOURCE_PROMPT": "yellow liquid inside the perfume bottle",
    #     "STYLE_TARGET_PROMPT": "ocean blue liquid inside the perfume bottle",
    # },
    # # 6) Add a digital clock display to the remote controller
    # {
    #     "name": "remote_digital",
    #     "DATA_NAME": "blue_sofa",
    #     "PROMPT": "Make the remote controller screen glow neon green",
    #     "SEG_PROMPT": "screen of the white remote controller",
    #     "TARGET_PROMPT": "glowing neon green screen of the remote controller",
    #     "STYLE_SOURCE_PROMPT": "screen of the white remote controller",
    #     "STYLE_TARGET_PROMPT": "glowing neon green screen of the remote controller",
    # },
    # # 7) Turn the plush toy into a tiger
    # {
    #     "name": "plush_tiger",
    #     "DATA_NAME": "blue_sofa",
    #     "PROMPT": "Change the plush toy's pattern to tiger stripes",
    #     "SEG_PROMPT": "plush toy",
    #     "TARGET_PROMPT": "tiger striped plush toy",
    #     "STYLE_SOURCE_PROMPT": "plush toy",
    #     "STYLE_TARGET_PROMPT": "tiger striped plush toy",
    # },
    # # 8) Change the speaker color to gold
    # {
    #     "name": "speaker_gold",
    #     "DATA_NAME": "blue_sofa",
    #     "PROMPT": "Change the speaker to a shiny gold texture",
    #     "SEG_PROMPT": "grey speaker body",
    #     "TARGET_PROMPT": "shiny metallic gold speaker body",
    #     "STYLE_SOURCE_PROMPT": "grey speaker body",
    #     "STYLE_TARGET_PROMPT": "shiny metallic gold speaker body",
    # },
    # # 9) Make the sunglasses look like aviator glasses
    # {
    #     "name": "glasses_aviator",
    #     "DATA_NAME": "blue_sofa",
    #     "PROMPT": "Change the sunglasses style to gold-rimmed aviators",
    #     "SEG_PROMPT": "black sunglasses",
    #     "TARGET_PROMPT": "gold-rimmed aviator sunglasses",
    #     "STYLE_SOURCE_PROMPT": "black sunglasses",
    #     "STYLE_TARGET_PROMPT": "gold-rimmed aviator sunglasses",
    # },
    # # 10) Change the perfume bottle cap to silver
    # {
    #     "name": "perfume_silver_cap",
    #     "DATA_NAME": "blue_sofa",
    #     "PROMPT": "Change the perfume bottle cap to silver",
    #     "SEG_PROMPT": "black cap of the perfume bottle",
    #     "TARGET_PROMPT": "polished silver cap of the perfume bottle",
    #     "STYLE_SOURCE_PROMPT": "black cap of the perfume bottle",
    #     "STYLE_TARGET_PROMPT": "polished silver cap of the perfume bottle",
    # },

    # ## room
    # # 1) Change the rubber chicken's color to red
    # {
    #     "name": "chicken_red",
    #     "DATA_NAME": "room",
    #     "PROMPT": "Change the rubber chicken's color to red",
    #     "SEG_PROMPT": "yellow rubber chicken",
    #     "TARGET_PROMPT": "red rubber chicken",
    #     "STYLE_SOURCE_PROMPT": "yellow rubber chicken",
    #     "STYLE_TARGET_PROMPT": "red rubber chicken",
    # },
    # # 2) Make the rabbit figure white
    # {
    #     "name": "rabbit_white",
    #     "DATA_NAME": "room",
    #     "PROMPT": "Make the rabbit figure white",
    #     "SEG_PROMPT": "grey rabbit figure",
    #     "TARGET_PROMPT": "white rabbit figure",
    #     "STYLE_SOURCE_PROMPT": "grey rabbit figure",
    #     "STYLE_TARGET_PROMPT": "white rabbit figure",
    # },
    # # 3) Change the dinosaur figure to green
    # {
    #     "name": "dino_green",
    #     "DATA_NAME": "room",
    #     "PROMPT": "Change the dinosaur figure to green",
    #     "SEG_PROMPT": "brown dinosaur figure",
    #     "TARGET_PROMPT": "green dinosaur figure",
    #     "STYLE_SOURCE_PROMPT": "brown dinosaur figure",
    #     "STYLE_TARGET_PROMPT": "green dinosaur figure",
    # },
    # # 4) Replace the baseball with a tennis ball
    # {
    #     "name": "ball_tennis",
    #     "DATA_NAME": "room",
    #     "PROMPT": "Replace the baseball with a tennis ball",
    #     "SEG_PROMPT": "white baseball",
    #     "TARGET_PROMPT": "yellow tennis ball",
    #     "STYLE_SOURCE_PROMPT": "white baseball",
    #     "STYLE_TARGET_PROMPT": "yellow tennis ball",
    # },
    # # 5) Change the basket material to wood
    # {
    #     "name": "basket_wood",
    #     "DATA_NAME": "room",
    #     "PROMPT": "Change the basket material to dark wood",
    #     "SEG_PROMPT": "woven basket",
    #     "TARGET_PROMPT": "dark wooden basket",
    #     "STYLE_SOURCE_PROMPT": "woven basket",
    #     "STYLE_TARGET_PROMPT": "dark wooden basket",
    # },
    # # 6) Add a tiny bow tie to the rubber chicken
    # {
    #     "name": "chicken_bow_tie",
    #     "DATA_NAME": "room",
    #     "PROMPT": "Add a tiny blue bow tie to the rubber chicken's neck",
    #     "SEG_PROMPT": "neck of the yellow rubber chicken",
    #     "TARGET_PROMPT": "yellow rubber chicken wearing a tiny blue bow tie",
    #     "STYLE_SOURCE_PROMPT": "neck of the yellow rubber chicken",
    #     "STYLE_TARGET_PROMPT": "yellow rubber chicken wearing a tiny blue bow tie",
    # },
    # # 7) Give the rabbit figure sunglasses
    # {
    #     "name": "rabbit_sunglasses",
    #     "DATA_NAME": "room",
    #     "PROMPT": "Give the rabbit figure a pair of small sunglasses",
    #     "SEG_PROMPT": "face of the grey rabbit figure",
    #     "TARGET_PROMPT": "grey rabbit figure wearing small sunglasses",
    #     "STYLE_SOURCE_PROMPT": "face of the grey rabbit figure",
    #     "STYLE_TARGET_PROMPT": "grey rabbit figure wearing small sunglasses",
    # },
    # # 8) Make the dinosaur figure look like it's made of metal
    # {
    #     "name": "dino_metal",
    #     "DATA_NAME": "room",
    #     "PROMPT": "Make the dinosaur figure look like it's made of shiny metal",
    #     "SEG_PROMPT": "brown dinosaur figure",
    #     "TARGET_PROMPT": "shiny metallic dinosaur figure",
    #     "STYLE_SOURCE_PROMPT": "brown dinosaur figure",
    #     "STYLE_TARGET_PROMPT": "shiny metallic dinosaur figure",
    # },
    # # 9) Change the baseball to a golden ball
    # {
    #     "name": "ball_gold",
    #     "DATA_NAME": "room",
    #     "PROMPT": "Change the baseball to a solid golden ball",
    #     "SEG_PROMPT": "white baseball",
    #     "TARGET_PROMPT": "solid golden ball",
    #     "STYLE_SOURCE_PROMPT": "white baseball",
    #     "STYLE_TARGET_PROMPT": "solid golden ball",
    # },
    # # 10) Fill the basket with apples
    # {
    #     "name": "basket_apples",
    #     "DATA_NAME": "room",
    #     "PROMPT": "Fill the empty space in the basket with red apples",
    #     "SEG_PROMPT": "inside of the woven basket",
    #     "TARGET_PROMPT": "woven basket filled with red apples",
    #     "STYLE_SOURCE_PROMPT": "inside of the woven basket",
    #     "STYLE_TARGET_PROMPT": "woven basket filled with red apples",
    # },
    # 1) Change the red flower into a sunflower
    {
        "name": "flower_to_sunflower",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Change the red flower into a sunflower",
        "SEG_PROMPT": "red flower",
        "TARGET_PROMPT": "sunflower",
        "STYLE_SOURCE_PROMPT": "red flower",
        "STYLE_TARGET_PROMPT": "sunflower",
    },
    # 2) Change the shampoo bottle into a wine bottle
    {
        "name": "bottle_to_wine",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Change the shampoo bottle into a wine bottle",
        "SEG_PROMPT": "shampoo bottle",
        "TARGET_PROMPT": "wine bottle",
        "STYLE_SOURCE_PROMPT": "shampoo bottle",
        "STYLE_TARGET_PROMPT": "wine bottle",
    },
    # 3) Make the pooh wear a black hat
    {
        "name": "pooh_hat",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Make the pooh wear a black hat",
        "SEG_PROMPT": "head of the pooh",
        "TARGET_PROMPT": "black hat",
        "STYLE_SOURCE_PROMPT": "head of the pooh",
        "STYLE_TARGET_PROMPT": "pooh wearing a black hat",
    },
    # 4) Change the shaving razor into a toy car
    {
        "name": "razor_to_car",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Change the shaving razor into a toy car",
        "SEG_PROMPT": "shaving razor",
        "TARGET_PROMPT": "toy car",
        "STYLE_SOURCE_PROMPT": "shaving razor",
        "STYLE_TARGET_PROMPT": "toy car",
    },
    # 5) Change the jam jar into a honey pot
    {
        "name": "jar_to_honey",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Change the jam jar into a honey pot",
        "SEG_PROMPT": "small jar on the right",
        "TARGET_PROMPT": "honey pot",
        "STYLE_SOURCE_PROMPT": "small jar on the right",
        "STYLE_TARGET_PROMPT": "honey pot",
    },
    # 6) Change the pooh's sweater color to green
    {
        "name": "sweater_green",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Change the pooh's sweater color to green",
        "SEG_PROMPT": "red sweater of the pooh",
        "TARGET_PROMPT": "green sweater of the pooh",
        "STYLE_SOURCE_PROMPT": "red sweater of the pooh",
        "STYLE_TARGET_PROMPT": "green sweater of the pooh",
    },
    # 7) Make the pooh look like a tiger
    {
        "name": "pooh_tigger",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Make the pooh look like a tiger",
        "SEG_PROMPT": "pooh",
        "TARGET_PROMPT": "tiger",
        "STYLE_SOURCE_PROMPT": "pooh",
        "STYLE_TARGET_PROMPT": "tiger with orange and black stripes",
    },
    # 8) Replace the red flower with a blue butterfly
    {
        "name": "flower_to_butterfly",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Replace the red flower with a blue butterfly",
        "SEG_PROMPT": "red flower",
        "TARGET_PROMPT": "blue butterfly",
        "STYLE_SOURCE_PROMPT": "red flower",
        "STYLE_TARGET_PROMPT": "blue butterfly",
    },
    # 9) Put a red bowtie on the pooh
    {
        "name": "pooh_bowtie",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Put a red bowtie on the pooh",
        "SEG_PROMPT": "neck of the pooh",
        "TARGET_PROMPT": "red bowtie",
        "STYLE_SOURCE_PROMPT": "neck of the pooh",
        "STYLE_TARGET_PROMPT": "pooh wearing a red bowtie",
    },
    # 10) Change the shampoo bottle into a cactus
    {
        "name": "bottle_to_cactus",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Change the shampoo bottle into a cactus",
        "SEG_PROMPT": "shampoo bottle",
        "TARGET_PROMPT": "cactus",
        "STYLE_SOURCE_PROMPT": "shampoo bottle",
        "STYLE_TARGET_PROMPT": "cactus in a terracotta pot",
    },
    # 11) Add a pair of headphones to the pooh
    {
        "name": "pooh_headphones",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Add a pair of headphones to the pooh",
        "SEG_PROMPT": "head of the pooh",
        "TARGET_PROMPT": "headphones",
        "STYLE_SOURCE_PROMPT": "head of the pooh",
        "STYLE_TARGET_PROMPT": "pooh wearing modern headphones",
    },
    # 12) Change the shaving razor into a banana
    {
        "name": "razor_to_banana",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Change the shaving razor into a banana",
        "SEG_PROMPT": "shaving razor",
        "TARGET_PROMPT": "banana",
        "STYLE_SOURCE_PROMPT": "shaving razor",
        "STYLE_TARGET_PROMPT": "ripe yellow banana",
    },
    # 13) Make the pooh wear sunglasses
    {
        "name": "pooh_sunglasses",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Make the pooh wear sunglasses",
        "SEG_PROMPT": "face of the pooh",
        "TARGET_PROMPT": "sunglasses",
        "STYLE_SOURCE_PROMPT": "face of the pooh",
        "STYLE_TARGET_PROMPT": "pooh wearing cool sunglasses",
    },
    # 14) Change the red sweater into a tuxedo
    {
        "name": "sweater_to_tuxedo",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Change the red sweater into a tuxedo",
        "SEG_PROMPT": "red sweater of the pooh",
        "TARGET_PROMPT": "tuxedo",
        "STYLE_SOURCE_PROMPT": "red sweater of the pooh",
        "STYLE_TARGET_PROMPT": "pooh wearing a formal tuxedo",
    },
    # 15) Change the jam jar into a gold trophy
    {
        "name": "jar_to_trophy",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Change the jam jar into a gold trophy",
        "SEG_PROMPT": "small jar",
        "TARGET_PROMPT": "gold trophy",
        "STYLE_SOURCE_PROMPT": "small jar",
        "STYLE_TARGET_PROMPT": "shining gold trophy",
    },
## blue_sofa_extended
    # 11) Change the sofa material to brown leather
    {
        "name": "sofa_leather",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Change the blue sofa to a brown leather texture",
        "SEG_PROMPT": "blue sofa background",
        "TARGET_PROMPT": "rustic brown leather sofa background",
        "STYLE_SOURCE_PROMPT": "blue sofa background",
        "STYLE_TARGET_PROMPT": "rustic brown leather sofa background",
    },
    # 12) Turn the plush toy into a panda
    {
        "name": "plush_panda",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Turn the plush toy into a black and white panda",
        "SEG_PROMPT": "yellow plush toy",
        "TARGET_PROMPT": "black and white panda plush toy",
        "STYLE_SOURCE_PROMPT": "yellow plush toy",
        "STYLE_TARGET_PROMPT": "black and white panda plush toy",
    },
    # 13) Change sunglasses lenses to purple gradient
    {
        "name": "glasses_purple_lens",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Change the sunglasses lenses to a purple gradient tint",
        "SEG_PROMPT": "dark lenses of the sunglasses",
        "TARGET_PROMPT": "semi-transparent purple gradient lenses",
        "STYLE_SOURCE_PROMPT": "dark lenses of the sunglasses",
        "STYLE_TARGET_PROMPT": "semi-transparent purple gradient lenses",
    },
    # 14) Change the remote controller body to matte black
    {
        "name": "remote_black",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Change the remote controller body to sleek matte black",
        "SEG_PROMPT": "white body of the remote controller",
        "TARGET_PROMPT": "sleek matte black body of the remote controller",
        "STYLE_SOURCE_PROMPT": "white body of the remote controller",
        "STYLE_TARGET_PROMPT": "sleek matte black body of the remote controller",
    },
    # 15) Make the perfume bottle look like frosted glass
    {
        "name": "perfume_frosted",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Change the perfume bottle to frosted glass material",
        "SEG_PROMPT": "clear glass perfume bottle",
        "TARGET_PROMPT": "semi-transparent frosted glass perfume bottle",
        "STYLE_SOURCE_PROMPT": "clear glass perfume bottle",
        "STYLE_TARGET_PROMPT": "semi-transparent frosted glass perfume bottle",
    },
    # 16) Change the JBL logo to a music note icon
    {
        "name": "speaker_logo_note",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Replace the logo on the speaker with a musical note icon",
        "SEG_PROMPT": "JBL logo on the speaker",
        "TARGET_PROMPT": "embossed musical note icon on the speaker",
        "STYLE_SOURCE_PROMPT": "JBL logo on the speaker",
        "STYLE_TARGET_PROMPT": "embossed musical note icon on the speaker",
    },
    # 17) Add a red bowtie to the plush toy
    {
        "name": "plush_bowtie",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Add a small red bowtie to the plush toy",
        "SEG_PROMPT": "neck area of the plush toy",
        "TARGET_PROMPT": "a small red silk bowtie on the plush toy",
        "STYLE_SOURCE_PROMPT": "neck area of the plush toy",
        "STYLE_TARGET_PROMPT": "a small red silk bowtie on the plush toy",
    },
    # 18) Change the speaker fabric to camouflage pattern
    {
        "name": "speaker_camouflage",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Change the speaker's fabric to a camouflage pattern",
        "SEG_PROMPT": "grey fabric of the speaker",
        "TARGET_PROMPT": "green and brown camouflage fabric",
        "STYLE_SOURCE_PROMPT": "grey fabric of the speaker",
        "STYLE_TARGET_PROMPT": "green and brown camouflage fabric",
    },
    # 19) Make the sunglasses look like cyberpunk visors
    {
        "name": "glasses_cyberpunk",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Change the sunglasses into glowing cyberpunk visors",
        "SEG_PROMPT": "black sunglasses",
        "TARGET_PROMPT": "futuristic glowing cyberpunk visor glasses",
        "STYLE_SOURCE_PROMPT": "black sunglasses",
        "STYLE_TARGET_PROMPT": "futuristic glowing cyberpunk visor glasses",
    },
    # 20) Change the liquid inside the bottle to glowing lava
    {
        "name": "perfume_lava",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Make the liquid inside the bottle glow like molten lava",
        "SEG_PROMPT": "yellow liquid inside the perfume bottle",
        "TARGET_PROMPT": "glowing orange and red molten lava liquid",
        "STYLE_SOURCE_PROMPT": "yellow liquid inside the perfume bottle",
        "STYLE_TARGET_PROMPT": "glowing orange and red molten lava liquid",
    },
   ## room_extended
    # 11) Change the dinosaur figure's color to blue
    {
        "name": "dino_blue",
        "DATA_NAME": "room",
        "PROMPT": "Change the dinosaur figure's color to blue",
        "SEG_PROMPT": "brown dinosaur figure",
        "TARGET_PROMPT": "vibrant blue dinosaur figure",
        "STYLE_SOURCE_PROMPT": "brown dinosaur figure",
        "STYLE_TARGET_PROMPT": "vibrant blue dinosaur figure",
    },
    # 12) Change the rabbit figure to a rough stone texture
    {
        "name": "rabbit_stone",
        "DATA_NAME": "room",
        "PROMPT": "Change the rabbit figure to a rough stone texture",
        "SEG_PROMPT": "grey rabbit figure",
        "TARGET_PROMPT": "rough stone carved rabbit figure",
        "STYLE_SOURCE_PROMPT": "grey rabbit figure",
        "STYLE_TARGET_PROMPT": "rough stone carved rabbit figure",
    },
    # 13) Replace the baseball with a small potted succulent plant
    {
        "name": "ball_succulent",
        "DATA_NAME": "room",
        "PROMPT": "Replace the baseball with a small potted succulent plant",
        "SEG_PROMPT": "white baseball",
        "TARGET_PROMPT": "small potted succulent plant",
        "STYLE_SOURCE_PROMPT": "white baseball",
        "STYLE_TARGET_PROMPT": "small potted succulent plant",
    },
    # 14) Add a tiny, delicate flower behind the rabbit figure's ear
    {
        "name": "rabbit_flower",
        "DATA_NAME": "room",
        "PROMPT": "Add a tiny, delicate flower behind the rabbit figure's ear",
        "SEG_PROMPT": "head of the grey rabbit figure",
        "TARGET_PROMPT": "grey rabbit figure wearing a tiny flower behind its ear",
        "STYLE_SOURCE_PROMPT": "head of the grey rabbit figure",
        "STYLE_TARGET_PROMPT": "grey rabbit figure wearing a tiny flower behind its ear",
    },
    # 15) Make the dinosaur figure's eyes glow with a red light
    {
        "name": "dino_eyes_glow",
        "DATA_NAME": "room",
        "PROMPT": "Make the dinosaur figure's eyes glow with a red light",
        "SEG_PROMPT": "eyes of the brown dinosaur figure",
        "TARGET_PROMPT": "brown dinosaur figure with glowing red eyes",
        "STYLE_SOURCE_PROMPT": "eyes of the brown dinosaur figure",
        "STYLE_TARGET_PROMPT": "brown dinosaur figure with glowing red eyes",
    },
    # 16) Change the chicken's red wattle and comb to polished wood
    {
        "name": "chicken_parts_wood",
        "DATA_NAME": "room",
        "PROMPT": "Change the chicken's red wattle and comb to polished wood",
        "SEG_PROMPT": "red wattle and comb of the rubber chicken",
        "TARGET_PROMPT": "red wattle and comb of the rubber chicken, now made of polished wood",
        "STYLE_SOURCE_PROMPT": "red wattle and comb of the rubber chicken",
        "STYLE_TARGET_PROMPT": "red wattle and comb of the rubber chicken, now made of polished wood",
    },
    # 17) Fill the empty space in the basket with colorful glass marbles
    {
        "name": "basket_marbles",
        "DATA_NAME": "room",
        "PROMPT": "Fill the empty space in the basket with colorful glass marbles",
        "SEG_PROMPT": "inside of the woven basket",
        "TARGET_PROMPT": "woven basket filled with colorful glass marbles",
        "STYLE_SOURCE_PROMPT": "inside of the woven basket",
        "STYLE_TARGET_PROMPT": "woven basket filled with colorful glass marbles",
    },
    # 18) Replace the rubber chicken with a small, stylized decorative skull
    {
        "name": "chicken_skull",
        "DATA_NAME": "room",
        "PROMPT": "Replace the rubber chicken with a small, stylized, decorative skull",
        "SEG_PROMPT": "yellow rubber chicken",
        "TARGET_PROMPT": "small, stylized, decorative skull",
        "STYLE_SOURCE_PROMPT": "yellow rubber chicken",
        "STYLE_TARGET_PROMPT": "small, stylized, decorative skull",
    },
    # 19) Make the baseball look like it's crudely carved from wood
    {
        "name": "ball_carved",
        "DATA_NAME": "room",
        "PROMPT": "Make the baseball look like it's crudely carved from a single piece of wood",
        "SEG_PROMPT": "white baseball",
        "TARGET_PROMPT": "a ball crudely carved from wood to look like a baseball",
        "STYLE_SOURCE_PROMPT": "white baseball",
        "STYLE_TARGET_PROMPT": "a ball crudely carved from wood to look like a baseball",
    },
    # 20) Add a tiny magnifying glass resting on the floor next to the dinosaur figure
    {
        "name": "dino_magnifying_glass",
        "DATA_NAME": "room",
        "PROMPT": "Add a tiny magnifying glass resting on the floor next to the dinosaur figure",
        "SEG_PROMPT": "floor area next to the dinosaur",
        "TARGET_PROMPT": "floor area next to the dinosaur containing a tiny magnifying glass",
        "STYLE_SOURCE_PROMPT": "floor area next to the dinosaur",
        "STYLE_TARGET_PROMPT": "floor area next to the dinosaur containing a tiny magnifying glass",
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

