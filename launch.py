import argparse
import contextlib
import logging
import os
import sys
import wandb


class ColoredFilter(logging.Filter):
    """
    A logging filter to add color to certain log levels.
    """

    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    COLORS = {
        "WARNING": YELLOW,
        "INFO": GREEN,
        "DEBUG": BLUE,
        "CRITICAL": MAGENTA,
        "ERROR": RED,
    }

    RESET = "\x1b[0m"

    def __init__(self):
        super().__init__()

    def filter(self, record):
        if record.levelname in self.COLORS:
            color_start = self.COLORS[record.levelname]
            record.levelname = f"{color_start}[{record.levelname}]"
            record.msg = f"{record.msg}{self.RESET}"
        return True


def main(args, extras) -> None:
    # set CUDA_VISIBLE_DEVICES if needed, then import pytorch-lightning
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env_gpus_str = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    env_gpus = list(env_gpus_str.split(",")) if env_gpus_str else []
    selected_gpus = [0]

    # Always rely on CUDA_VISIBLE_DEVICES if specific GPU ID(s) are specified.
    # As far as Pytorch Lightning is concerned, we always use all available GPUs
    # (possibly filtered by CUDA_VISIBLE_DEVICES).
    devices = -1
    if len(env_gpus) > 0:
        # CUDA_VISIBLE_DEVICES was set already, e.g. within SLURM srun or higher-level script.
        n_gpus = len(env_gpus)
    else:
        selected_gpus = list(args.gpu.split(","))
        n_gpus = len(selected_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import pytorch_lightning as pl
    import torch
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    from pytorch_lightning.utilities.rank_zero import rank_zero_only

    if args.typecheck:
        from jaxtyping import install_import_hook

        install_import_hook("threestudio", "typeguard.typechecked")

    import threestudio
    from threestudio.systems.base import BaseSystem
    from threestudio.utils.callbacks import (
        CodeSnapshotCallback,
        ConfigSnapshotCallback,
        CustomProgressBar,
        ProgressCallback,
    )
    from threestudio.utils.config import ExperimentConfig, load_config
    from threestudio.utils.misc import get_rank
    from threestudio.utils.typing import Optional

    logger = logging.getLogger("pytorch_lightning")
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    for handler in logger.handlers:
        if handler.stream == sys.stderr:  # type: ignore
            if not args.gradio:
                handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
                handler.addFilter(ColoredFilter())
            else:
                handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    # parse YAML config to OmegaConf
    cfg: ExperimentConfig
    cfg = load_config(args.config, cli_args=extras, n_gpus=n_gpus)

    # set a different seed for each device
    pl.seed_everything(cfg.seed + get_rank(), workers=True)

    dm = threestudio.find(cfg.data_type)(cfg.data)
    system: BaseSystem = threestudio.find(cfg.system_type)(
        cfg.system, resumed=cfg.resume is not None
    )
    system.set_save_dir(os.path.join(cfg.trial_dir, "save"))
    # Expose run directory for timing logs written inside pipelines
    os.environ["VCEDIT_TIMING_DIR"] = cfg.trial_dir
    if system.cfg.loggers.wandb.enable:
        try:
            wandb.config.update({"cfg": cfg})
            # wandb.run.log_code(".")
        except Exception as e:
            print(e)
            wandb.init(
                project=system.cfg.loggers.wandb.project,
                name=system.cfg.loggers.wandb.name,
                group=system.cfg.loggers.wandb.get("group", None),
            )
            wandb.config.update({"cfg": cfg})
            # wandb.run.log_code(".")
    if args.gradio:
        fh = logging.FileHandler(os.path.join(cfg.trial_dir, "logs"))
        fh.setLevel(logging.INFO)
        if args.verbose:
            fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(fh)

    callbacks = []
    if args.train:
        callbacks += [
            ModelCheckpoint(
                dirpath=os.path.join(cfg.trial_dir, "ckpts"), **cfg.checkpoint
            ),
            LearningRateMonitor(logging_interval="step"),
            # CodeSnapshotCallback(
            #     os.path.join(cfg.trial_dir, "code"), use_version=False
            # ),
            ConfigSnapshotCallback(
                args.config,
                cfg,
                os.path.join(cfg.trial_dir, "configs"),
                use_version=False,
            ),
        ]
        if args.gradio:
            callbacks += [
                ProgressCallback(save_path=os.path.join(cfg.trial_dir, "progress"))
            ]
        else:
            callbacks += [CustomProgressBar(refresh_rate=1)]

    def write_to_text(file, lines):
        with open(file, "w") as f:
            for line in lines:
                f.write(line + "\n")

    loggers = []
    if args.train:
        # make tensorboard logging dir to suppress warning
        rank_zero_only(
            lambda: os.makedirs(os.path.join(cfg.trial_dir, "tb_logs"), exist_ok=True)
        )()
        loggers += [
            TensorBoardLogger(cfg.trial_dir, name="tb_logs"),
            CSVLogger(cfg.trial_dir, name="csv_logs"),
        ] + system.get_loggers()
        rank_zero_only(
            lambda: write_to_text(
                os.path.join(cfg.trial_dir, "cmd.txt"),
                ["python " + " ".join(sys.argv), str(args)],
            )
        )()

    trainer = Trainer(
        callbacks=callbacks,
        logger=loggers,
        inference_mode=False,
        accelerator="gpu",
        devices=devices,
        **cfg.trainer,
    )

    # Helpers to write overall timing summaries at the launcher level
    def _write_overall_timing_summary(trial_dir: str, fit_wall: float, test_wall: float) -> None:
        import time
        import os
        summary_path = os.path.join(trial_dir, "overall_timing_summary.txt")

        # Aggregate training_timing.csv if exists
        train_csv = os.path.join(trial_dir, "training_timing.csv")
        training_totals = {
            "total": 0.0,
            "render": 0.0,
            "guidance": 0.0,
            "loss": 0.0,
            "other": 0.0,
            "steps": 0,
        }
        if os.path.exists(train_csv):
            with open(train_csv, "r") as f:
                header = True
                for line in f:
                    if header:
                        header = False
                        continue
                    parts = line.strip().split(",")
                    if len(parts) < 10:
                        continue
                    training_totals["steps"] += 1
                    training_totals["total"] += float(parts[1])
                    training_totals["render"] += float(parts[2])
                    training_totals["guidance"] += float(parts[3])
                    training_totals["loss"] += float(parts[4])
                    training_totals["other"] += float(parts[5])

        # Aggregate inference summary if exists (generated by pipeline)
        timings_summary = os.path.join(trial_dir, "timings_summary.txt")
        inference_lines = []
        if os.path.exists(timings_summary):
            with open(timings_summary, "r") as f:
                inference_lines = [l.rstrip("\n") for l in f]

        with open(summary_path, "w") as f:
            f.write("Overall Timing Summary\n")
            f.write(f"wall_time.fit={fit_wall:.6f}\n")
            if test_wall >= 0:
                f.write(f"wall_time.test={test_wall:.6f}\n")

            # Training aggregates
            denom = max(training_totals["total"], 1e-8)
            f.write("\n[training aggregates]\n")
            f.write(f"steps={training_totals['steps']}\n")
            f.write(f"total={training_totals['total']:.6f}\n")
            f.write(
                f"render={training_totals['render']:.6f}, ratio={training_totals['render']/denom:.6f}\n"
            )
            f.write(
                f"guidance={training_totals['guidance']:.6f}, ratio={training_totals['guidance']/denom:.6f}\n"
            )
            f.write(
                f"loss={training_totals['loss']:.6f}, ratio={training_totals['loss']/denom:.6f}\n"
            )
            f.write(
                f"other={training_totals['other']:.6f}, ratio={training_totals['other']/denom:.6f}\n"
            )

            # Inference (pipeline) breakdown passthrough
            if inference_lines:
                f.write("\n[inference breakdown from timings_summary.txt]\n")
                for line in inference_lines:
                    f.write(line + "\n")

    def set_system_status(system: BaseSystem, ckpt_path: Optional[str]):
        if ckpt_path is None:
            return
        ckpt = torch.load(ckpt_path, map_location="cpu")
        system.set_resume_status(ckpt["epoch"], ckpt["global_step"])

    if args.train:
        import time
        fit_t0 = time.time()
        trainer.fit(system, datamodule=dm, ckpt_path=cfg.resume)
        fit_wall = time.time() - fit_t0

        test_wall = -1.0
        try:
            test_t0 = time.time()
            trainer.test(system, datamodule=dm)
            test_wall = time.time() - test_t0
        except Exception:
            pass

        # Write consolidated timing summary
        _write_overall_timing_summary(cfg.trial_dir, fit_wall, test_wall)
        if args.gradio:
            # also export assets if in gradio mode
            trainer.predict(system, datamodule=dm)
    elif args.validate:
        # manually set epoch and global_step as they cannot be automatically resumed
        set_system_status(system, cfg.resume)
        trainer.validate(system, datamodule=dm, ckpt_path=cfg.resume)
    elif args.test:
        # manually set epoch and global_step as they cannot be automatically resumed
        set_system_status(system, cfg.resume)
        trainer.test(system, datamodule=dm, ckpt_path=cfg.resume)
    elif args.export:
        set_system_status(system, cfg.resume)
        trainer.predict(system, datamodule=dm, ckpt_path=cfg.resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU(s) to be used. 0 means use the 1st available GPU. "
        "1,2 means use the 2nd and 3rd available GPU. "
        "If CUDA_VISIBLE_DEVICES is set before calling `launch.py`, "
        "this argument is ignored and all available GPUs are always used.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--validate", action="store_true")
    group.add_argument("--test", action="store_true")
    group.add_argument("--export", action="store_true")

    parser.add_argument(
        "--gradio", action="store_true", help="if true, run in gradio mode"
    )

    parser.add_argument(
        "--verbose", action="store_true", help="if true, set logging level to DEBUG"
    )

    parser.add_argument(
        "--typecheck",
        action="store_true",
        help="whether to enable dynamic type checking",
    )

    args, extras = parser.parse_known_args()

    if args.gradio:
        # FIXME: no effect, stdout is not captured
        with contextlib.redirect_stdout(sys.stderr):
            main(args, extras)
    else:
        main(args, extras)
