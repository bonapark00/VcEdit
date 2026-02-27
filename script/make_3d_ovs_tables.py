import os
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
from PIL import Image


# 리포 루트에서 실행한다고 가정
BASE_DIR = Path("outputs/edit-inf/3d-ovs")
SUMMARY_DIR = BASE_DIR / "summary"
IMAGE_DPI = 300  # summary 이미지 저장 해상도

# 씬별로 사용할 뷰 인덱스
VIEW_INDICES: Dict[str, List[int]] = {
    "room": [5, 20, 30],
    "covered_desk": [1, 10, 20],
    "blue_sofa": [8, 23, 0],
}


def concat_images_horiz(images: List[Any]):
    """원본 비율을 유지한 채로 이미지를 좌우로 딱 붙여 하나로 합친다."""
    imgs = [img for img in images if img is not None]
    if not imgs:
        return None

    # 기준 높이: 가장 큰 높이
    max_h = max(im.height for im in imgs)
    resized = []
    for im in imgs:
        if im.height != max_h:
            new_w = int(im.width * (max_h / im.height))
            resized.append(im.resize((new_w, max_h), Image.LANCZOS))
        else:
            resized.append(im)

    total_w = sum(im.width for im in resized)
    out = Image.new("RGB", (total_w, max_h))
    x = 0
    for im in resized:
        out.paste(im, (x, 0))
        x += im.width
    return out


def parse_train_latency(latency_path: Path) -> str:
    """overall_timing_summary.txt에서 train 총 latency (초)를 문자열로 파싱.

    예시 라인:
    "Latency Summary (seconds) - Total Time: 1743.763s"
    """
    text = latency_path.read_text()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Latency Summary (seconds) - Total Time:"):
            parts = line.split()
            # 마지막 토큰이 "1743.763s" 형태라고 가정
            if parts:
                val = parts[-1]
                if val.endswith("s"):
                    val = val[:-1]
                try:
                    secs = float(val)
                    return f"{secs:.1f}s"
                except ValueError:
                    pass
    return "N/A"


def parse_clip_metrics(eval_path: Path) -> str:
    """eval_clip.txt에서 4개 CLIP metric을 파싱해 한 셀에 넣을 문자열 생성."""
    text = eval_path.read_text()
    metrics_order = [
        "CLIP directional consistency:",
        "CLIP_F (scaled):",
        "CLIP Score:",
        "CLIP directional similarity:",
    ]
    values: Dict[str, Any] = {}

    for line in text.splitlines():
        line = line.strip()
        for key in metrics_order:
            if line.startswith(key):
                try:
                    val = float(line.split(":")[1].strip())
                except (IndexError, ValueError):
                    val = float("nan")
                values[key] = val

    lines = []
    for key in metrics_order:
        if key in values:
            short = key.replace("CLIP ", "").replace(" (scaled)", "")
            short = short.replace("directional ", "dir_").replace("Score", "score")
            lines.append(f"{short} {values[key]:.4f}")
    return "\n".join(lines) if lines else "N/A"


def collect_rows_for_scene(scene_name: str, view_indices: List[int]):
    """하나의 scene(room/covered_desk/blue_sofa)에 대해 row 데이터 수집."""
    scene_dir = BASE_DIR / scene_name
    if not scene_dir.exists():
        print(f"[WARN] {scene_dir} not found, skip.")
        return []

    rows = []
    # 구조: scene_dir / task_name / run_dir (e.g. Change_the_...@timestamp)
    for task_dir in sorted(scene_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        for run_dir in sorted(task_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            save_dir = run_dir / "save" / "it800-test"
            latency_path = run_dir / "overall_timing_summary.txt"
            eval_path = run_dir / "eval_clip.txt"

            if not (save_dir.exists() and latency_path.exists() and eval_path.exists()):
                continue

            # prompt: run_dir.name의 "@timestamp" 앞, "_"를 space로 변환
            run_name = run_dir.name
            prompt_raw = run_name.split("@")[0]
            prompt = prompt_raw.replace("_", " ")

            # 이미지 로드 (존재하지 않으면 None)
            images: List[Any] = []
            for idx in view_indices:
                img_path = save_dir / f"{idx}.png"
                if img_path.exists():
                    try:
                        img = Image.open(img_path)
                    except Exception:
                        img = None
                else:
                    img = None
                images.append(img)

            latency = parse_train_latency(latency_path)
            clip_text = parse_clip_metrics(eval_path)

            rows.append(
                {
                    "prompt": prompt,
                    "images": images,
                    "latency": latency,
                    "clip_text": clip_text,
                    "run_dir": run_dir,
                }
            )

    return rows


def make_scene_figure(scene_name: str, rows: List[Dict[str, Any]], view_indices: List[int]):
    """rows 데이터를 이용해 하나의 scene 요약 이미지를 생성."""
    if not rows:
        print(f"[INFO] No rows for scene {scene_name}, skip figure.")
        return

    # summary 이미지들을 모아둘 디렉토리 생성
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    n_rows = len(rows)
    # 외부 열: 0=prompt, 1=images(3개 묶음), 2=latency, 3=clip
    n_outer_cols = 4

    # 행 수, 열 수에 따라 적당히 크기 조정 (이미지 크게)
    fig_width = 3.5 * 6  # 기존 6컬럼 기준 너비 유지
    fig_height = 2.5 * n_rows
    fig = plt.figure(figsize=(fig_width, fig_height))

    # 외부 GridSpec: prompt | images(3묶음) | latency | clip
    outer_gs = fig.add_gridspec(
        n_rows,
        n_outer_cols,
        wspace=0.15,
        hspace=0.25,
        width_ratios=[1.2, 7.5, 1.4, 1.6],
        left=0.02, right=0.98, top=0.94, bottom=0.02,
    )

    for r_idx, row in enumerate(rows):
        # --- 0열: prompt ---
        ax_prompt = fig.add_subplot(outer_gs[r_idx, 0])
        ax_prompt.set_xticks([])
        ax_prompt.set_yticks([])
        ax_prompt.set_frame_on(False)
        ax_prompt.text(0.5, 0.5, row["prompt"], ha="center", va="center", wrap=True, fontsize=10)

        # --- 1열: 이미지 3개를 "완전 무간격"으로 하나의 이미지로 합쳐서 배치 ---
        ax_imgs = fig.add_subplot(outer_gs[r_idx, 1])
        ax_imgs.set_axis_off()

        concat = concat_images_horiz(row["images"])
        if concat is not None:
            # 개별 run에 대해 3-view 합친 이미지를 summary_프롬프트이름으로 저장
            run_dir = row.get("run_dir")
            if run_dir is not None:
                run_name = run_dir.name
                prompt_raw = run_name.split("@")[0]  # 언더스코어가 포함된 원래 프롬프트 토큰
                out_single = SUMMARY_DIR / f"summary_{prompt_raw}.png"
                try:
                    concat.save(out_single)
                except Exception:
                    pass

            ax_imgs.imshow(concat)
        else:
            ax_imgs.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=10)

        # --- 2열: latency ---
        ax_lat = fig.add_subplot(outer_gs[r_idx, 2])
        ax_lat.set_xticks([])
        ax_lat.set_yticks([])
        ax_lat.set_frame_on(False)
        ax_lat.text(0.5, 0.5, f"train: {row['latency']}", ha="center", va="center", fontsize=10)

        # --- 3열: CLIP metrics ---
        ax_clip = fig.add_subplot(outer_gs[r_idx, 3])
        ax_clip.set_xticks([])
        ax_clip.set_yticks([])
        ax_clip.set_frame_on(False)
        ax_clip.text(0.5, 0.5, row["clip_text"], ha="center", va="center", fontsize=9)

    fig.suptitle(f"{scene_name}", fontsize=14)

    out_path = SUMMARY_DIR / f"summary_{scene_name}.png"
    fig.savefig(out_path, dpi=IMAGE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {out_path}")


def main():
    # 한 장당 최대 row 수 (예: 5개씩 잘라서 2장으로)
    max_rows_per_fig = 5

    for scene_name, view_indices in VIEW_INDICES.items():
        rows = collect_rows_for_scene(scene_name, view_indices)
        print(f"[INFO] {scene_name}: {len(rows)} rows")

        if not rows:
            continue

        # rows를 max_rows_per_fig 단위로 나눠 여러 이미지로 저장
        for start in range(0, len(rows), max_rows_per_fig):
            chunk = rows[start : start + max_rows_per_fig]
            part_idx = start // max_rows_per_fig + 1

            # 5개 이하 한 번만 그릴 땐 기존 이름 유지, 그 이상이면 _partN 붙이기
            if len(rows) <= max_rows_per_fig:
                scene_label = scene_name
            else:
                scene_label = f"{scene_name}_part{part_idx}"

            make_scene_figure(scene_label, chunk, view_indices)


if __name__ == "__main__":
    main()