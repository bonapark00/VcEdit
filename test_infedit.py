import torch
import os
import cv2
import numpy as np
from PIL import Image
import argparse
import sys

# 프로젝트 경로 추가
sys.path.append('.')

from threestudio.systems.VcEdit import VcEdit
from threestudio.data.gs_load import GSLoadDataModule
from threestudio.utils.config import load_config
from threestudio.utils.misc import get_device
import threestudio


def test_infedit_only():
    """InfEdit만 실행하여 결과 이미지들을 저장하는 함수"""
    
    # 1. 설정 로드
    config_path = "configs/edit-inf.yaml"
    cfg = load_config(config_path, cli_args=[], n_gpus=1)
    
    # 2. 데이터 모듈 초기화
    dm = threestudio.find(cfg.data_type)(cfg.data)
    dm.setup()
    
    # 3. VcEdit 시스템 초기화 (training 설정 없이)
    system = threestudio.find(cfg.system_type)(cfg.system, resumed=False)
    
    # 4. 시스템 설정
    system.cfg.gs_source = "gs_data/trained_gs_models/bear/point_cloud.ply"
    system.cfg.seg_prompt = "bear"
    
    # 5. Gaussian 모델 로드
    system.gaussian.load_ply(system.cfg.gs_source)
    
    # 6. 카메라 정보 설정
    system.view_list = dm.train_dataset.n2n_view_index
    system.view_num = len(system.view_list)
    
    # 7. 원본 프레임 렌더링
    print("Rendering original frames...")
    system.origin_frames = system.render_all_view(cache_name="origin_render")
    
    # 8. InfEdit Guidance 초기화
    if hasattr(system, 'guidance'):
        print("InfEdit Guidance already initialized")
    else:
        print("Initializing InfEdit Guidance...")
        system.guidance = threestudio.find(cfg.system.guidance_type)(cfg.system.guidance)
    
    # 9. 프롬프트 설정
    system.guidance.src_prompt = "bear"
    system.guidance.tgt_prompt = "brown bear"
    
    # 10. 결과 저장 디렉토리 생성
    output_dir = "infedit_test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 11. 각 뷰별로 InfEdit 실행
    print("Running InfEdit for each view...")
    for view_idx in system.view_list:
        print(f"Processing view {view_idx}...")
        
        # 현재 뷰의 원본 이미지
        original_image = system.origin_frames[view_idx][0]  # [H, W, C]
        
        # InfEdit 실행
        curr_frames = {view_idx: original_image[None]}  # [1, H, W, C]
        
        try:
            result = system.guidance(
                curr_frames,
                copy.deepcopy(system.gaussian),
                [dm.train_dataset.scene.cameras[view_idx]],  # 단일 카메라
                system.pipe if hasattr(system, 'pipe') else None,
                [view_idx],  # 단일 뷰
                calling_idx=0,  # 첫 번째 편집
            )
            
            # 편집된 이미지 추출
            edited_image = result["edit_images"][0]  # [H, W, C]
            
            # 이미지 저장
            # 원본 이미지
            original_save_path = os.path.join(output_dir, f"view_{view_idx:04d}_original.png")
            original_img_np = (original_image.cpu().numpy() * 255).astype(np.uint8)
            original_img_np = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(original_save_path, original_img_np)
            
            # 편집된 이미지
            edited_save_path = os.path.join(output_dir, f"view_{view_idx:04d}_edited.png")
            edited_img_np = (edited_image.cpu().numpy() * 255).astype(np.uint8)
            edited_img_np = cv2.cvtColor(edited_img_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(edited_save_path, edited_img_np)
            
            print(f"Saved view {view_idx}: {original_save_path}, {edited_save_path}")
            
        except Exception as e:
            print(f"Error processing view {view_idx}: {e}")
            continue
    
    # 12. 비교 이미지 생성 (원본 vs 편집)
    print("Creating comparison images...")
    for view_idx in system.view_list:
        original_path = os.path.join(output_dir, f"view_{view_idx:04d}_original.png")
        edited_path = os.path.join(output_dir, f"view_{view_idx:04d}_edited.png")
        
        if os.path.exists(original_path) and os.path.exists(edited_path):
            # 이미지 로드
            original_img = cv2.imread(original_path)
            edited_img = cv2.imread(edited_path)
            
            # 수평으로 연결
            comparison = np.hstack([original_img, edited_img])
            
            # 제목 추가
            comparison_with_text = np.zeros((comparison.shape[0] + 50, comparison.shape[1], 3), dtype=np.uint8)
            comparison_with_text[50:] = comparison
            
            # 텍스트 추가
            cv2.putText(comparison_with_text, f"View {view_idx}: Original (Left) vs Edited (Right)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 비교 이미지 저장
            comparison_path = os.path.join(output_dir, f"view_{view_idx:04d}_comparison.png")
            cv2.imwrite(comparison_path, comparison_with_text)
            print(f"Saved comparison: {comparison_path}")
    
    print(f"\nAll results saved to: {output_dir}")
    print("Files created:")
    print("- view_XXXX_original.png: 원본 이미지")
    print("- view_XXXX_edited.png: 편집된 이미지") 
    print("- view_XXXX_comparison.png: 비교 이미지")


if __name__ == "__main__":
    test_infedit_only()

