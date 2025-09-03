#!/usr/bin/env python3
"""
간단한 InfEdit 디버깅 테스트
"""

import torch
import os
import sys

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from threestudio.utils.config import load_config
import threestudio

def test_simple():
    print("🔍 Simple InfEdit Test 시작")
    
    # 1. 설정 파일 로드
    print("1. Loading config...")
    config = load_config("configs/edit-inf.yaml")
    
    # 2. 데이터 모듈 초기화
    print("2. Initializing data module...")
    dm = threestudio.find(config.data_type)(config.data)
    dm.setup()
    
    # 3. 시스템 초기화
    print("3. Initializing system...")
    system = threestudio.find(config.system_type)(config.system)
    system.configure()
    
    # 4. Gaussian 모델 설정
    print("4. Setting up Gaussian model...")
    system.gaussian.load_ply(system.cfg.gs_source)
    system.configure_optimizers()
    
    # 5. Guidance 초기화
    print("5. Initializing guidance...")
    if hasattr(system.cfg, 'guidance_type') and system.cfg.guidance_type:
        system.guidance = threestudio.find(system.cfg.guidance_type)(system.cfg.guidance)
        
        # EditConsistPipeline에 필요한 속성들을 동적으로 추가
        if hasattr(system.guidance, 'pipe') and hasattr(system.guidance.pipe, '__class__'):
            if 'EditConsistPipeline' in system.guidance.pipe.__class__.__name__:
                system.guidance.pipe.compute_cov3D_python = False
                system.guidance.pipe.convert_SHs_python = False
                system.guidance.pipe.debug = False
    
    # 6. 카메라 정보 가져오기
    print("6. Getting camera information...")
    cameras = dm.train_dataset.scene.cameras
    num_views = min(4, len(cameras))
    
    # 7. 더미 프레임 생성
    print("7. Creating dummy frames...")
    origin_frames = torch.randn(num_views, 3, 512, 512, device="cuda")
    
    # 8. Guidance 호출
    print("8. Calling guidance...")
    try:
        result = system.guidance(
            origin_frames=origin_frames,
            cameras=cameras[:num_views],
            prompt_processor=system.prompt_processor,
            **system.guidance_kwargs
        )
        
        if isinstance(result, dict) and result.get("debug_stopped"):
            print("🎉 Debug stop successful!")
            print(f"view_img_preds shape: {result['view_img_preds'].shape}")
            
            # 결과를 파일로 저장
            debug_dir = "debug_intermediate_results"
            if os.path.exists(debug_dir):
                files = os.listdir(debug_dir)
                print(f"Files in {debug_dir}: {files}")
        else:
            print("⚠️  Expected debug stop but got different result")
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ Test completed!")

if __name__ == "__main__":
    test_simple()
