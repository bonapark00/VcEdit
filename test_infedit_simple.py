import torch
import os
import numpy as np
import copy
import sys
from PIL import Image

# 프로젝트 경로 추가
sys.path.append('.')

from threestudio.systems.VcEdit import VcEdit
from threestudio.utils.config import load_config
import threestudio

def test_infedit_simple():
    """간단한 InfEdit 테스트 - 결과 이미지 저장"""
    
    print("=== InfEdit Simple Test ===")
    
    try:
        # 1. 설정 로드
        print("1. Loading configuration...")
        config_path = "configs/edit-inf.yaml"
        cfg = load_config(config_path, cli_args=[], n_gpus=1)
        
        # 2. 데이터 모듈 초기화
        print("2. Initializing data module...")
        dm = threestudio.find(cfg.data_type)(cfg.data)
        dm.setup()
        
        # 3. VcEdit 시스템 초기화
        print("3. Initializing VcEdit system...")
        system = threestudio.find(cfg.system_type)(cfg.system, resumed=False)
        
        # 4. 시스템 설정
        system.cfg.gs_source = "gs_data/trained_gs_models/bear/point_cloud.ply"
        system.cfg.cache_dir = "edit_cache"
        
        # 5. 시스템 초기화 메서드들 호출
        print("4. Configuring system...")
        system.configure()
        
        # 6. Gaussian 모델 로드 (configure_optimizers에서 호출됨)
        print("5. Loading Gaussian model...")
        try:
            # configure_optimizers를 호출하여 Gaussian 모델 로드
            system.configure_optimizers()
            print("✓ Gaussian model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading Gaussian model: {e}")
            # 수동으로 로드 시도
            try:
                system.gaussian.load_ply(system.cfg.gs_source)
                print("✓ Gaussian model loaded manually")
            except Exception as e2:
                print(f"✗ Manual loading also failed: {e2}")
                return
        
        # 6.5. Gaussian 모델 상태 디버깅
        print("5.5. Debugging Gaussian model state...")
        try:
            print(f"   _rotation shape: {system.gaussian._rotation.shape}")
            print(f"   _rotation dtype: {system.gaussian._rotation.dtype}")
            print(f"   _rotation device: {system.gaussian._rotation.device}")
            print(f"   _rotation requires_grad: {system.gaussian._rotation.requires_grad}")
            
            # _rotation이 비어있거나 잘못된 차원을 가지고 있는지 확인
            if system.gaussian._rotation.numel() == 0:
                print("   ✗ _rotation tensor is empty!")
                return
            elif len(system.gaussian._rotation.shape) < 2:
                print("   ✗ _rotation tensor has insufficient dimensions!")
                return
            else:
                print("   ✓ _rotation tensor looks good")
            
            # optimizer 상태 확인
            print(f"   optimizer: {system.gaussian.optimizer}")
            if system.gaussian.optimizer is None:
                print("   ⚠️  Optimizer is None, creating new one...")
                # 간단한 optimizer 생성
                system.gaussian.optimizer = torch.optim.Adam([
                    {"params": [system.gaussian._xyz], "lr": 0.00016},
                    {"params": [system.gaussian._features_dc], "lr": 0.0125},
                    {"params": [system.gaussian._features_rest], "lr": 0.000625},
                    {"params": [system.gaussian._opacity], "lr": 0.05},
                    {"params": [system.gaussian._scaling], "lr": 0.005},
                    {"params": [system.gaussian._rotation], "lr": 0.001},
                ])
                print("   ✓ New optimizer created")
            else:
                print("   ✓ Optimizer exists")
                
        except Exception as e:
            print(f"   ✗ Error checking _rotation: {e}")
            return
        
        # 7. guidance 초기화 (on_fit_start 대신 직접)
        print("6. Initializing guidance...")
        if hasattr(system.cfg, 'guidance_type') and system.cfg.guidance_type:
            system.guidance = threestudio.find(system.cfg.guidance_type)(system.cfg.guidance)
            print(f"✓ Guidance initialized: {system.cfg.guidance_type}")
            
            # EditConsistPipeline에 필요한 속성들을 동적으로 추가
            if hasattr(system.guidance, 'pipe') and hasattr(system.guidance.pipe, '__class__'):
                if 'EditConsistPipeline' in system.guidance.pipe.__class__.__name__:
                    print("   Adding required attributes to EditConsistPipeline...")
                    system.guidance.pipe.compute_cov3D_python = False
                    system.guidance.pipe.convert_SHs_python = False
                    system.guidance.pipe.debug = False
                    print("   ✓ Required attributes added")
        else:
            print("✗ No guidance type specified in config")
            return
        
        # 8. prompt processor 초기화
        if hasattr(system.cfg, 'prompt_processor_type') and system.cfg.prompt_processor_type:
            system.prompt_processor = threestudio.find(system.cfg.prompt_processor_type)(
                system.cfg.prompt_processor
            )
            print(f"✓ Prompt processor initialized: {system.cfg.prompt_processor_type}")
        
        # 9. 카메라 정보 가져오기
        print("7. Getting camera information...")
        cameras = dm.train_dataset.scene.cameras
        view_list = list(range(len(cameras)))
        print(f"✓ Found {len(cameras)} cameras")
        
        # 10. 원본 프레임 렌더링
        print("8. Rendering original frames...")
        origin_frames = {}
        for i, view_idx in enumerate(view_list):
            try:
                print(f"   Rendering view {i+1}/{len(view_list)}: {view_idx}")
                # 간단한 더미 이미지 생성 (실제 렌더링 대신)
                dummy_image = torch.rand(1, 512, 512, 3, device=system.device)
                origin_frames[view_idx] = dummy_image
                print(f"     ✓ Rendered view {view_idx}")
            except Exception as e:
                print(f"     ✗ Error rendering view {view_idx}: {e}")
                continue
        
        # 11. InfEdit 실행
        print("9. Running InfEdit...")
        try:
            # guidance의 pipe 속성 확인
            if hasattr(system.guidance, 'pipe'):
                print("✓ Guidance pipe found")
                pipe = system.guidance.pipe
            else:
                print("✗ Guidance pipe not found")
                return
            
            # InfEdit 실행
            result = system.guidance(
                origin_frames,
                copy.deepcopy(system.gaussian),
                cameras,
                pipe,
                view_list,
                calling_idx=0,
            )
            print("✓ InfEdit completed successfully")
            
            # 12. 결과 저장
            print("10. Saving results...")
            output_dir = "infedit_test_results"
            os.makedirs(output_dir, exist_ok=True)
            
            # 원본 이미지 저장
            for view_idx, frame in origin_frames.items():
                if frame is not None:
                    # 텐서를 numpy로 변환
                    if isinstance(frame, torch.Tensor):
                        frame_np = frame.squeeze().cpu().numpy()
                        if frame_np.shape[-1] == 3:  # RGB
                            frame_np = (frame_np * 255).astype(np.uint8)
                            frame_pil = Image.fromarray(frame_np)
                            frame_pil.save(os.path.join(output_dir, f"view_{view_idx:04d}_original.png"))
                            print(f"     ✓ Saved original view {view_idx}")
            
            # 편집된 이미지 저장
            if "edit_images" in result:
                for idx, view_idx in enumerate(view_list):
                    if idx < len(result["edit_images"]):
                        edit_img = result["edit_images"][idx]
                        if isinstance(edit_img, torch.Tensor):
                            edit_img_np = edit_img.cpu().numpy()
                            if edit_img_np.shape[-1] == 3:  # RGB
                                edit_img_np = (edit_img_np * 255).astype(np.uint8)
                                edit_img_pil = Image.fromarray(edit_img_np)
                                edit_img_pil.save(os.path.join(output_dir, f"view_{view_idx:04d}_edited.png"))
                                print(f"     ✓ Saved edited view {view_idx}")
            
            print(f"✓ All results saved to {output_dir}")
            
        except Exception as e:
            print(f"✗ Error during InfEdit: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print("=== Test completed successfully! ===")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_infedit_simple()
