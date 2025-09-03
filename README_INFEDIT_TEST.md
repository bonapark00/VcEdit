# InfEdit 테스트 스크립트 사용법

이 스크립트는 VcEdit 시스템에서 InfEdit만 실행하여 "bear"를 "brown bear"로 변환하는 결과를 테스트하고 저장합니다.

## 파일 설명

- **`test_infedit_simple.py`**: 간단한 InfEdit 테스트 스크립트
- **`test_infedit.py`**: 더 상세한 테스트 스크립트 (비교 이미지 포함)

## 실행 방법

### 1. 기본 실행
```bash
python test_infedit_simple.py
```

### 2. 상세 테스트 실행 (비교 이미지 포함)
```bash
python test_infedit.py
```

## 실행 과정

1. **설정 로드**: `configs/edit-inf.yaml` 설정 파일 로드
2. **데이터 초기화**: Gaussian Splatting 데이터 로드
3. **시스템 초기화**: VcEdit 시스템 및 InfEdit Guidance 초기화
4. **원본 렌더링**: 각 뷰의 원본 이미지 렌더링
5. **InfEdit 실행**: "bear" → "brown bear" 변환
6. **결과 저장**: 원본 및 편집된 이미지 저장

## 출력 파일

### `test_infedit_simple.py` 실행 시:
- `infedit_test_results/view_XXXX_original.png`: 원본 이미지
- `infedit_test_results/view_XXXX_edited.png`: 편집된 이미지

### `test_infedit.py` 실행 시 (추가):
- `infedit_test_results/view_XXXX_comparison.png`: 원본 vs 편집 비교 이미지

## 프롬프트 설정

- **Source prompt**: "bear" (원본)
- **Target prompt**: "brown bear" (변환 목표)

## 주의사항

1. **GPU 메모리**: InfEdit은 상당한 GPU 메모리를 사용합니다 (20-40GB 권장)
2. **데이터 경로**: `gs_data/trained_gs_models/bear/point_cloud.ply` 파일이 존재해야 합니다
3. **의존성**: OpenCV, PyTorch, threestudio 등이 설치되어 있어야 합니다

## 문제 해결

### GPU 메모리 부족 시:
- 배치 크기를 줄이거나
- 이미지 해상도를 낮추거나
- 더 작은 모델 사용

### 파일 경로 오류 시:
- `gs_data` 폴더가 올바른 위치에 있는지 확인
- 상대 경로가 올바른지 확인

## 예상 결과

성공적으로 실행되면:
- 각 뷰별로 원본 "bear" 이미지와 편집된 "brown bear" 이미지가 생성됩니다
- 이미지는 512x512 해상도로 저장됩니다
- RGB 형식으로 저장되며, OpenCV로 열 수 있습니다

## 로그 확인

실행 중 다음과 같은 로그를 확인할 수 있습니다:
```
=== InfEdit Simple Test ===
1. Loading configuration...
2. Initializing data module...
3. Initializing VcEdit system...
...
11. Running InfEdit for each view...
   Processing view 1/16: 0
     ✓ Saved: view_0000_original.png, view_0000_edited.png
...
```

