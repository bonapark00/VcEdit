#!/bin/bash

echo "=== InfEdit Test Runner ==="
echo ""

# 현재 디렉토리 확인
echo "Current directory: $(pwd)"
echo ""

# Python 환경 확인
echo "Checking Python environment..."
python --version
echo ""

# 필요한 파일들 확인
echo "Checking required files..."
if [ -f "configs/edit-inf.yaml" ]; then
    echo "✓ configs/edit-inf.yaml found"
else
    echo "✗ configs/edit-inf.yaml not found"
    exit 1
fi

if [ -d "gs_data" ]; then
    echo "✓ gs_data directory found"
else
    echo "✗ gs_data directory not found"
    exit 1
fi

if [ -f "gs_data/trained_gs_models/bear/point_cloud.ply" ]; then
    echo "✓ bear point cloud found"
else
    echo "✗ bear point cloud not found"
    exit 1
fi

echo ""

# 테스트 실행
echo "Running InfEdit test..."
echo "This will process all views and save results to 'infedit_test_results' folder"
echo ""

# 간단한 테스트 실행
echo "Running simple test..."
python test_infedit_simple.py

echo ""
echo "=== Test completed ==="
echo "Check the 'infedit_test_results' folder for output images"
echo ""

# 결과 확인
if [ -d "infedit_test_results" ]; then
    echo "Results folder contents:"
    ls -la infedit_test_results/
    echo ""
    echo "Total files: $(ls infedit_test_results/ | wc -l)"
else
    echo "No results folder created. Check for errors above."
fi

