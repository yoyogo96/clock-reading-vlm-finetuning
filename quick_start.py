#!/usr/bin/env python3
"""
Quick Start Script for Clock Reading VLM
시계 읽기 VLM 빠른 시작 스크립트
"""

import os
import sys
import argparse
from pathlib import Path

def check_requirements():
    """필요한 패키지 확인"""
    required_packages = [
        'torch', 'transformers', 'datasets', 'PIL', 'tqdm', 
        'numpy', 'sklearn', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            if package == 'PIL':
                missing_packages.append('Pillow')
            elif package == 'sklearn':
                missing_packages.append('scikit-learn')
            else:
                missing_packages.append(package)
    
    if missing_packages:
        print("❌ 누락된 패키지:", ', '.join(missing_packages))
        print("💡 설치 명령어: pip install " + ' '.join(missing_packages))
        return False
    else:
        print("✅ 모든 필요 패키지가 설치되어 있습니다.")
        return True

def check_dataset():
    """데이터셋 확인"""
    dataset_dir = Path("dataset")
    
    if not dataset_dir.exists():
        print("❌ 데이터셋 폴더가 없습니다.")
        print("💡 먼저 'python dataset_generator.py --num_samples 1000'을 실행하세요.")
        return False
    
    required_files = [
        "annotations/train.json",
        "annotations/val.json", 
        "annotations/test.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not (dataset_dir / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ 누락된 데이터셋 파일:", ', '.join(missing_files))
        return False
    else:
        print("✅ 데이터셋이 준비되어 있습니다.")
        return True

def check_gpu():
    """GPU 확인 (Mac GPU 포함)"""
    try:
        import torch
        
        if torch.backends.mps.is_available():
            print("✅ Mac GPU (MPS) 사용 가능")
            return True
        elif torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ NVIDIA GPU 사용 가능: {gpu_name} (총 {gpu_count}개)")
            return True
        else:
            print("⚠️  GPU를 사용할 수 없습니다. CPU로 학습합니다.")
            return False
    except ImportError:
        print("❌ PyTorch가 설치되지 않았습니다.")
        return False

def run_data_test():
    """데이터 로더 테스트"""
    print("\n🧪 데이터 로더 테스트 중...")
    
    try:
        from clock_dataset import test_dataset
        test_dataset()
        print("✅ 데이터 로더 테스트 성공")
        return True
    except Exception as e:
        print(f"❌ 데이터 로더 테스트 실패: {e}")
        return False

def run_training(args):
    """학습 실행"""
    print("\n🚀 모델 학습을 시작합니다...")
    
    # 학습 명령어 구성
    cmd_args = [
        "python", "train_clock_vlm.py",
        "--data_dir", args.data_dir,
        "--output_dir", args.output_dir,
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--num_epochs", str(args.num_epochs)
    ]
    
    if args.use_wandb:
        cmd_args.append("--use_wandb")
    
    if args.reasoning_mode:
        cmd_args.append("--reasoning_mode")
    
    print("실행 명령어:", ' '.join(cmd_args))
    
    # 학습 실행
    try:
        os.system(' '.join(cmd_args))
        print("✅ 학습 완료")
        return True
    except Exception as e:
        print(f"❌ 학습 실패: {e}")
        return False

def run_quick_evaluation():
    """빠른 평가"""
    print("\n📊 빠른 평가를 수행합니다...")
    
    # 최고 모델 경로 확인
    best_model_path = "checkpoints/best_model"
    if not os.path.exists(best_model_path):
        print("❌ 학습된 모델이 없습니다. 먼저 학습을 완료하세요.")
        return False
    
    # 평가 실행
    cmd = f"python inference.py --model_path {best_model_path} --mode evaluate"
    
    try:
        os.system(cmd)
        print("✅ 평가 완료")
        return True
    except Exception as e:
        print(f"❌ 평가 실패: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Clock Reading VLM Quick Start")
    parser.add_argument("--mode", type=str, 
                       choices=['check', 'test', 'train', 'eval', 'all'],
                       default='all',
                       help="실행 모드")
    
    # 학습 관련 인수
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="배치 크기 (GPU 메모리에 따라 조정)")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=5,
                       help="에포크 수 (빠른 테스트용)")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--reasoning_mode", action="store_true", default=True)
    
    args = parser.parse_args()
    
    print("🎯 시계 읽기 VLM 빠른 시작")
    print("=" * 50)
    
    # 환경 확인
    if args.mode in ['check', 'all']:
        print("\n🔍 환경 확인 중...")
        
        if not check_requirements():
            sys.exit(1)
        
        if not check_dataset():
            print("💡 데이터셋 생성을 먼저 수행하세요:")
            print("   python dataset_generator.py --num_samples 1000")
            sys.exit(1)
        
        check_gpu()
    
    # 데이터 테스트
    if args.mode in ['test', 'all']:
        if not run_data_test():
            sys.exit(1)
    
    # 학습 실행
    if args.mode in ['train', 'all']:
        if not run_training(args):
            sys.exit(1)
    
    # 평가 실행
    if args.mode in ['eval', 'all']:
        if not run_quick_evaluation():
            sys.exit(1)
    
    print("\n🎉 모든 과정이 완료되었습니다!")
    print("\n📁 생성된 파일들:")
    print("   - checkpoints/: 학습된 모델 체크포인트")
    print("   - checkpoints/best_model/: 최고 성능 모델")
    print("   - evaluation_results.json: 평가 결과")
    print("   - evaluation_plots.png: 결과 시각화")
    
    print("\n🔥 추가 사용법:")
    print("   # 단일 이미지 예측")
    print("   python inference.py --model_path checkpoints/best_model --mode single --image_path example_clock.png")
    print("   ")
    print("   # 추론 과정 포함 예측")  
    print("   python inference.py --model_path checkpoints/best_model --mode single --image_path example_clock.png --reasoning")


if __name__ == "__main__":
    main()