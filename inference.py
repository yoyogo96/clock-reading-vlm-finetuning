#!/usr/bin/env python3
"""
Clock Reading Model Inference
학습된 시계 읽기 모델 추론 및 평가
"""

import os
import json
import argparse
from typing import List, Dict, Tuple, Optional
from PIL import Image
import torch
from transformers import Blip2ForConditionalGeneration, Blip2Processor
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class ClockInferenceEngine:
    """시계 읽기 추론 엔진"""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None
    ):
        self.model_path = model_path
        
        # 디바이스 설정 (Mac GPU 지원)
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("✅ Mac GPU (MPS) 사용 가능")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("✅ NVIDIA GPU (CUDA) 사용 가능")
            else:
                self.device = torch.device("cpu")
                print("⚠️  GPU 없음, CPU 사용")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # 모델과 프로세서 로드
        self._load_model()
    
    def _load_model(self):
        """모델과 프로세서 로드"""
        print(f"Loading model from: {self.model_path}")
        
        self.processor = Blip2Processor.from_pretrained(self.model_path)
        
        # Mac GPU에 최적화된 모델 로딩
        if self.device.type == "mps":
            # Mac GPU용 설정
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,  # MPS는 현재 float32만 지원
                low_cpu_mem_usage=True
            )
        elif self.device.type == "cuda":
            # NVIDIA GPU용 설정
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16
            )
        else:
            # CPU용 설정
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32
            )
        
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully")
    
    def predict_time(
        self, 
        image_path: str, 
        question: str = "이 시계가 가리키는 시간은 몇 시 몇 분입니까?",
        max_length: int = 150,
        num_beams: int = 3
    ) -> str:
        """단일 이미지에 대한 시간 예측"""
        # 이미지 로드
        image = Image.open(image_path).convert('RGB')
        
        # 입력 전처리
        inputs = self.processor(
            images=image,
            text=question,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 생성
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
        
        # 디코딩
        generated_text = self.processor.decode(
            generated_ids[0], 
            skip_special_tokens=True
        )
        
        return generated_text
    
    def predict_with_reasoning(
        self, 
        image_path: str,
        max_length: int = 300
    ) -> str:
        """추론 과정 포함 예측"""
        question = "이 시계의 시간을 단계별로 분석하여 읽어주세요."
        return self.predict_time(image_path, question, max_length)
    
    def batch_predict(
        self,
        image_paths: List[str],
        questions: Optional[List[str]] = None,
        batch_size: int = 8
    ) -> List[str]:
        """배치 예측"""
        if questions is None:
            questions = ["이 시계가 가리키는 시간은 몇 시 몇 분입니까?"] * len(image_paths)
        
        predictions = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Predicting"):
            batch_images = image_paths[i:i+batch_size]
            batch_questions = questions[i:i+batch_size]
            
            batch_preds = []
            for img_path, question in zip(batch_images, batch_questions):
                pred = self.predict_time(img_path, question)
                batch_preds.append(pred)
            
            predictions.extend(batch_preds)
        
        return predictions
    
    def extract_time_from_text(self, text: str) -> Optional[Tuple[int, int]]:
        """텍스트에서 시간 추출"""
        # 한국어 시간 패턴 매칭
        patterns = [
            r'(\d{1,2})시\s*(\d{1,2})분',
            r'(\d{1,2}):(\d{1,2})',
            r'(\d{1,2})\s*시\s*(\d{1,2})\s*분'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                hour = int(match.group(1))
                minute = int(match.group(2))
                
                # 유효성 검사
                if 1 <= hour <= 12 and 0 <= minute <= 59:
                    return (hour, minute)
        
        return None


class ClockModelEvaluator:
    """시계 모델 평가기"""
    
    def __init__(self, inference_engine: ClockInferenceEngine):
        self.inference_engine = inference_engine
    
    def evaluate_dataset(
        self,
        test_file: str,
        image_dir: str,
        output_file: str = "evaluation_results.json"
    ) -> Dict:
        """데이터셋 전체 평가"""
        # 테스트 데이터 로드
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        print(f"Evaluating {len(test_data)} samples...")
        
        results = []
        correct_total = 0
        difficulty_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        style_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for sample in tqdm(test_data, desc="Evaluating"):
            # 이미지 경로
            image_path = os.path.join(image_dir, sample['filename'])
            
            # 예측
            prediction = self.inference_engine.predict_time(image_path)
            
            # 정답 시간
            target_hour = sample['target_time']['hour']
            target_minute = sample['target_time']['minute']
            
            # 예측 시간 추출
            pred_time = self.inference_engine.extract_time_from_text(prediction)
            
            # 정확도 계산
            is_correct = False
            if pred_time:
                pred_hour, pred_minute = pred_time
                is_correct = (pred_hour == target_hour and pred_minute == target_minute)
            
            if is_correct:
                correct_total += 1
            
            # 통계 업데이트
            difficulty = sample['metadata']['difficulty_level']
            style = sample['metadata']['clock_style']
            
            difficulty_stats[difficulty]['total'] += 1
            style_stats[style]['total'] += 1
            
            if is_correct:
                difficulty_stats[difficulty]['correct'] += 1
                style_stats[style]['correct'] += 1
            
            # 결과 저장
            result = {
                'sample_id': sample['id'],
                'filename': sample['filename'],
                'target_time': f"{target_hour}시 {target_minute:02d}분",
                'prediction': prediction,
                'predicted_time': f"{pred_time[0]}시 {pred_time[1]:02d}분" if pred_time else "추출 실패",
                'is_correct': is_correct,
                'difficulty': difficulty,
                'style': style
            }
            
            results.append(result)
        
        # 전체 통계 계산
        total_samples = len(test_data)
        overall_accuracy = correct_total / total_samples
        
        # 난이도별 정확도
        difficulty_accuracies = {}
        for diff, stats in difficulty_stats.items():
            if stats['total'] > 0:
                difficulty_accuracies[diff] = stats['correct'] / stats['total']
        
        # 스타일별 정확도
        style_accuracies = {}
        for style, stats in style_stats.items():
            if stats['total'] > 0:
                style_accuracies[style] = stats['correct'] / stats['total']
        
        evaluation_summary = {
            'overall_accuracy': overall_accuracy,
            'total_samples': total_samples,
            'correct_predictions': correct_total,
            'difficulty_breakdown': {
                diff: {
                    'accuracy': acc,
                    'correct': difficulty_stats[diff]['correct'],
                    'total': difficulty_stats[diff]['total']
                }
                for diff, acc in difficulty_accuracies.items()
            },
            'style_breakdown': {
                style: {
                    'accuracy': acc,
                    'correct': style_stats[style]['correct'],
                    'total': style_stats[style]['total']
                }
                for style, acc in style_accuracies.items()
            },
            'detailed_results': results
        }
        
        # 결과 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_summary, f, ensure_ascii=False, indent=2)
        
        # 요약 출력
        print(f"\n=== 평가 결과 ===")
        print(f"전체 정확도: {overall_accuracy:.4f} ({correct_total}/{total_samples})")
        print(f"\n난이도별 정확도:")
        for diff, acc in difficulty_accuracies.items():
            stats = difficulty_stats[diff]
            print(f"  {diff}: {acc:.4f} ({stats['correct']}/{stats['total']})")
        print(f"\n스타일별 정확도:")
        for style, acc in style_accuracies.items():
            stats = style_stats[style]
            print(f"  {style}: {acc:.4f} ({stats['correct']}/{stats['total']})")
        
        return evaluation_summary
    
    def analyze_errors(self, evaluation_results: Dict) -> Dict:
        """오류 분석"""
        errors = []
        
        for result in evaluation_results['detailed_results']:
            if not result['is_correct']:
                errors.append(result)
        
        print(f"\n=== 오류 분석 ===")
        print(f"총 오류 수: {len(errors)}")
        
        # 난이도별 오류 분포
        error_by_difficulty = defaultdict(int)
        for error in errors:
            error_by_difficulty[error['difficulty']] += 1
        
        print(f"\n난이도별 오류 분포:")
        for diff, count in error_by_difficulty.items():
            print(f"  {diff}: {count}개")
        
        # 몇 가지 오류 예시 출력
        print(f"\n오류 예시 (처음 5개):")
        for i, error in enumerate(errors[:5]):
            print(f"  {i+1}. 파일: {error['filename']}")
            print(f"     정답: {error['target_time']}")
            print(f"     예측: {error['prediction']}")
            print(f"     난이도: {error['difficulty']}")
            print()
        
        return {
            'total_errors': len(errors),
            'error_by_difficulty': dict(error_by_difficulty),
            'error_examples': errors[:10]
        }
    
    def visualize_results(self, evaluation_results: Dict, save_path: str = "evaluation_plots.png"):
        """결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 전체 정확도
        ax1 = axes[0, 0]
        accuracy = evaluation_results['overall_accuracy']
        ax1.bar(['전체'], [accuracy], color='skyblue')
        ax1.set_ylabel('정확도')
        ax1.set_title('전체 정확도')
        ax1.set_ylim(0, 1)
        
        # 2. 난이도별 정확도
        ax2 = axes[0, 1]
        difficulty_data = evaluation_results['difficulty_breakdown']
        difficulties = list(difficulty_data.keys())
        accuracies = [difficulty_data[d]['accuracy'] for d in difficulties]
        
        ax2.bar(difficulties, accuracies, color=['lightcoral', 'lightgreen', 'lightsalmon'])
        ax2.set_ylabel('정확도')
        ax2.set_title('난이도별 정확도')
        ax2.set_ylim(0, 1)
        
        # 3. 스타일별 정확도
        ax3 = axes[1, 0]
        style_data = evaluation_results['style_breakdown']
        styles = list(style_data.keys())
        style_accuracies = [style_data[s]['accuracy'] for s in styles]
        
        ax3.bar(styles, style_accuracies, color=['gold', 'lightblue', 'lightpink'])
        ax3.set_ylabel('정확도')
        ax3.set_title('스타일별 정확도')
        ax3.set_ylim(0, 1)
        
        # 4. 혼동 행렬 (정확/부정확)
        ax4 = axes[1, 1]
        correct = evaluation_results['correct_predictions']
        incorrect = evaluation_results['total_samples'] - correct
        
        ax4.pie([correct, incorrect], labels=['정확', '부정확'], 
               colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%')
        ax4.set_title('전체 예측 결과 분포')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"결과 시각화 저장: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Clock Reading Model Inference")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--mode", type=str, choices=['single', 'evaluate'], 
                       default='single', help="Inference mode")
    parser.add_argument("--image_path", type=str, 
                       help="Single image path (for single mode)")
    parser.add_argument("--test_file", type=str, default="dataset/annotations/test.json",
                       help="Test annotation file (for evaluate mode)")
    parser.add_argument("--image_dir", type=str, default="dataset/images",
                       help="Test images directory (for evaluate mode)")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json",
                       help="Output file for evaluation results")
    parser.add_argument("--reasoning", action="store_true",
                       help="Use reasoning mode")
    
    args = parser.parse_args()
    
    # 추론 엔진 초기화
    inference_engine = ClockInferenceEngine(args.model_path)
    
    if args.mode == 'single':
        if not args.image_path:
            print("Error: --image_path is required for single mode")
            return
        
        # 단일 이미지 예측
        if args.reasoning:
            result = inference_engine.predict_with_reasoning(args.image_path)
        else:
            result = inference_engine.predict_time(args.image_path)
        
        print(f"이미지: {args.image_path}")
        print(f"예측 결과: {result}")
        
    elif args.mode == 'evaluate':
        # 데이터셋 평가
        evaluator = ClockModelEvaluator(inference_engine)
        
        results = evaluator.evaluate_dataset(
            args.test_file,
            args.image_dir,
            args.output_file
        )
        
        # 오류 분석
        error_analysis = evaluator.analyze_errors(results)
        
        # 시각화
        evaluator.visualize_results(results)


if __name__ == "__main__":
    main()