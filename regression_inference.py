#!/usr/bin/env python3
"""
English Clock Reading Model Inference with Regression-based Minute Evaluation
영어 시계 읽기 모델 추론 (분은 regression으로 평가)
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
import numpy as np


class RegressionClockInferenceEngine:
    """Regression 기반 시계 읽기 추론 엔진"""
    
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
                print("✅ Mac GPU (MPS) available")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("✅ NVIDIA GPU (CUDA) available")
            else:
                self.device = torch.device("cpu")
                print("⚠️  No GPU available, using CPU")
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
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,  # MPS는 현재 float32만 지원
                low_cpu_mem_usage=True
            )
        elif self.device.type == "cuda":
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16
            )
        else:
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
        question: str = "What time does this clock show?",
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
    
    def extract_time_from_text(self, text: str) -> Optional[Tuple[int, int]]:
        """텍스트에서 시간 추출 (영어 패턴)"""
        # 영어 시간 패턴 매칭
        patterns = [
            # "4:38", "04:38", "4:38 PM" 형태
            r'(\d{1,2}):(\d{1,2})',
            # "4 o'clock 38 minutes", "4 hours 38 minutes" 형태
            r'(\d{1,2})\s*(?:o\'?clock|hours?)\s*(?:and\s*)?(\d{1,2})\s*minutes?',
            # "thirty-eight minutes past four" 형태 (복잡하므로 생략)
            # "four thirty-eight" 형태
            r'(?:at\s+)?(\d{1,2})\s+(\d{1,2})',
            # "It shows 4:38" 형태
            r'shows?\s+(\d{1,2}):(\d{1,2})',
            # "The time is 4:38" 형태
            r'time\s+is\s+(\d{1,2}):(\d{1,2})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                hour = int(match.group(1))
                minute = int(match.group(2))
                
                # 유효성 검사
                if 1 <= hour <= 12 and 0 <= minute <= 59:
                    return (hour, minute)
        
        return None


class RegressionEvaluationEngine:
    """시간(classification) + 분(regression) 평가 엔진"""
    
    def __init__(self, inference_engine: RegressionClockInferenceEngine):
        self.inference_engine = inference_engine
    
    def calculate_minute_regression_score(self, predicted_minute: int, target_minute: int, tolerance: int = 5) -> float:
        """분 regression 점수 계산
        
        Args:
            predicted_minute: 예측된 분
            target_minute: 실제 분
            tolerance: 허용 오차 (분)
            
        Returns:
            0-1 사이의 점수 (1이 완벽, 0이 최악)
        """
        if predicted_minute is None:
            return 0.0
        
        # 절대 오차 계산
        error = abs(predicted_minute - target_minute)
        
        # 원형 시계 특성 고려 (예: 58분과 2분의 차이는 4분)
        circular_error = min(error, 60 - error)
        
        # 허용 오차 내면 1.0, 그 외에는 선형적으로 감소
        if circular_error <= tolerance:
            return 1.0
        else:
            # 최대 오차는 30분 (시계 반바퀴)
            max_error = 30
            score = max(0.0, (max_error - circular_error) / (max_error - tolerance))
            return score
    
    def evaluate_dataset(
        self,
        test_file: str,
        image_dir: str,
        minute_tolerance: int = 5,
        output_file: str = "regression_evaluation_results.json"
    ) -> Dict:
        """데이터셋 전체 평가 (시간은 classification, 분은 regression)"""
        # 테스트 데이터 로드
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        print(f"Evaluating {len(test_data)} samples with regression-based minute scoring...")
        print(f"Minute tolerance: ±{minute_tolerance} minutes")
        
        results = []
        
        # 정확도 카운터
        hour_correct = 0
        minute_regression_scores = []
        both_correct = 0  # 시간 정확 + 분 regression 점수 > 0.8
        total_samples = len(test_data)
        
        # 세부 통계
        difficulty_stats = defaultdict(lambda: {
            'hour_correct': 0, 'minute_scores': [], 'both_correct': 0, 'total': 0
        })
        style_stats = defaultdict(lambda: {
            'hour_correct': 0, 'minute_scores': [], 'both_correct': 0, 'total': 0
        })
        
        for sample in tqdm(test_data, desc="Evaluating"):
            # 이미지 경로 구성
            if 'filename' in sample:
                image_path = os.path.join(image_dir, sample['filename'])
            else:
                # image_path에서 파일명 추출
                filename = os.path.basename(sample['image_path'])
                image_path = os.path.join(image_dir, filename)
            
            # 예측
            prediction = self.inference_engine.predict_time(image_path)
            
            # 정답 시간
            target_hour = sample['target_time']['hour']
            target_minute = sample['target_time']['minute']
            
            # 예측 시간 추출
            pred_time = self.inference_engine.extract_time_from_text(prediction)
            
            # 정확도 계산
            hour_match = False
            minute_regression_score = 0.0
            
            if pred_time:
                pred_hour, pred_minute = pred_time
                hour_match = (pred_hour == target_hour)
                minute_regression_score = self.calculate_minute_regression_score(
                    pred_minute, target_minute, minute_tolerance
                )
            
            # 카운터 업데이트
            if hour_match:
                hour_correct += 1
            minute_regression_scores.append(minute_regression_score)
            
            # "both correct"는 시간 정확 + 분 regression 점수 > 0.8
            if hour_match and minute_regression_score > 0.8:
                both_correct += 1
            
            # 통계 업데이트
            difficulty = sample['metadata']['difficulty_level']
            style = sample['metadata']['clock_style']
            
            difficulty_stats[difficulty]['total'] += 1
            style_stats[style]['total'] += 1
            
            if hour_match:
                difficulty_stats[difficulty]['hour_correct'] += 1
                style_stats[style]['hour_correct'] += 1
            
            difficulty_stats[difficulty]['minute_scores'].append(minute_regression_score)
            style_stats[style]['minute_scores'].append(minute_regression_score)
            
            if hour_match and minute_regression_score > 0.8:
                difficulty_stats[difficulty]['both_correct'] += 1
                style_stats[style]['both_correct'] += 1
            
            # 결과 저장
            result = {
                'sample_id': sample['id'],
                'filename': os.path.basename(image_path),
                'target_hour': target_hour,
                'target_minute': target_minute,
                'target_time': f"{target_hour}:{target_minute:02d}",
                'prediction': prediction,
                'predicted_time': f"{pred_time[0]}:{pred_time[1]:02d}" if pred_time else "Parse failed",
                'hour_correct': hour_match,
                'minute_regression_score': minute_regression_score,
                'minute_error': abs(pred_time[1] - target_minute) if pred_time else 60,
                'both_correct': hour_match and minute_regression_score > 0.8,
                'difficulty': difficulty,
                'style': style
            }
            
            results.append(result)
        
        # 전체 정확도 계산
        hour_accuracy = hour_correct / total_samples
        minute_regression_avg = np.mean(minute_regression_scores)
        both_accuracy = both_correct / total_samples
        combined_score = (hour_accuracy + minute_regression_avg) / 2  # 시간과 분의 평균
        
        # 난이도별 정확도
        difficulty_accuracies = {}
        for diff, stats in difficulty_stats.items():
            if stats['total'] > 0:
                difficulty_accuracies[diff] = {
                    'hour_accuracy': stats['hour_correct'] / stats['total'],
                    'minute_regression_avg': np.mean(stats['minute_scores']),
                    'both_accuracy': stats['both_correct'] / stats['total'],
                    'combined_score': (stats['hour_correct'] / stats['total'] + np.mean(stats['minute_scores'])) / 2
                }
        
        # 스타일별 정확도
        style_accuracies = {}
        for style, stats in style_stats.items():
            if stats['total'] > 0:
                style_accuracies[style] = {
                    'hour_accuracy': stats['hour_correct'] / stats['total'],
                    'minute_regression_avg': np.mean(stats['minute_scores']),
                    'both_accuracy': stats['both_correct'] / stats['total'],
                    'combined_score': (stats['hour_correct'] / stats['total'] + np.mean(stats['minute_scores'])) / 2
                }
        
        evaluation_summary = {
            'evaluation_method': 'hour_classification_minute_regression',
            'minute_tolerance': minute_tolerance,
            'overall_performance': {
                'hour_accuracy': hour_accuracy,
                'minute_regression_avg': minute_regression_avg,
                'both_accuracy': both_accuracy,
                'combined_score': combined_score
            },
            'total_samples': total_samples,
            'performance_counts': {
                'hour_correct': hour_correct,
                'minute_scores_distribution': {
                    'mean': float(np.mean(minute_regression_scores)),
                    'std': float(np.std(minute_regression_scores)),
                    'min': float(np.min(minute_regression_scores)),
                    'max': float(np.max(minute_regression_scores)),
                    'median': float(np.median(minute_regression_scores))
                },
                'both_correct': both_correct
            },
            'difficulty_breakdown': difficulty_accuracies,
            'style_breakdown': style_accuracies,
            'detailed_results': results
        }
        
        # 결과 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_summary, f, ensure_ascii=False, indent=2)
        
        # 요약 출력
        print(f"\n=== Evaluation Results (Hour Classification + Minute Regression) ===")
        print(f"Hour Accuracy: {hour_accuracy:.4f} ({hour_correct}/{total_samples})")
        print(f"Minute Regression Avg: {minute_regression_avg:.4f}")
        print(f"Both High Performance: {both_accuracy:.4f} ({both_correct}/{total_samples})")
        print(f"Combined Score: {combined_score:.4f}")
        
        print(f"\nMinute Regression Distribution:")
        print(f"  Mean: {np.mean(minute_regression_scores):.3f}")
        print(f"  Std:  {np.std(minute_regression_scores):.3f}")
        print(f"  Min:  {np.min(minute_regression_scores):.3f}")
        print(f"  Max:  {np.max(minute_regression_scores):.3f}")
        
        print(f"\nBreakdown by Difficulty:")
        for diff, acc in difficulty_accuracies.items():
            print(f"  {diff}: Combined={acc['combined_score']:.3f} (H={acc['hour_accuracy']:.3f}, M_reg={acc['minute_regression_avg']:.3f})")
        
        print(f"\nBreakdown by Style:")
        for style, acc in style_accuracies.items():
            print(f"  {style}: Combined={acc['combined_score']:.3f} (H={acc['hour_accuracy']:.3f}, M_reg={acc['minute_regression_avg']:.3f})")
        
        return evaluation_summary
    
    def visualize_results(self, evaluation_results: Dict, save_path: str = "regression_evaluation_plots.png"):
        """결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 전체 성능 (시간 classification + 분 regression)
        ax1 = axes[0, 0]
        overall = evaluation_results['overall_performance']
        categories = ['Hour\nAccuracy', 'Minute\nRegression', 'Both High\nPerformance', 'Combined\nScore']
        values = [
            overall['hour_accuracy'],
            overall['minute_regression_avg'],
            overall['both_accuracy'],
            overall['combined_score']
        ]
        
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'gold']
        bars = ax1.bar(categories, values, color=colors)
        ax1.set_ylabel('Score')
        ax1.set_title('Overall Performance (Hour Classification + Minute Regression)')
        ax1.set_ylim(0, 1)
        
        # 값 표시
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 2. 난이도별 combined score
        ax2 = axes[0, 1]
        difficulty_data = evaluation_results['difficulty_breakdown']
        difficulties = list(difficulty_data.keys())
        combined_scores = [difficulty_data[d]['combined_score'] for d in difficulties]
        hour_accs = [difficulty_data[d]['hour_accuracy'] for d in difficulties]
        minute_regs = [difficulty_data[d]['minute_regression_avg'] for d in difficulties]
        
        x = np.arange(len(difficulties))
        width = 0.25
        
        ax2.bar(x - width, hour_accs, width, label='Hour Accuracy', color='lightblue')
        ax2.bar(x, minute_regs, width, label='Minute Regression', color='lightgreen')
        ax2.bar(x + width, combined_scores, width, label='Combined Score', color='gold')
        
        ax2.set_ylabel('Score')
        ax2.set_title('Performance by Difficulty Level')
        ax2.set_xticks(x)
        ax2.set_xticklabels(difficulties)
        ax2.legend()
        ax2.set_ylim(0, max(max(combined_scores), max(hour_accs), max(minute_regs)) + 0.1)
        
        # 3. 스타일별 성능
        ax3 = axes[1, 0]
        style_data = evaluation_results['style_breakdown']
        styles = list(style_data.keys())
        style_combined = [style_data[s]['combined_score'] for s in styles]
        style_hour = [style_data[s]['hour_accuracy'] for s in styles]
        style_minute = [style_data[s]['minute_regression_avg'] for s in styles]
        
        x3 = np.arange(len(styles))
        ax3.bar(x3 - width, style_hour, width, label='Hour Accuracy', color='lightpink')
        ax3.bar(x3, style_minute, width, label='Minute Regression', color='lightsalmon')
        ax3.bar(x3 + width, style_combined, width, label='Combined Score', color='lightcyan')
        
        ax3.set_ylabel('Score')
        ax3.set_title('Performance by Clock Style')
        ax3.set_xticks(x3)
        ax3.set_xticklabels(styles)
        ax3.legend()
        ax3.set_ylim(0, max(max(style_combined), max(style_hour), max(style_minute)) + 0.1)
        
        # 4. 분 regression 점수 분포
        ax4 = axes[1, 1]
        minute_scores = [result['minute_regression_score'] for result in evaluation_results['detailed_results']]
        
        ax4.hist(minute_scores, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
        ax4.axvline(np.mean(minute_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(minute_scores):.3f}')
        ax4.axvline(np.median(minute_scores), color='blue', linestyle='--', 
                   label=f'Median: {np.median(minute_scores):.3f}')
        
        ax4.set_xlabel('Minute Regression Score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Minute Regression Scores')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Regression-based Clock Reading Model Inference")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--mode", type=str, choices=['single', 'evaluate'], 
                       default='single', help="Inference mode")
    parser.add_argument("--image_path", type=str, 
                       help="Single image path (for single mode)")
    parser.add_argument("--test_file", type=str, default="medium_test_english.json",
                       help="Test annotation file (for evaluate mode)")
    parser.add_argument("--image_dir", type=str, default="dataset/images",
                       help="Test images directory (for evaluate mode)")
    parser.add_argument("--output_file", type=str, default="regression_evaluation_results.json",
                       help="Output file for evaluation results")
    parser.add_argument("--minute_tolerance", type=int, default=5,
                       help="Tolerance for minute regression (in minutes)")
    
    args = parser.parse_args()
    
    # 추론 엔진 초기화
    inference_engine = RegressionClockInferenceEngine(args.model_path)
    
    if args.mode == 'single':
        if not args.image_path:
            print("Error: --image_path is required for single mode")
            return
        
        # 단일 이미지 예측
        result = inference_engine.predict_time(args.image_path)
        print(f"Image: {args.image_path}")
        print(f"Prediction: {result}")
        
        # 시간 추출 시도
        time_extracted = inference_engine.extract_time_from_text(result)
        if time_extracted:
            print(f"Extracted time: {time_extracted[0]}:{time_extracted[1]:02d}")
        else:
            print("Failed to extract time from prediction")
        
    elif args.mode == 'evaluate':
        # 데이터셋 평가
        evaluator = RegressionEvaluationEngine(inference_engine)
        
        results = evaluator.evaluate_dataset(
            args.test_file,
            args.image_dir,
            args.minute_tolerance,
            args.output_file
        )
        
        # 시각화
        evaluator.visualize_results(results)


if __name__ == "__main__":
    main()