#!/usr/bin/env python3
"""
Clock Dataset Generator
대규모 시계 이미지 데이터셋과 추론 데이터를 생성하는 스크립트
"""

import os
import json
import random
from datetime import datetime
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse

from clock_generator import ClockGenerator


class DatasetGenerator:
    def __init__(self, output_dir: str = "dataset", image_size: int = 512):
        self.output_dir = output_dir
        self.image_size = image_size
        self.generator = ClockGenerator(image_size)
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
    
    def generate_time_distribution(self, num_samples: int) -> List[Tuple[int, int]]:
        """다양한 시간 분포 생성"""
        times = []
        
        # 균등 분포 (50%)
        for _ in range(num_samples // 2):
            hour = random.randint(1, 12)
            minute = random.randint(0, 59)
            times.append((hour, minute))
        
        # 특정 시간대 강조 (25%)
        common_times = [
            (12, 0), (12, 30), (1, 0), (1, 30), (2, 0), (2, 30),
            (3, 0), (3, 15), (3, 30), (3, 45), (6, 0), (6, 30),
            (9, 0), (9, 15), (9, 30), (9, 45)
        ]
        for _ in range(num_samples // 4):
            times.append(random.choice(common_times))
        
        # 어려운 시간 (분침이 5분 단위가 아닌 경우) (25%)
        for _ in range(num_samples // 4):
            hour = random.randint(1, 12)
            minute = random.choice([1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 
                                  16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29,
                                  31, 32, 33, 34, 36, 37, 38, 39, 41, 42, 43, 44,
                                  46, 47, 48, 49, 51, 52, 53, 54, 56, 57, 58, 59])
            times.append((hour, minute))
        
        random.shuffle(times)
        return times
    
    def generate_style_distribution(self, num_samples: int) -> List[Dict]:
        """스타일과 설정 분포 생성"""
        styles = list(self.generator.styles.keys())
        configurations = []
        
        for _ in range(num_samples):
            config = {
                'style': random.choice(styles),
                'show_numbers': random.choice([True, False]),
                'noise_level': random.choice(['none', 'low', 'medium']),  # 추후 노이즈 추가용
                'rotation': random.uniform(-5, 5)  # 약간의 회전
            }
            configurations.append(config)
        
        return configurations
    
    def generate_single_sample(self, idx: int, hour: int, minute: int, config: Dict) -> Dict:
        """단일 샘플 생성"""
        try:
            # 시계 이미지와 추론 데이터 생성
            image, reasoning_data = self.generator.generate_clock_image(
                hour=hour,
                minute=minute,
                style=config['style'],
                show_numbers=config['show_numbers']
            )
            
            # 이미지 파일명
            filename = f"clock_{idx:06d}.png"
            image_path = os.path.join(self.output_dir, "images", filename)
            
            # 이미지 저장
            image.save(image_path)
            
            # 메타데이터 추가
            sample_data = {
                "id": idx,
                "filename": filename,
                "image_path": image_path,
                "generation_config": config,
                "target_time": reasoning_data["target_time"],
                "reasoning_process": reasoning_data["reasoning_process"],
                "metadata": reasoning_data["metadata"],
                "created_at": datetime.now().isoformat()
            }
            
            return sample_data
            
        except Exception as e:
            print(f"Error generating sample {idx}: {e}")
            return None
    
    def generate_dataset(self, num_samples: int, num_workers: int = 4) -> List[Dict]:
        """데이터셋 생성"""
        print(f"Generating {num_samples} clock samples...")
        
        # 시간과 스타일 분포 생성
        times = self.generate_time_distribution(num_samples)
        configs = self.generate_style_distribution(num_samples)
        
        # 병렬 처리로 샘플 생성
        samples = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 작업 제출
            futures = []
            for i in range(num_samples):
                hour, minute = times[i % len(times)]
                config = configs[i % len(configs)]
                future = executor.submit(self.generate_single_sample, i, hour, minute, config)
                futures.append(future)
            
            # 결과 수집
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating samples"):
                result = future.result()
                if result is not None:
                    samples.append(result)
        
        return samples
    
    def save_annotations(self, samples: List[Dict], split_ratios: Dict[str, float] = None):
        """주석 데이터 저장 및 데이터셋 분할"""
        if split_ratios is None:
            split_ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
        
        random.shuffle(samples)
        
        total = len(samples)
        train_end = int(total * split_ratios["train"])
        val_end = train_end + int(total * split_ratios["val"])
        
        splits = {
            "train": samples[:train_end],
            "val": samples[train_end:val_end],
            "test": samples[val_end:]
        }
        
        # 각 분할 저장
        for split_name, split_data in splits.items():
            # JSON 형태로 저장
            json_path = os.path.join(self.output_dir, "annotations", f"{split_name}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            
            # JSONL 형태로도 저장 (HuggingFace 호환)
            jsonl_path = os.path.join(self.output_dir, "annotations", f"{split_name}.jsonl")
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for sample in split_data:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            print(f"{split_name}: {len(split_data)} samples saved")
    
    def generate_reasoning_examples(self, num_examples: int = 50) -> List[Dict]:
        """추론 과정 예시 생성 (모델 학습용)"""
        examples = []
        times = self.generate_time_distribution(num_examples)
        configs = self.generate_style_distribution(num_examples)
        
        for i in range(num_examples):
            hour, minute = times[i % len(times)]
            config = configs[i % len(configs)]
            
            # 시계 이미지 생성 (실제 저장하지 않음)
            _, reasoning_data = self.generator.generate_clock_image(
                hour=hour, minute=minute, 
                style=config['style'], show_numbers=config['show_numbers']
            )
            
            # 추론 과정을 텍스트로 변환
            reasoning_text = self.format_reasoning_as_text(reasoning_data)
            
            example = {
                "id": f"reasoning_{i:03d}",
                "time": reasoning_data["target_time"]["formatted"],
                "style": config['style'],
                "has_numbers": config['show_numbers'],
                "reasoning_text": reasoning_text,
                "structured_reasoning": reasoning_data["reasoning_process"]
            }
            
            examples.append(example)
        
        # 추론 예시 저장
        reasoning_path = os.path.join(self.output_dir, "reasoning_examples.json")
        with open(reasoning_path, 'w', encoding='utf-8') as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        
        return examples
    
    def format_reasoning_as_text(self, reasoning_data: Dict) -> str:
        """추론 과정을 자연어 텍스트로 변환"""
        steps = reasoning_data["reasoning_process"]
        text_parts = []
        
        for step in steps:
            step_text = f"**단계 {step['step']}: {step['description']}**\n"
            step_text += f"{step['observation']}\n"
            
            if 'details' in step:
                details = step['details']
                for key, value in details.items():
                    if isinstance(value, str):
                        step_text += f"- {key}: {value}\n"
                    else:
                        step_text += f"- {key}: {value}\n"
            
            text_parts.append(step_text)
        
        final_time = reasoning_data["target_time"]["formatted"]
        conclusion = f"\n**최종 결론**: 이 시계가 가리키는 시간은 {final_time}입니다."
        
        return "\n".join(text_parts) + conclusion
    
    def generate_summary_report(self, samples: List[Dict]) -> Dict:
        """데이터셋 요약 리포트 생성"""
        if not samples:
            return {}
        
        # 통계 수집
        styles = {}
        difficulties = {}
        times_by_hour = {}
        show_numbers_count = {"with_numbers": 0, "without_numbers": 0}
        
        for sample in samples:
            # 스타일 분포
            style = sample["metadata"]["clock_style"]
            styles[style] = styles.get(style, 0) + 1
            
            # 난이도 분포
            difficulty = sample["metadata"]["difficulty_level"]
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
            
            # 시간대 분포
            hour = sample["target_time"]["hour"]
            times_by_hour[hour] = times_by_hour.get(hour, 0) + 1
            
            # 숫자 표시 여부
            if sample["metadata"]["has_numbers"]:
                show_numbers_count["with_numbers"] += 1
            else:
                show_numbers_count["without_numbers"] += 1
        
        summary = {
            "total_samples": len(samples),
            "style_distribution": styles,
            "difficulty_distribution": difficulties,
            "hour_distribution": times_by_hour,
            "number_display_distribution": show_numbers_count,
            "generated_at": datetime.now().isoformat()
        }
        
        # 요약 리포트 저장
        report_path = os.path.join(self.output_dir, "dataset_summary.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Generate clock reading dataset")
    parser.add_argument("--num_samples", type=int, default=1000, 
                       help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="dataset",
                       help="Output directory")
    parser.add_argument("--image_size", type=int, default=512,
                       help="Image size (width and height)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--reasoning_examples", type=int, default=100,
                       help="Number of reasoning examples to generate")
    
    args = parser.parse_args()
    
    # 데이터셋 생성기 초기화
    generator = DatasetGenerator(args.output_dir, args.image_size)
    
    # 메인 데이터셋 생성
    print("Generating main dataset...")
    samples = generator.generate_dataset(args.num_samples, args.num_workers)
    
    # 주석 저장 및 분할
    print("Saving annotations and splitting dataset...")
    generator.save_annotations(samples)
    
    # 추론 예시 생성
    print("Generating reasoning examples...")
    reasoning_examples = generator.generate_reasoning_examples(args.reasoning_examples)
    
    # 요약 리포트 생성
    print("Generating summary report...")
    summary = generator.generate_summary_report(samples)
    
    print(f"\nDataset generation complete!")
    print(f"Total samples: {len(samples)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Reasoning examples: {len(reasoning_examples)}")
    print("\nStyle distribution:")
    for style, count in summary["style_distribution"].items():
        print(f"  {style}: {count}")
    print("\nDifficulty distribution:")
    for difficulty, count in summary["difficulty_distribution"].items():
        print(f"  {difficulty}: {count}")


if __name__ == "__main__":
    main()