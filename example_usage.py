#!/usr/bin/env python3
"""
Example usage of the clock dataset generator
시계 데이터셋 생성기 사용 예시
"""

import json
from clock_generator import ClockGenerator
from dataset_generator import DatasetGenerator


def example_single_clock():
    """단일 시계 이미지 생성 예시"""
    print("=== 단일 시계 이미지 생성 예시 ===")
    
    generator = ClockGenerator(image_size=512)
    
    # 특정 시간의 시계 생성
    image, reasoning = generator.generate_clock_image(
        hour=3, minute=45, style='vintage', show_numbers=True
    )
    
    # 이미지 저장
    image.save("example_clock.png")
    print("예시 시계 이미지 저장: example_clock.png")
    
    # 추론 과정 출력
    print("\n추론 과정:")
    for step in reasoning['reasoning_process']:
        print(f"단계 {step['step']}: {step['description']}")
        print(f"  관찰: {step['observation']}")
        print()
    
    print(f"최종 시간: {reasoning['target_time']['formatted']}")
    print(f"난이도: {reasoning['metadata']['difficulty_level']}")


def example_small_dataset():
    """소규모 데이터셋 생성 예시"""
    print("\n=== 소규모 데이터셋 생성 예시 ===")
    
    # 50개 샘플로 구성된 작은 데이터셋 생성
    dataset_gen = DatasetGenerator(output_dir="example_dataset", image_size=256)
    
    print("50개 샘플 생성 중...")
    samples = dataset_gen.generate_dataset(num_samples=50, num_workers=2)
    
    print("주석 저장 및 데이터 분할...")
    dataset_gen.save_annotations(samples)
    
    print("추론 예시 생성...")
    reasoning_examples = dataset_gen.generate_reasoning_examples(num_examples=10)
    
    print("요약 리포트 생성...")
    summary = dataset_gen.generate_summary_report(samples)
    
    print(f"\n데이터셋 생성 완료!")
    print(f"총 샘플 수: {len(samples)}")
    print(f"스타일 분포: {summary['style_distribution']}")
    print(f"난이도 분포: {summary['difficulty_distribution']}")


def example_reasoning_analysis():
    """추론 과정 분석 예시"""
    print("\n=== 추론 과정 분석 예시 ===")
    
    generator = ClockGenerator()
    
    # 여러 시간의 추론 과정 비교
    test_times = [(12, 0), (3, 15), (7, 23), (10, 47)]
    
    for hour, minute in test_times:
        print(f"\n--- {hour}시 {minute:02d}분 추론 과정 ---")
        
        _, reasoning = generator.generate_clock_image(
            hour=hour, minute=minute, style='modern', show_numbers=True
        )
        
        # 핵심 추론 단계만 출력
        for step in reasoning['reasoning_process']:
            if step['step'] in [3, 4, 5]:  # 시침, 분침, 결론 단계만
                print(f"{step['description']}: {step['observation']}")
        
        print(f"난이도: {reasoning['metadata']['difficulty_level']}")


def example_custom_reasoning_format():
    """커스텀 추론 형식 예시"""
    print("\n=== 커스텀 추론 형식 예시 ===")
    
    dataset_gen = DatasetGenerator()
    
    # 추론 예시 생성
    examples = dataset_gen.generate_reasoning_examples(num_examples=3)
    
    for i, example in enumerate(examples):
        print(f"\n--- 예시 {i+1}: {example['time']} ---")
        print("구조화된 추론:")
        
        # 자연어 추론 텍스트 출력
        reasoning_text = example['reasoning_text']
        # 마크다운 서식 제거하고 출력
        clean_text = reasoning_text.replace('**', '').replace('\n\n', '\n')
        print(clean_text)


if __name__ == "__main__":
    # 모든 예시 실행
    example_single_clock()
    example_small_dataset() 
    example_reasoning_analysis()
    example_custom_reasoning_format()
    
    print("\n=== 모든 예시 완료 ===")
    print("생성된 파일들:")
    print("- example_clock.png: 단일 시계 이미지")
    print("- example_dataset/: 소규모 데이터셋")
    print("- 콘솔 출력: 다양한 추론 과정 예시")