#!/usr/bin/env python3
"""
영어 데이터 기반 최종 성능 비교 분석
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_final_comparison():
    """영어 데이터 기반 최종 비교 분석"""
    
    print("🎯 영어 데이터 기반 시계 읽기 모델 성능 비교")
    print("=" * 60)
    
    # 베이스라인 결과 (영어 데이터)
    baseline_results = {
        "hour_accuracy": 0.0,
        "minute_accuracy": 0.0,
        "average_accuracy": 0.0,
        "both_accuracy": 0.0
    }
    
    # 미세조정 결과 (영어 데이터, 30 샘플)
    finetuned_results = {
        "hour_accuracy": 0.2,  # 20% (6/30)
        "minute_accuracy": 0.0,  # 0% (0/30)
        "average_accuracy": 0.1,  # 10% 평균
        "both_accuracy": 0.0   # 0% (0/30)
    }
    
    print(f"📊 베이스라인 모델 (BLIP2-OPT-2.7B) - 영어")
    print(f"   시간 정확도: {baseline_results['hour_accuracy']:.1%}")
    print(f"   분 정확도: {baseline_results['minute_accuracy']:.1%}")
    print(f"   평균 정확도: {baseline_results['average_accuracy']:.1%}")
    print(f"   완전 정확도: {baseline_results['both_accuracy']:.1%}")
    print(f"   특징: 빈 응답 또는 질문 반복")
    
    print(f"\n📈 영어 미세조정 모델")
    print(f"   시간 정확도: {finetuned_results['hour_accuracy']:.1%}")
    print(f"   분 정확도: {finetuned_results['minute_accuracy']:.1%}")
    print(f"   평균 정확도: {finetuned_results['average_accuracy']:.1%}")
    print(f"   완전 정확도: {finetuned_results['both_accuracy']:.1%}")
    print(f"   특징: 추론 과정 생성, 일부 시간 인식 가능")
    
    # 성능 개선 계산
    hour_improvement = finetuned_results['hour_accuracy'] - baseline_results['hour_accuracy']
    avg_improvement = finetuned_results['average_accuracy'] - baseline_results['average_accuracy']
    
    print(f"\n🚀 미세조정 효과")
    print(f"   시간 정확도 개선: +{hour_improvement:.1%}")
    print(f"   평균 정확도 개선: +{avg_improvement:.1%}")
    print(f"   상대적 개선: 무한대 (0%에서 {finetuned_results['average_accuracy']:.1%}로)")
    
    # 난이도별 성능 (30 샘플 결과)
    difficulty_performance = {
        "쉬움": {"hour": 0.571, "minute": 0.0, "average": 0.286},
        "보통": {"hour": 0.0, "minute": 0.0, "average": 0.0},
        "어려움": {"hour": 0.095, "minute": 0.0, "average": 0.048}
    }
    
    print(f"\n📊 난이도별 성능 (미세조정 모델)")
    for diff, perf in difficulty_performance.items():
        print(f"   {diff}: 평균={perf['average']:.1%} (시간={perf['hour']:.1%}, 분={perf['minute']:.1%})")
    
    # 스타일별 성능 (30 샘플 결과)
    style_performance = {
        "classic": {"hour": 0.333, "minute": 0.0, "average": 0.167},
        "modern": {"hour": 0.182, "minute": 0.0, "average": 0.091},
        "vintage": {"hour": 0.100, "minute": 0.0, "average": 0.050}
    }
    
    print(f"\n🎨 스타일별 성능 (미세조정 모델)")
    for style, perf in style_performance.items():
        print(f"   {style}: 평균={perf['average']:.1%} (시간={perf['hour']:.1%}, 분={perf['minute']:.1%})")
    
    # 주요 발견사항
    print(f"\n✨ 주요 발견사항")
    print(f"   1. 영어 데이터로 미세조정 시 명확한 성능 향상")
    print(f"   2. 시간(hour) 인식은 일부 가능하지만 분(minute) 인식은 여전히 어려움")
    print(f"   3. 쉬운 난이도에서 상당한 성과 (57.1% 시간 정확도)")
    print(f"   4. classic 스타일에서 가장 좋은 성능")
    print(f"   5. 여전히 분 단위 정확도는 0%로 개선 필요")
    
    # 개선 방안
    print(f"\n💡 향후 개선 방안")
    print(f"   1. 분침 인식에 특화된 학습 데이터 추가")
    print(f"   2. 더 긴 학습 시간 또는 더 많은 에포크")
    print(f"   3. 데이터 증강을 통한 다양한 각도의 시계 이미지")
    print(f"   4. 시간과 분을 각각 예측하는 별도 태스크 설계")
    
    # 시각화
    create_english_comparison_chart(baseline_results, finetuned_results, 
                                  difficulty_performance, style_performance)
    
    # 결과 저장
    final_results = {
        "evaluation_method": "separate_hour_minute_english",
        "baseline_model": "Salesforce/blip2-opt-2.7b",
        "finetuned_model": "english_finetuned_clock_model",
        "test_samples": 30,
        "baseline_performance": baseline_results,
        "finetuned_performance": finetuned_results,
        "improvement": {
            "hour_accuracy": hour_improvement,
            "average_accuracy": avg_improvement
        },
        "difficulty_breakdown": difficulty_performance,
        "style_breakdown": style_performance,
        "key_findings": [
            "영어 데이터로 미세조정 시 명확한 성능 향상",
            "시간 인식 20%, 분 인식 0%",
            "쉬운 난이도에서 57.1% 시간 정확도 달성",
            "classic 스타일에서 최고 성능",
            "분 단위 인식이 가장 큰 개선 포인트"
        ]
    }
    
    with open('final_english_comparison_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    return final_results

def create_english_comparison_chart(baseline, finetuned, difficulty, style):
    """영어 데이터 기반 비교 차트 생성"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # 2x2 레이아웃
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. 전체 성능 비교
    ax1 = fig.add_subplot(gs[0, 0])
    
    models = ['Baseline\n(BLIP2)', 'Fine-tuned\n(English)']
    metrics = ['Hour Acc', 'Minute Acc', 'Average Acc', 'Both Correct']
    
    baseline_values = [baseline['hour_accuracy'], baseline['minute_accuracy'], 
                      baseline['average_accuracy'], baseline['both_accuracy']]
    finetuned_values = [finetuned['hour_accuracy'], finetuned['minute_accuracy'], 
                       finetuned['average_accuracy'], finetuned['both_accuracy']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline_values, width, label='Baseline', 
                   color='lightcoral', alpha=0.8)
    bars2 = ax1.bar(x + width/2, finetuned_values, width, label='Fine-tuned', 
                   color='lightblue', alpha=0.8)
    
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Overall Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.set_ylim(0, 0.6)
    
    # 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 2. 난이도별 성능
    ax2 = fig.add_subplot(gs[0, 1])
    
    difficulties = list(difficulty.keys())
    avg_accs = [difficulty[d]['average'] for d in difficulties]
    hour_accs = [difficulty[d]['hour'] for d in difficulties]
    
    x2 = np.arange(len(difficulties))
    ax2.bar(x2 - 0.2, hour_accs, 0.4, label='Hour Accuracy', color='gold', alpha=0.8)
    ax2.bar(x2 + 0.2, avg_accs, 0.4, label='Average Accuracy', color='lightgreen', alpha=0.8)
    
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Performance by Difficulty')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(difficulties)
    ax2.legend()
    ax2.set_ylim(0, 0.7)
    
    # 3. 스타일별 성능
    ax3 = fig.add_subplot(gs[1, 0])
    
    styles = list(style.keys())
    style_avg_accs = [style[s]['average'] for s in styles]
    style_hour_accs = [style[s]['hour'] for s in styles]
    
    x3 = np.arange(len(styles))
    ax3.bar(x3 - 0.2, style_hour_accs, 0.4, label='Hour Accuracy', color='lightpink', alpha=0.8)
    ax3.bar(x3 + 0.2, style_avg_accs, 0.4, label='Average Accuracy', color='lightsalmon', alpha=0.8)
    
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Performance by Clock Style')
    ax3.set_xticks(x3)
    ax3.set_xticklabels(styles)
    ax3.legend()
    ax3.set_ylim(0, 0.4)
    
    # 4. 개선 효과 요약
    ax4 = fig.add_subplot(gs[1, 1])
    
    improvement_categories = ['Hour\nRecognition', 'Overall\nCapability', 'Easy Tasks', 'Classic Style']
    improvements = [0.2, 0.1, 0.286, 0.167]  # 각 카테고리별 최고 성능
    colors = ['skyblue', 'lightgreen', 'gold', 'lightcoral']
    
    bars4 = ax4.bar(improvement_categories, improvements, color=colors, alpha=0.8)
    ax4.set_ylabel('Achievement Level')
    ax4.set_title('Key Improvements After Fine-tuning')
    ax4.set_ylim(0, 0.35)
    
    # 값 표시
    for bar, value in zip(bars4, improvements):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{value:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('English Data Fine-tuning Results: Hour/Minute Separate Evaluation', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('final_english_comparison_chart.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 차트 저장: final_english_comparison_chart.png")

def main():
    """메인 함수"""
    results = create_final_comparison()
    
    print(f"\n✅ 영어 데이터 기반 성능 비교 분석 완료")
    print(f"📁 저장된 파일:")
    print(f"   - final_english_comparison_results.json: 상세 분석 결과")
    print(f"   - final_english_comparison_chart.png: 종합 비교 차트")
    print(f"   - english_evaluation_plots.png: 세부 평가 시각화")

if __name__ == "__main__":
    main()