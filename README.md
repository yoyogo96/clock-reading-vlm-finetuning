# Clock Reading Dataset Generator

LLM이 아날로그 시계를 정확하게 읽을 수 있도록 fine-tuning하기 위한 데이터셋을 생성하는 도구입니다.

## 주요 기능

- **다양한 시계 이미지 생성**: 클래식, 모던, 빈티지 스타일의 아날로그 시계
- **체계적인 시간 읽기 추론**: 5단계 추론 과정으로 시계 읽기 논리 설명
- **대규모 데이터셋 생성**: 병렬 처리를 통한 효율적인 데이터셋 생성
- **다양한 난이도**: 쉬운 시간(정각, 30분)부터 어려운 시간(불규칙한 분)까지

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 기본 사용

```bash
# 1000개 샘플 생성
python dataset_generator.py --num_samples 1000

# 사용자 정의 설정
python dataset_generator.py \
    --num_samples 5000 \
    --output_dir my_dataset \
    --image_size 256 \
    --num_workers 8 \
    --reasoning_examples 200
```

### 단일 시계 이미지 테스트

```bash
python clock_generator.py
```

## 출력 구조

```
dataset/
├── images/                 # 시계 이미지 파일들
│   ├── clock_000001.png
│   └── ...
├── annotations/           # 주석 데이터
│   ├── train.json        # 훈련 세트 (80%)
│   ├── val.json          # 검증 세트 (10%)
│   ├── test.json         # 테스트 세트 (10%)
│   ├── train.jsonl       # HuggingFace 호환 형식
│   ├── val.jsonl
│   └── test.jsonl
├── reasoning_examples.json # 추론 과정 예시
└── dataset_summary.json   # 데이터셋 통계
```

## 데이터 형식

### 시계 이미지 주석

```json
{
  "id": 1,
  "filename": "clock_000001.png",
  "target_time": {
    "hour": 3,
    "minute": 15,
    "formatted": "3시 15분"
  },
  "reasoning_process": [
    {
      "step": 1,
      "description": "시계 구조 파악",
      "observation": "이 시계는 classic 스타일의 아날로그 시계입니다.",
      "details": {
        "has_numbers": true,
        "clock_type": "12시간제 아날로그 시계"
      }
    }
  ],
  "metadata": {
    "clock_style": "classic",
    "has_numbers": true,
    "difficulty_level": "쉬움"
  }
}
```

### 추론 과정 5단계

1. **시계 구조 파악**: 시계 타입과 기본 구조 인식
2. **바늘 식별**: 시침과 분침 구분
3. **시침 위치 분석**: 시침이 가리키는 시간 해석
4. **분침 위치 분석**: 분침이 가리키는 분 해석  
5. **최종 시간 판독**: 종합적인 시간 결론

## 시계 스타일

- **Classic**: 전통적인 흰색 바탕, 검은색 숫자와 바늘
- **Modern**: 현대적인 회색톤, 깔끔한 디자인
- **Vintage**: 빈티지 느낌의 베이지/갈색 톤

## 난이도 설정

- **쉬움**: 정각, 15분, 30분, 45분
- **보통**: 5분 단위 시간
- **어려움**: 불규칙한 분 단위 시간

## VLM 모델 파인튜닝

### 빠른 시작

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 데이터셋 생성 (이미 완료된 경우 생략)
python dataset_generator.py --num_samples 1000

# 3. 모든 과정 자동 실행 (권장)
python quick_start.py --mode all --num_epochs 5

# 4. 개별 단계 실행
python quick_start.py --mode check    # 환경 확인
python quick_start.py --mode test     # 데이터 테스트  
python quick_start.py --mode train    # 모델 학습
python quick_start.py --mode eval     # 모델 평가
```

### 상세 학습 과정

```bash
# 1. BLIP-2 모델 파인튜닝
python train_clock_vlm.py \
    --data_dir dataset \
    --output_dir checkpoints \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --num_epochs 10 \
    --use_wandb \
    --reasoning_mode

# 2. 모델 평가
python inference.py \
    --model_path checkpoints/best_model \
    --mode evaluate \
    --test_file dataset/annotations/test.json \
    --image_dir dataset/images

# 3. 단일 이미지 예측
python inference.py \
    --model_path checkpoints/best_model \
    --mode single \
    --image_path example_clock.png \
    --reasoning
```

### 학습 설정

- **모델**: BLIP-2 (Salesforce/blip2-opt-2.7b)
- **데이터**: 1000개 시계 이미지 + 체계적 추론 과정
- **학습 방식**: 추론 과정 포함 대화형 학습
- **평가 지표**: 시간 정확도, 난이도별/스타일별 성능

## 활용 예시

생성된 데이터셋과 모델은 다음과 같이 활용할 수 있습니다:

1. **Vision-Language 모델 Fine-tuning**: BLIP-2, LLaVA 등
2. **시계 읽기 특화 모델 훈련**: 체계적 추론 능력 학습
3. **추론 능력 개선 학습**: 5단계 논리적 사고 과정
4. **멀티모달 이해 성능 평가**: 아날로그 값 해석 능력 측정