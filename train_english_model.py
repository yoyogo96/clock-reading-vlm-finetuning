#!/usr/bin/env python3
"""
Train English Clock VLM
영어 데이터셋으로 시계 읽기 VLM 파인튜닝
"""

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from english_dataset import EnglishClockDataModule
from tqdm import tqdm
import json
import time

def train_english_clock_model():
    """영어 데이터셋으로 시계 모델 학습"""
    
    print("🚀 영어 시계 읽기 모델 학습 시작")
    print("=" * 50)
    
    # 디바이스 설정
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Mac GPU (MPS) 사용")
    else:
        device = torch.device("cpu")
        print("💻 CPU 사용")
    
    # 모델 로드 (새로운 베이스라인부터 시작)
    model_name = "Salesforce/blip2-opt-2.7b"
    print(f"📦 모델 로딩: {model_name}")
    
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if device.type == "mps" else torch.float16
    ).to(device)
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 영어 데이터 설정
    data_module = EnglishClockDataModule(
        data_dir="dataset",
        processor=processor,
        batch_size=4,  # 더 안정적인 배치 크기
        num_workers=0,  # Mac에서 안정적
        reasoning_mode=True
    )
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    
    # 옵티마이저 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    
    # 학습 파라미터
    num_epochs = 1
    steps_per_epoch = 20  # 빠른 학습을 위해
    
    print(f"\n🎯 학습 설정:")
    print(f"  에포크: {num_epochs}")
    print(f"  스텝/에포크: {steps_per_epoch}")
    print(f"  총 스텝: {num_epochs * steps_per_epoch}")
    print(f"  학습률: 5e-6")
    
    # 학습 시작
    model.train()
    all_losses = []
    
    print(f"\n📚 영어 학습 시작...")
    
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        
        train_iter = iter(train_loader)
        epoch_losses = []
        
        for step in range(steps_per_epoch):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            # 디바이스로 이동
            batch['pixel_values'] = batch['pixel_values'].to(device)
            batch['input_ids'] = batch['input_ids'].to(device)
            batch['attention_mask'] = batch['attention_mask'].to(device)
            batch['labels'] = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            loss_value = loss.item()
            epoch_losses.append(loss_value)
            all_losses.append(loss_value)
            
            print(f"  Step {step+1}/{steps_per_epoch}: Loss = {loss_value:.4f}")
            
            # 샘플 정보 출력 (첫 번째 배치만)
            if step == 0:
                print(f"    샘플 예시: {batch['answer_texts'][0]}")
        
        epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"  Epoch {epoch+1} 평균 손실: {epoch_avg_loss:.4f}")
    
    # 모델 저장
    print(f"\n💾 영어 모델 저장...")
    model.save_pretrained("./english_finetuned_clock_model")
    processor.save_pretrained("./english_finetuned_clock_model")
    
    # 학습 결과 저장
    training_results = {
        'model_name': model_name,
        'dataset': 'English Clock Dataset',
        'device': str(device),
        'num_epochs': num_epochs,
        'steps_per_epoch': steps_per_epoch,
        'total_steps': len(all_losses),
        'losses': all_losses,
        'final_loss': all_losses[-1],
        'initial_loss': all_losses[0],
        'loss_reduction': all_losses[0] - all_losses[-1],
        'training_date': '2025-06-21'
    }
    
    with open('english_training_results.json', 'w', encoding='utf-8') as f:
        json.dump(training_results, f, ensure_ascii=False, indent=2)
    
    # 결과 요약
    print(f"\n📊 영어 학습 결과:")
    print(f"  총 학습 스텝: {len(all_losses)}")
    print(f"  초기 손실: {all_losses[0]:.4f}")
    print(f"  최종 손실: {all_losses[-1]:.4f}")
    print(f"  손실 감소: {all_losses[0] - all_losses[-1]:.4f}")
    
    if all_losses[-1] < all_losses[0]:
        print(f"✅ 학습 성공: 손실 감소 확인!")
    else:
        print(f"⚠️ 추가 학습 필요")
    
    print(f"\n🎉 영어 모델 학습 완료!")
    print(f"📁 저장된 파일:")
    print(f"  🤖 ./english_finetuned_clock_model/ - 영어 파인튜닝 모델")
    print(f"  📄 english_training_results.json - 학습 로그")
    
    return training_results

if __name__ == "__main__":
    results = train_english_clock_model()