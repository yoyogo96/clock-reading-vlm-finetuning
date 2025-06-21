#!/usr/bin/env python3
"""
Clock Reading VLM Fine-tuning Script
BLIP-2를 사용한 시계 읽기 모델 파인튜닝
"""

import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    Blip2ForConditionalGeneration, 
    Blip2Processor,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  Wandb not available, logging disabled")
from sklearn.metrics import accuracy_score
import numpy as np

from clock_dataset import ClockDataModule


class ClockVLMTrainer:
    """시계 읽기 VLM 트레이너"""
    
    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        data_dir: str = "dataset",
        output_dir: str = "checkpoints",
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        num_epochs: int = 10,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        eval_steps: int = 500,
        save_steps: int = 1000,
        logging_steps: int = 50,
        use_wandb: bool = False,
        reasoning_mode: bool = True,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.use_wandb = use_wandb
        self.reasoning_mode = reasoning_mode
        
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
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 모델과 프로세서 로드
        self._load_model_and_processor()
        
        # 데이터 모듈 설정
        self._setup_data()
        
        # 옵티마이저 및 스케줄러 설정
        self._setup_training()
        
        # Wandb 초기화
        if self.use_wandb:
            self._init_wandb()
    
    def _load_model_and_processor(self):
        """모델과 프로세서 로드"""
        print(f"Loading model: {self.model_name}")
        
        self.processor = Blip2Processor.from_pretrained(self.model_name)
        
        # Mac GPU에 최적화된 모델 로딩
        if self.device.type == "mps":
            # Mac GPU용 설정
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # MPS는 현재 float32만 지원
                low_cpu_mem_usage=True
            )
            self.model.to(self.device)
        elif self.device.type == "cuda":
            # NVIDIA GPU용 설정
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            # CPU용 설정
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32
            )
            self.model.to(self.device)
        
        print(f"Model loaded with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def _setup_data(self):
        """데이터 설정"""
        print("Setting up data...")
        
        self.data_module = ClockDataModule(
            data_dir=self.data_dir,
            processor=self.processor,
            batch_size=self.batch_size,
            reasoning_mode=self.reasoning_mode
        )
        
        self.data_module.setup()
        
        self.train_loader = self.data_module.train_dataloader()
        self.val_loader = self.data_module.val_dataloader()
        self.test_loader = self.data_module.test_dataloader()
    
    def _setup_training(self):
        """트레이닝 설정"""
        # 총 스텝 수 계산
        self.total_steps = len(self.train_loader) * self.num_epochs
        
        # 옵티마이저 (Mac GPU 최적화)
        if self.device.type == "mps":
            # Mac GPU에서는 더 보수적인 설정 사용
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=0.01,
                eps=1e-6  # MPS 안정성을 위한 더 큰 epsilon
            )
        else:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=0.01
            )
        
        # 스케줄러
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps
        )
        
        print(f"Training setup complete. Total steps: {self.total_steps}")
    
    def _init_wandb(self):
        """Wandb 초기화"""
        if WANDB_AVAILABLE and self.use_wandb:
            wandb.init(
                project="clock-reading-vlm",
                config={
                    "model_name": self.model_name,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "num_epochs": self.num_epochs,
                    "reasoning_mode": self.reasoning_mode
                }
            )
        else:
            self.use_wandb = False
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """한 에포크 학습"""
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for step, batch in enumerate(pbar):
            # 배치를 디바이스로 이동
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            outputs = self.model(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # 통계 업데이트
            total_loss += loss.item()
            num_batches += 1
            
            # Progress bar 업데이트
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # 로깅
            if step % self.logging_steps == 0:
                current_step = epoch * len(self.train_loader) + step
                self._log_metrics({
                    'train/loss': loss.item(),
                    'train/learning_rate': self.scheduler.get_last_lr()[0],
                    'train/step': current_step
                })
            
            # 평가
            if step % self.eval_steps == 0 and step > 0:
                val_metrics = self.validate()
                self._log_metrics(val_metrics)
                self.model.train()
            
            # 체크포인트 저장
            if step % self.save_steps == 0 and step > 0:
                self.save_checkpoint(epoch, step)
        
        return {'train/loss': total_loss / num_batches}
    
    def validate(self) -> Dict[str, float]:
        """검증"""
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                batch = self._move_batch_to_device(batch)
                
                # Loss 계산
                outputs = self.model(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                total_loss += outputs.loss.item()
                num_batches += 1
                
                # 생성 및 정확도 계산
                generated_ids = self.model.generate(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=150,
                    num_beams=3,
                    early_stopping=True
                )
                
                generated_texts = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                
                # 정확도 계산 (시간 추출 기반)
                for pred, target in zip(generated_texts, batch['answer_texts']):
                    if self._extract_time_accuracy(pred, target):
                        correct_predictions += 1
                    total_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        metrics = {
            'val/loss': total_loss / num_batches,
            'val/accuracy': accuracy,
            'val/num_samples': total_predictions
        }
        
        print(f"Validation - Loss: {metrics['val/loss']:.4f}, Accuracy: {metrics['val/accuracy']:.4f}")
        
        return metrics
    
    def _extract_time_accuracy(self, prediction: str, target: str) -> bool:
        """시간 추출 기반 정확도 계산"""
        import re
        
        # 시간 패턴 매칭 (예: "3시 45분", "12시 30분")
        time_pattern = r'(\d{1,2})시\s*(\d{1,2})분'
        
        pred_match = re.search(time_pattern, prediction)
        target_match = re.search(time_pattern, target)
        
        if pred_match and target_match:
            pred_hour = int(pred_match.group(1))
            pred_minute = int(pred_match.group(2))
            target_hour = int(target_match.group(1))
            target_minute = int(target_match.group(2))
            
            return pred_hour == target_hour and pred_minute == target_minute
        
        return False
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """배치를 디바이스로 이동"""
        return {
            'pixel_values': batch['pixel_values'].to(self.device),
            'input_ids': batch['input_ids'].to(self.device),
            'attention_mask': batch['attention_mask'].to(self.device),
            'labels': batch['labels'].to(self.device),
            'answer_texts': batch['answer_texts'],
            'sample_ids': batch['sample_ids'],
            'difficulties': batch['difficulties']
        }
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """메트릭 로깅"""
        if WANDB_AVAILABLE and self.use_wandb:
            wandb.log(metrics)
        
        # 콘솔 로깅 (일부 메트릭만)
        if 'train/loss' in metrics:
            print(f"Train Loss: {metrics['train/loss']:.4f}")
    
    def save_checkpoint(self, epoch: int, step: int):
        """체크포인트 저장"""
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch}-step-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 모델과 프로세서 저장
        self.model.save_pretrained(checkpoint_dir)
        self.processor.save_pretrained(checkpoint_dir)
        
        # 트레이닝 상태 저장
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': None,  # 필요시 추가
        }, os.path.join(checkpoint_dir, 'training_state.pt'))
        
        print(f"Checkpoint saved: {checkpoint_dir}")
    
    def train(self):
        """전체 학습 루프"""
        print("Starting training...")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            print(f"\n=== Epoch {epoch+1}/{self.num_epochs} ===")
            
            # 학습
            train_metrics = self.train_epoch(epoch)
            
            # 검증
            val_metrics = self.validate()
            
            # 메트릭 로깅
            epoch_metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
            self._log_metrics(epoch_metrics)
            
            # 최고 성능 모델 저장
            if val_metrics['val/loss'] < best_val_loss:
                best_val_loss = val_metrics['val/loss']
                self.save_best_model()
            
            # 에포크 체크포인트 저장
            if (epoch + 1) % 2 == 0:  # 2 에포크마다 저장
                self.save_checkpoint(epoch, -1)
        
        print("Training completed!")
        
        # 최종 테스트
        test_metrics = self.test()
        print(f"Final test metrics: {test_metrics}")
        
        if WANDB_AVAILABLE and self.use_wandb:
            wandb.finish()
    
    def save_best_model(self):
        """최고 성능 모델 저장"""
        best_model_dir = os.path.join(self.output_dir, "best_model")
        os.makedirs(best_model_dir, exist_ok=True)
        
        self.model.save_pretrained(best_model_dir)
        self.processor.save_pretrained(best_model_dir)
        
        print(f"Best model saved: {best_model_dir}")
    
    def test(self) -> Dict[str, float]:
        """테스트 평가"""
        print("Running final test...")
        
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0
        difficulty_stats = {'쉬움': [0, 0], '보통': [0, 0], '어려움': [0, 0]}
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                batch = self._move_batch_to_device(batch)
                
                generated_ids = self.model.generate(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=150,
                    num_beams=3,
                    early_stopping=True
                )
                
                generated_texts = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                
                for pred, target, difficulty in zip(
                    generated_texts, batch['answer_texts'], batch['difficulties']
                ):
                    is_correct = self._extract_time_accuracy(pred, target)
                    if is_correct:
                        correct_predictions += 1
                        difficulty_stats[difficulty][0] += 1
                    
                    total_predictions += 1
                    difficulty_stats[difficulty][1] += 1
        
        overall_accuracy = correct_predictions / total_predictions
        
        # 난이도별 정확도 계산
        difficulty_accuracies = {}
        for diff, (correct, total) in difficulty_stats.items():
            if total > 0:
                difficulty_accuracies[f'test/accuracy_{diff}'] = correct / total
        
        test_metrics = {
            'test/accuracy': overall_accuracy,
            'test/total_samples': total_predictions,
            **difficulty_accuracies
        }
        
        return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Clock Reading VLM")
    parser.add_argument("--model_name", type=str, default="Salesforce/blip2-opt-2.7b",
                       help="Pre-trained model name")
    parser.add_argument("--data_dir", type=str, default="dataset",
                       help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                       help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of epochs")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--reasoning_mode", action="store_true", default=True,
                       help="Enable reasoning mode")
    
    args = parser.parse_args()
    
    # 트레이너 초기화 및 학습
    trainer = ClockVLMTrainer(
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        use_wandb=args.use_wandb,
        reasoning_mode=args.reasoning_mode
    )
    
    trainer.train()


if __name__ == "__main__":
    main()