#!/usr/bin/env python3
"""
Clock Reading Dataset Loader for VLM Fine-tuning
시계 읽기 VLM 파인튜닝용 데이터셋 로더
"""

import json
import os
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Blip2Processor
import random


class ClockReadingDataset(Dataset):
    """시계 읽기 데이터셋 클래스"""
    
    def __init__(
        self, 
        annotation_file: str,
        image_dir: str,
        processor: Blip2Processor,
        mode: str = "train",
        max_length: int = 512,
        reasoning_mode: bool = True
    ):
        """
        Args:
            annotation_file: JSON 주석 파일 경로
            image_dir: 이미지 디렉토리 경로
            processor: BLIP-2 프로세서
            mode: train/val/test
            max_length: 최대 텍스트 길이
            reasoning_mode: 추론 과정 포함 여부
        """
        self.annotation_file = annotation_file
        self.image_dir = image_dir
        self.processor = processor
        self.mode = mode
        self.max_length = max_length
        self.reasoning_mode = reasoning_mode
        
        # 주석 데이터 로드
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        print(f"Loaded {len(self.annotations)} samples from {annotation_file}")
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict:
        """데이터 샘플 반환"""
        sample = self.annotations[idx]
        
        # 이미지 로드
        image_path = os.path.join(self.image_dir, sample['filename'])
        image = Image.open(image_path).convert('RGB')
        
        # 질문 생성
        question = self._generate_question(sample)
        
        # 답변 생성
        answer = self._generate_answer(sample)
        
        # 프로세서로 전처리
        inputs = self.processor(
            images=image,
            text=question,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # 답변을 질문과 합쳐서 라벨 생성 (generative training용)
        full_text = question + " " + answer
        full_inputs = self.processor.tokenizer(
            full_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # 라벨은 full_inputs와 같은 길이로 설정하되, 질문 부분은 -100으로 마스킹
        question_inputs = self.processor.tokenizer(
            question,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        labels = full_inputs['input_ids'].clone()
        question_length = question_inputs['input_ids'].shape[1]
        labels[:, :question_length] = -100  # 질문 부분 마스킹
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'input_ids': full_inputs['input_ids'].squeeze(0),
            'attention_mask': full_inputs['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
            'answer_text': answer,
            'sample_id': sample['id'],
            'difficulty': sample['metadata']['difficulty_level']
        }
    
    def _generate_question(self, sample: Dict) -> str:
        """질문 생성"""
        questions = [
            "이 시계가 가리키는 시간은 몇 시 몇 분입니까?",
            "시계를 보고 현재 시간을 알려주세요.",
            "이 아날로그 시계의 시간을 읽어주세요.",
            "시계 바늘이 가리키는 시간은 무엇입니까?",
            "이 시계가 표시하는 정확한 시간을 말해주세요."
        ]
        
        if self.reasoning_mode and random.random() < 0.5:
            reasoning_questions = [
                "이 시계의 시간을 단계별로 분석하여 읽어주세요.",
                "시침과 분침의 위치를 분석하여 시간을 알려주세요.",
                "시계 읽기 과정을 설명하면서 시간을 알려주세요.",
                "시계 바늘들을 차례로 분석하여 정확한 시간을 구해주세요."
            ]
            questions.extend(reasoning_questions)
        
        return random.choice(questions)
    
    def _generate_answer(self, sample: Dict) -> str:
        """답변 생성"""
        target_time = sample['target_time']['formatted']
        
        if self.reasoning_mode and random.random() < 0.7:
            # 추론 과정 포함 답변
            answer = self._generate_reasoning_answer(sample)
        else:
            # 간단한 답변
            simple_answers = [
                f"{target_time}입니다.",
                f"시계가 가리키는 시간은 {target_time}입니다.",
                f"현재 시간은 {target_time}입니다.",
                f"이 시계는 {target_time}을 나타냅니다."
            ]
            answer = random.choice(simple_answers)
        
        return answer
    
    def _generate_reasoning_answer(self, sample: Dict) -> str:
        """추론 과정 포함 답변 생성"""
        reasoning_steps = sample['reasoning_process']
        target_time = sample['target_time']['formatted']
        
        # 핵심 추론 단계만 선택 (3, 4, 5단계)
        key_steps = [step for step in reasoning_steps if step['step'] in [3, 4, 5]]
        
        reasoning_text = ""
        for step in key_steps:
            if step['step'] == 3:  # 시침 분석
                reasoning_text += f"시침을 보면 {step['observation']} "
            elif step['step'] == 4:  # 분침 분석  
                reasoning_text += f"분침을 보면 {step['observation']} "
            elif step['step'] == 5:  # 결론
                reasoning_text += f"따라서 {target_time}입니다."
        
        return reasoning_text
    
    def get_difficulty_distribution(self) -> Dict[str, int]:
        """난이도별 분포 반환"""
        distribution = {}
        for sample in self.annotations:
            difficulty = sample['metadata']['difficulty_level']
            distribution[difficulty] = distribution.get(difficulty, 0) + 1
        return distribution
    
    def get_style_distribution(self) -> Dict[str, int]:
        """스타일별 분포 반환"""
        distribution = {}
        for sample in self.annotations:
            style = sample['metadata']['clock_style']
            distribution[style] = distribution.get(style, 0) + 1
        return distribution


class ClockDataModule:
    """데이터 모듈 클래스"""
    
    def __init__(
        self,
        data_dir: str,
        processor: Blip2Processor,
        batch_size: int = 8,
        num_workers: int = 4,
        reasoning_mode: bool = True
    ):
        self.data_dir = data_dir
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.reasoning_mode = reasoning_mode
        
        # 데이터셋 경로
        self.image_dir = os.path.join(data_dir, "images")
        self.train_file = os.path.join(data_dir, "annotations", "train.json")
        self.val_file = os.path.join(data_dir, "annotations", "val.json")
        self.test_file = os.path.join(data_dir, "annotations", "test.json")
    
    def setup(self):
        """데이터셋 설정"""
        self.train_dataset = ClockReadingDataset(
            self.train_file, self.image_dir, self.processor, 
            mode="train", reasoning_mode=self.reasoning_mode
        )
        
        self.val_dataset = ClockReadingDataset(
            self.val_file, self.image_dir, self.processor,
            mode="val", reasoning_mode=self.reasoning_mode
        )
        
        self.test_dataset = ClockReadingDataset(
            self.test_file, self.image_dir, self.processor,
            mode="test", reasoning_mode=self.reasoning_mode
        )
        
        print(f"Train dataset: {len(self.train_dataset)} samples")
        print(f"Val dataset: {len(self.val_dataset)} samples") 
        print(f"Test dataset: {len(self.test_dataset)} samples")
        
        # 분포 출력
        print("\nTrain difficulty distribution:", self.train_dataset.get_difficulty_distribution())
        print("Train style distribution:", self.train_dataset.get_style_distribution())
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """배치 collate 함수"""
        from torch.nn.utils.rnn import pad_sequence
        
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        
        # 패딩을 통한 텐서 크기 맞춤
        input_ids = pad_sequence([item['input_ids'] for item in batch], 
                                batch_first=True, padding_value=self.processor.tokenizer.pad_token_id)
        attention_mask = pad_sequence([item['attention_mask'] for item in batch], 
                                     batch_first=True, padding_value=0)
        labels = pad_sequence([item['labels'] for item in batch], 
                             batch_first=True, padding_value=-100)
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'answer_texts': [item['answer_text'] for item in batch],
            'sample_ids': [item['sample_id'] for item in batch],
            'difficulties': [item['difficulty'] for item in batch]
        }


def test_dataset():
    """데이터셋 테스트 함수"""
    from transformers import Blip2Processor
    
    # 프로세서 로드
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    
    # 데이터 모듈 생성
    data_module = ClockDataModule(
        data_dir="dataset",
        processor=processor,
        batch_size=4,
        reasoning_mode=True
    )
    
    # 데이터 설정
    data_module.setup()
    
    # 샘플 확인
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    
    print("Batch keys:", batch.keys())
    print("Pixel values shape:", batch['pixel_values'].shape)
    print("Input IDs shape:", batch['input_ids'].shape)
    print("Labels shape:", batch['labels'].shape)
    print("\nSample answers:")
    for i, answer in enumerate(batch['answer_texts'][:2]):
        print(f"Sample {i+1}: {answer}")


if __name__ == "__main__":
    test_dataset()