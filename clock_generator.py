#!/usr/bin/env python3
"""
Analog Clock Image Generator with Reasoning Output
시계 이미지와 시간 읽기 추론 과정을 생성하는 스크립트
"""

import math
import random
import json
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
import os
from typing import Tuple, Dict, List


class ClockGenerator:
    def __init__(self, image_size: int = 512):
        self.image_size = image_size
        self.center = image_size // 2
        self.clock_radius = int(image_size * 0.4)
        
        # 시계 스타일 설정
        self.styles = {
            'classic': {
                'face_color': '#FFFFFF',
                'border_color': '#000000',
                'hour_hand_color': '#000000',
                'minute_hand_color': '#000000',
                'number_color': '#000000',
                'tick_color': '#000000'
            },
            'modern': {
                'face_color': '#F8F8F8',
                'border_color': '#333333',
                'hour_hand_color': '#2C3E50',
                'minute_hand_color': '#34495E',
                'number_color': '#2C3E50',
                'tick_color': '#BDC3C7'
            },
            'vintage': {
                'face_color': '#FFF8DC',
                'border_color': '#8B4513',
                'hour_hand_color': '#8B4513',
                'minute_hand_color': '#A0522D',
                'number_color': '#8B4513',
                'tick_color': '#CD853F'
            }
        }
    
    def generate_random_time(self) -> Tuple[int, int]:
        """랜덤한 시간 생성 (시, 분)"""
        hour = random.randint(1, 12)
        minute = random.randint(0, 59)
        return hour, minute
    
    def draw_clock_face(self, draw: ImageDraw.ImageDraw, style: Dict[str, str]):
        """시계 바탕과 테두리 그리기"""
        # 시계 바탕
        draw.ellipse(
            [self.center - self.clock_radius, self.center - self.clock_radius,
             self.center + self.clock_radius, self.center + self.clock_radius],
            fill=style['face_color'],
            outline=style['border_color'],
            width=3
        )
    
    def draw_hour_markers(self, draw: ImageDraw.ImageDraw, style: Dict[str, str], show_numbers: bool = True):
        """시간 표시 (숫자 또는 눈금)"""
        for i in range(1, 13):
            angle = math.radians(i * 30 - 90)  # 12시 방향을 0도로 설정
            
            if show_numbers:
                # 숫자 표시
                x = self.center + (self.clock_radius - 30) * math.cos(angle)
                y = self.center + (self.clock_radius - 30) * math.sin(angle)
                
                try:
                    font = ImageFont.truetype("arial.ttf", 24)
                except:
                    font = ImageFont.load_default()
                
                text = str(i)
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                draw.text(
                    (x - text_width//2, y - text_height//2),
                    text,
                    fill=style['number_color'],
                    font=font
                )
            else:
                # 큰 눈금 표시
                x1 = self.center + (self.clock_radius - 20) * math.cos(angle)
                y1 = self.center + (self.clock_radius - 20) * math.sin(angle)
                x2 = self.center + (self.clock_radius - 5) * math.cos(angle)
                y2 = self.center + (self.clock_radius - 5) * math.sin(angle)
                
                draw.line([x1, y1, x2, y2], fill=style['tick_color'], width=3)
        
        # 작은 눈금 (분 표시)
        for i in range(60):
            if i % 5 != 0:  # 5분 단위가 아닌 경우만
                angle = math.radians(i * 6 - 90)
                x1 = self.center + (self.clock_radius - 10) * math.cos(angle)
                y1 = self.center + (self.clock_radius - 10) * math.sin(angle)
                x2 = self.center + (self.clock_radius - 5) * math.cos(angle)
                y2 = self.center + (self.clock_radius - 5) * math.sin(angle)
                
                draw.line([x1, y1, x2, y2], fill=style['tick_color'], width=1)
    
    def draw_hands(self, draw: ImageDraw.ImageDraw, hour: int, minute: int, style: Dict[str, str]):
        """시계 바늘 그리기"""
        # 시침 각도 계산 (분도 고려)
        hour_angle = math.radians((hour % 12) * 30 + minute * 0.5 - 90)
        # 분침 각도 계산
        minute_angle = math.radians(minute * 6 - 90)
        
        # 시침 그리기
        hour_length = self.clock_radius * 0.5
        hour_x = self.center + hour_length * math.cos(hour_angle)
        hour_y = self.center + hour_length * math.sin(hour_angle)
        draw.line([self.center, self.center, hour_x, hour_y], 
                 fill=style['hour_hand_color'], width=6)
        
        # 분침 그리기
        minute_length = self.clock_radius * 0.7
        minute_x = self.center + minute_length * math.cos(minute_angle)
        minute_y = self.center + minute_length * math.sin(minute_angle)
        draw.line([self.center, self.center, minute_x, minute_y], 
                 fill=style['minute_hand_color'], width=3)
        
        # 중심점
        draw.ellipse([self.center-5, self.center-5, self.center+5, self.center+5], 
                    fill=style['hour_hand_color'])
    
    def generate_reasoning(self, hour: int, minute: int, style_name: str, show_numbers: bool) -> Dict:
        """시계 읽기 추론 과정 생성"""
        reasoning_steps = []
        
        # 1. 시계 구조 분석
        reasoning_steps.append({
            "step": 1,
            "description": "시계 구조 파악",
            "observation": f"이 시계는 {style_name} 스타일의 아날로그 시계입니다.",
            "details": {
                "has_numbers": show_numbers,
                "clock_type": "12시간제 아날로그 시계",
                "hand_count": 2
            }
        })
        
        # 2. 바늘 식별
        reasoning_steps.append({
            "step": 2,
            "description": "시계 바늘 식별",
            "observation": "두 개의 바늘을 확인할 수 있습니다.",
            "details": {
                "hour_hand": "짧고 굵은 바늘 (시침)",
                "minute_hand": "길고 얇은 바늘 (분침)",
                "center_position": "두 바늘 모두 시계 중심에서 시작"
            }
        })
        
        # 3. 시침 분석
        hour_position = self._get_hour_position_description(hour, minute)
        reasoning_steps.append({
            "step": 3,
            "description": "시침 위치 분석",
            "observation": f"시침이 {hour_position['description']}에 위치해 있습니다.",
            "details": {
                "pointing_towards": hour_position['pointing'],
                "exact_hour": hour,
                "minute_influence": f"분침의 영향으로 {hour}시에서 약간 이동"
            }
        })
        
        # 4. 분침 분석
        minute_position = self._get_minute_position_description(minute)
        reasoning_steps.append({
            "step": 4,
            "description": "분침 위치 분석",
            "observation": f"분침이 {minute_position['description']}에 위치해 있습니다.",
            "details": {
                "pointing_towards": minute_position['pointing'],
                "exact_minute": minute,
                "calculation": f"분침은 1분당 6도씩 이동하므로 {minute}분 = {minute * 6}도"
            }
        })
        
        # 5. 최종 시간 결론
        time_str = f"{hour}시 {minute:02d}분"
        reasoning_steps.append({
            "step": 5,
            "description": "최종 시간 판독",
            "observation": f"시침과 분침의 위치를 종합하여 현재 시간을 판독합니다.",
            "details": {
                "final_time": time_str,
                "confidence": "높음",
                "verification": f"시침이 {hour}시 방향, 분침이 {minute}분 방향을 가리킴"
            }
        })
        
        return {
            "target_time": {
                "hour": hour,
                "minute": minute,
                "formatted": time_str
            },
            "reasoning_process": reasoning_steps,
            "metadata": {
                "clock_style": style_name,
                "has_numbers": show_numbers,
                "difficulty_level": self._assess_difficulty(hour, minute)
            }
        }
    
    def _get_hour_position_description(self, hour: int, minute: int) -> Dict:
        """시침 위치 설명 생성"""
        hour_names = {
            1: "1시", 2: "2시", 3: "3시", 4: "4시", 5: "5시", 6: "6시",
            7: "7시", 8: "8시", 9: "9시", 10: "10시", 11: "11시", 12: "12시"
        }
        
        base_hour = hour
        next_hour = (hour % 12) + 1
        
        if minute < 15:
            description = f"{hour_names[base_hour]} 근처"
            pointing = f"{base_hour}시 방향"
        elif minute < 45:
            description = f"{hour_names[base_hour]}와 {hour_names[next_hour]} 사이"
            pointing = f"{base_hour}시와 {next_hour}시 사이"
        else:
            description = f"{hour_names[next_hour]} 근처"
            pointing = f"{next_hour}시 방향에 가까움"
        
        return {
            "description": description,
            "pointing": pointing
        }
    
    def _get_minute_position_description(self, minute: int) -> Dict:
        """분침 위치 설명 생성"""
        if minute == 0:
            return {"description": "12시 방향", "pointing": "12시 (0분)"}
        elif minute == 15:
            return {"description": "3시 방향", "pointing": "3시 (15분)"}
        elif minute == 30:
            return {"description": "6시 방향", "pointing": "6시 (30분)"}
        elif minute == 45:
            return {"description": "9시 방향", "pointing": "9시 (45분)"}
        else:
            # 가장 가까운 5분 단위 찾기
            nearest_5min = round(minute / 5) * 5
            hour_equivalent = nearest_5min // 5
            if hour_equivalent == 0:
                hour_equivalent = 12
            
            if abs(minute - nearest_5min) <= 2:
                return {
                    "description": f"{hour_equivalent}시 방향 근처",
                    "pointing": f"{hour_equivalent}시 방향 ({minute}분)"
                }
            else:
                next_5min = ((minute // 5) + 1) * 5
                prev_5min = (minute // 5) * 5
                next_hour = (next_5min // 5) or 12
                prev_hour = (prev_5min // 5) or 12
                
                return {
                    "description": f"{prev_hour}시와 {next_hour}시 사이",
                    "pointing": f"{prev_hour}시와 {next_hour}시 사이 ({minute}분)"
                }
    
    def _assess_difficulty(self, hour: int, minute: int) -> str:
        """시간 읽기 난이도 평가"""
        if minute in [0, 15, 30, 45]:
            return "쉬움"
        elif minute % 5 == 0:
            return "보통"
        else:
            return "어려움"
    
    def generate_clock_image(self, hour: int = None, minute: int = None, 
                           style: str = None, show_numbers: bool = None) -> Tuple[Image.Image, Dict]:
        """시계 이미지와 추론 데이터 생성"""
        # 랜덤 설정
        if hour is None or minute is None:
            hour, minute = self.generate_random_time()
        if style is None:
            style = random.choice(list(self.styles.keys()))
        if show_numbers is None:
            show_numbers = random.choice([True, False])
        
        # 이미지 생성
        image = Image.new('RGB', (self.image_size, self.image_size), 'white')
        draw = ImageDraw.Draw(image)
        
        style_config = self.styles[style]
        
        # 시계 그리기
        self.draw_clock_face(draw, style_config)
        self.draw_hour_markers(draw, style_config, show_numbers)
        self.draw_hands(draw, hour, minute, style_config)
        
        # 추론 데이터 생성
        reasoning_data = self.generate_reasoning(hour, minute, style, show_numbers)
        
        return image, reasoning_data


if __name__ == "__main__":
    generator = ClockGenerator()
    
    # 테스트 이미지 생성
    image, reasoning = generator.generate_clock_image()
    
    print("생성된 추론 데이터:")
    print(json.dumps(reasoning, ensure_ascii=False, indent=2))
    
    # 이미지 저장
    os.makedirs("output", exist_ok=True)
    image.save("output/test_clock.png")
    print("테스트 이미지가 output/test_clock.png에 저장되었습니다.")