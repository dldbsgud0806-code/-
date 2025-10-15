> 카메라를 활용한 실시간 당구 샷 분석 및 코칭 시스템

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-alpha-orange)]()
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

---

## 📋 한눈에 보기

- **상태**: `alpha`
- **라이선스**: `MIT License`
- **언어**: `Python 3.8+`
- **주요 기능**: `실시간 공 추적, 큐 각도 분석, 최적 샷 추천, 물리 시뮬레이션`

---

## 🎯 프로젝트 소개

본 프로젝트는 카메라를 활용하여 당구대 위의 공과 큐의 위치 및 각도를 실시간으로 추적하고, 물리 모델 기반 시뮬레이션을 통해 최적의 샷 후보를 제시하는 오픈소스 코칭 시스템입니다. 

시각적 오버레이로 예상 경로를 표시하며, 학습자의 실력 향상을 돕는 교육용 도구로 설계되었습니다. 모든 데이터는 로컬에서 처리되어 프라이버시를 보장하고, 모듈화된 구조로 다양한 플랫폼에 이식 가능합니다.

---

## 📚 목차

- [핵심 특징](#-핵심-특징)
- [설치 방법](#-설치-방법)
- [빠른 시작](#-빠른-시작)
- [사용 예제](#-사용-예제)
- [프로젝트 구조](#-프로젝트-구조)
- [모듈별 상세 코드](#-모듈별-상세-코드)
- [아키텍처](#-아키텍처)
- [결과 화면 예시](#-결과-화면-예시)
- [개발 로드맵](#-개발-로드맵)
- [기여하기](#-기여하기)
- [라이선스](#-라이선스)

---

## ✨ 핵심 특징

- **실시간 분석**: 30-60fps로 공의 위치와 큐의 각도를 추적합니다
- **물리 기반 시뮬레이션**: 충돌, 마찰, 쿠션 반사를 정확히 계산합니다
- **시각적 피드백**: 예상 경로와 최적 타점을 화면에 오버레이로 표시합니다
- **실패 원인 분석**: 득점 실패 시 공의 최종 위치를 분석하여 실패 원인을 자동으로 진단하고 개선 방안을 제시합니다
- **프라이버시 우선**: 모든 데이터는 로컬에서 처리되며, 익명화 옵션을 제공합니다
- **모듈화 설계**: 웹/데스크탑/모바일 환경에 쉽게 이식 가능합니다

---

## 🚀 설치 방법

### 시스템 요구사항

- Python 3.8 이상
- pip 최신 버전
- (선택) CUDA 지원 GPU

### 설치

```bash
git clone https://github.com/your-username/billiard-coaching-system.git
cd billiard-coaching-system

# 가상환경 생성 및 활성화 (Windows)
python -m venv venv
venv\Scripts\activate

# 가상환경 생성 및 활성화 (Mac/Linux)
python3 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### requirements.txt

```txt
# 컴퓨터 비전
opencv-python>=4.5.0
numpy>=1.20.0
scipy>=1.7.0

# 물리 시뮬레이션
matplotlib>=3.3.0

# 데이터 처리
pandas>=1.3.0

# 로깅 및 설정
python-dotenv>=0.19.0
pyyaml>=5.4.0

# 테스트
pytest>=6.2.0
pytest-cov>=2.12.0
```

### 환경 설정

프로젝트 루트에 `.env` 파일을 생성하고 필요한 설정을 입력합니다:

```env
CAMERA_ID=0
DEBUG_MODE=true
SAVE_LOGS=true
FPS_TARGET=30
```

---

## ⚡ 빠른 시작

### MVP 실행

```bash
# 비전 모듈 테스트
python main.py --mode vision --input examples/sample_videos/test.mp4

# 전체 코칭 시스템 실행
python main.py --mode full --camera 0
```

➡ 실행 후 OpenCV 윈도우가 열리며 실시간 분석 결과를 확인할 수 있습니다

---

## 📖 사용 예제

### 1️⃣ 비디오 파일에서 공 추적

```bash
python main.py --mode vision --input sample_video.mp4
```

### 2️⃣ 실시간 카메라로 샷 코칭

```bash
python main.py --mode full --camera 0
```

### 3️⃣ 결과 확인

- 선택된 샷 후보 및 예상 경로가 화면에 오버레이로 표시됩니다
- 최적 타점과 큐 각도 피드백이 텍스트로 제공됩니다
- **득점 실패 시** 공의 최종 위치를 분석하여 실패 원인(각도 오차, 파워 부족, 스핀 과다 등)과 개선 방안을 제시합니다
- 로그는 `logs/` 폴더에 자동 저장됩니다

### 4️⃣ 종료 방법

OpenCV 윈도우에서 `q` 키를 누르거나, 터미널에서 `Ctrl + C` 입력 후 종료  
가상환경 비활성화 → `deactivate`

---

## 📁 프로젝트 구조

```
billiard-coaching-system/
│
├── main.py                # 진입점
├── requirements.txt       # Python 의존성
├── .env                   # 환경 변수
├── config.yaml            # 설정 파일
│
├── vision/                # 비전 모듈
│   ├── __init__.py
│   ├── core.py           # 공/큐 감지
│   ├── calibration.py    # 카메라 캘리브레이션
│   └── tracking.py       # 객체 추적
│
├── physics/              # 물리 시뮬레이션
│   ├── __init__.py
│   ├── engine.py         # 시뮬레이션 엔진
│   ├── collision.py      # 충돌 처리
│   └── cushion.py        # 쿠션 반사
│
├── ranking/              # 샷 평가 엔진
│   ├── __init__.py
│   ├── score.py          # 점수 계산
│   ├── generator.py      # 후보 생성
│   └── optimizer.py      # 최적화
│
├── feedback/             # 피드백 엔진
│   ├── __init__.py
│   ├── overlay.py        # 시각적 오버레이
│   ├── text.py           # 텍스트 설명
│   └── failure_analyzer.py  # 실패 원인 분석
│
├── data/                 # 데이터 관리
│   ├── __init__.py
│   ├── storage.py        # 로컬 저장
│   └── privacy.py        # 익명화
│
├── utils/                # 유틸리티
│   ├── __init__.py
│   ├── logger.py         # 로깅
│   └── config.py         # 설정 로더
│
├── tests/                # 테스트
│   ├── test_vision.py
│   ├── test_physics.py
│   └── test_ranking.py
│
└── examples/             # 예제
    └── sample_videos/    # 샘플 비디오
```

---

## 💻 모듈별 상세 코드

### 1. main.py - 진입점

```python
"""
메인 실행 파일
"""
import argparse
import cv2
import numpy as np
from vision.core import BallDetector, CueDetector
from physics.engine import PhysicsEngine
from ranking.generator import ShotGenerator
from ranking.score import ShotScorer
from feedback.overlay import OverlayRenderer
from utils.logger import setup_logger
from utils.config import load_config

def main():
    parser = argparse.ArgumentParser(description='Billiard Shot Coaching System')
    parser.add_argument('--mode', choices=['vision', 'full'], default='full',
                        help='실행 모드 선택')
    parser.add_argument('--camera', type=int, default=0,
                        help='카메라 디바이스 ID')
    parser.add_argument('--input', type=str, default=None,
                        help='입력 비디오 파일 경로')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='설정 파일 경로')
    
    args = parser.parse_args()
    
    # 로거 및 설정 로드
    logger = setup_logger()
    config = load_config(args.config)
    
    # 모듈 초기화
    ball_detector = BallDetector(config['vision']['ball_detection'])
    cue_detector = CueDetector(config['vision']['cue_detection'])
    physics_engine = PhysicsEngine(config['physics'])
    shot_generator = ShotGenerator(config['ranking'])
    shot_scorer = ShotScorer(config['ranking'])
    overlay_renderer = OverlayRenderer()
    
    # 비디오 소스 설정
    if args.input:
        cap = cv2.VideoCapture(args.input)
    else:
        cap = cv2.VideoCapture(args.camera)
    
    logger.info(f"시스템 시작 - 모드: {args.mode}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1. 공 감지
        balls = ball_detector.detect(frame)
        
        if args.mode == 'full' and len(balls) > 0:
            # 2. 큐 감지
            cue = cue_detector.detect(frame, balls)
            
            # 3. 샷 후보 생성
            candidates = shot_generator.generate(balls, cue)
            
            # 4. 물리 시뮬레이션 및 점수 계산
            best_shot = None
            best_score = -1
            
            for candidate in candidates[:10]:  # 상위 10개만 시뮬레이션
                result = physics_engine.simulate(balls, candidate)
                score = shot_scorer.score(result, candidate)
                
                if score > best_score:
                    best_score = score
                    best_shot = (candidate, result)
            
            # 5. 시각적 피드백
            if best_shot:
                frame = overlay_renderer.render(frame, balls, cue, best_shot)
        else:
            # Vision 모드: 공만 표시
            frame = overlay_renderer.render_balls(frame, balls)
        
        # 화면 표시
        cv2.imshow('Billiard Coaching System', frame)
        
        # 'q' 키로 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    logger.info("시스템 종료")

if __name__ == '__main__':
    main()
```

---

### 2. vision/core.py - 공/큐 감지

```python
"""
비전 모듈: 공과 큐 감지
"""
import cv2
import numpy as np

class BallDetector:
    """당구공 감지 클래스"""
    
    def __init__(self, config):
        self.min_radius = config.get('min_radius', 5)
        self.max_radius = config.get('max_radius', 30)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
    
    def detect(self, frame):
        """
        프레임에서 당구공을 감지합니다.
        
        Args:
            frame: 입력 이미지 (BGR)
        
        Returns:
            balls: 감지된 공 리스트 [{'x': int, 'y': int, 'r': int, 'color': str}, ...]
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        
        # Hough Circle Transform으로 원 감지
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        
        balls = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                
                # 색상 분석
                color = self._analyze_color(frame, x, y, r)
                
                balls.append({
                    'x': int(x),
                    'y': int(y),
                    'r': int(r),
                    'color': color
                })
        
        return balls
    
    def _analyze_color(self, frame, x, y, r):
        """공의 색상을 분석합니다."""
        # ROI 추출
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        
        mean_color = cv2.mean(frame, mask=mask)[:3]
        b, g, r = mean_color
        
        # 간단한 색상 분류
        if r > 150 and g < 100 and b < 100:
            return 'red'
        elif r < 100 and g < 100 and b > 150:
            return 'blue'
        elif r > 200 and g > 200 and b > 200:
            return 'white'
        else:
            return 'other'


class CueDetector:
    """큐 감지 클래스"""
    
    def __init__(self, config):
        self.min_length = config.get('min_length', 50)
        self.angle_tolerance = config.get('angle_tolerance', 5)
    
    def detect(self, frame, balls):
        """
        프레임에서 큐를 감지합니다.
        
        Args:
            frame: 입력 이미지 (BGR)
            balls: 감지된 공 리스트
        
        Returns:
            cue: 큐 정보 {'start': (x1, y1), 'end': (x2, y2), 'angle': float} 또는 None
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough Line Transform으로 직선 감지
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=self.min_length,
            maxLineGap=10
        )
        
        if lines is None:
            return None
        
        # 가장 긴 선을 큐로 간주
        longest_line = None
        max_length = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if length > max_length:
                max_length = length
                longest_line = line[0]
        
        if longest_line is not None:
            x1, y1, x2, y2 = longest_line
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            return {
                'start': (x1, y1),
                'end': (x2, y2),
                'angle': angle
            }
        
        return None
```

---

### 3. physics/engine.py - 물리 시뮬레이션

```python
"""
물리 엔진: 당구공 충돌 및 경로 시뮬레이션
"""
import numpy as np

class Ball:
    """당구공 객체"""
    
    def __init__(self, x, y, vx=0, vy=0, r=2.25, m=0.17):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.r = r
        self.m = m


class PhysicsEngine:
    """물리 시뮬레이션 엔진"""
    
    def __init__(self, config):
        self.friction = config.get('friction_coefficient', 0.02)
        self.restitution = config.get('restitution', 0.95)
        self.dt = config.get('time_delta', 0.001)
        self.max_steps = config.get('simulation_steps', 1000)
        self.table_width = config.get('table_width', 254)  # cm
        self.table_height = config.get('table_height', 127)  # cm
    
    def simulate(self, balls_data, shot_candidate):
        """
        샷을 시뮬레이션합니다.
        
        Args:
            balls_data: 감지된 공 리스트
            shot_candidate: 샷 후보 {'cue_ball': int, 'direction': (vx, vy), 'power': float}
        
        Returns:
            result: 시뮬레이션 결과 {'path': [...], 'success': bool, 'collisions': [...]}
        """
        # Ball 객체 생성
        balls = []
        for b in balls_data:
            balls.append(Ball(b['x'], b['y'], 0, 0, b['r']))
        
        # 큐볼에 초기 속도 부여
        cue_idx = shot_candidate['cue_ball']
        direction = shot_candidate['direction']
        power = shot_candidate['power']
        
        balls[cue_idx].vx = direction[0] * power
        balls[cue_idx].vy = direction[1] * power
        
        # 시뮬레이션 실행
        path = []
        collisions = []
        
        for step in range(self.max_steps):
            # 모든 공 업데이트
            for ball in balls:
                self._update_ball(ball)
            
            # 충돌 감지 및 처리
            for i in range(len(balls)):
                for j in range(i + 1, len(balls)):
                    if self._check_collision(balls[i], balls[j]):
                        self._resolve_collision(balls[i], balls[j])
                        collisions.append((i, j, step * self.dt))
            
            # 경로 기록 (큐볼만)
            path.append((balls[cue_idx].x, balls[cue_idx].y))
            
            # 모든 공이 정지하면 종료
            if self._all_stopped(balls):
                break
        
        return {
            'path': path,
            'success': len(collisions) > 0,
            'collisions': collisions,
            'final_positions': [(b.x, b.y) for b in balls]
        }
    
    def _update_ball(self, ball):
        """공의 위치와 속도를 업데이트합니다."""
        # 마찰 적용
        speed = np.sqrt(ball.vx**2 + ball.vy**2)
        if speed > 0:
            friction_force = self.friction * self.dt
            ball.vx *= max(0, 1 - friction_force / speed)
            ball.vy *= max(0, 1 - friction_force / speed)
        
        # 위치 업데이트
        ball.x += ball.vx * self.dt
        ball.y += ball.vy * self.dt
        
        # 벽 충돌 처리
        if ball.x - ball.r <= 0 or ball.x + ball.r >= self.table_width:
            ball.vx *= -self.restitution
            ball.x = np.clip(ball.x, ball.r, self.table_width - ball.r)
        
        if ball.y - ball.r <= 0 or ball.y + ball.r >= self.table_height:
            ball.vy *= -self.restitution
            ball.y = np.clip(ball.y, ball.r, self.table_height - ball.r)
    
    def _check_collision(self, ball1, ball2):
        """두 공의 충돌 여부를 확인합니다."""
        dx = ball2.x - ball1.x
        dy = ball2.y - ball1.y
        distance = np.sqrt(dx**2 + dy**2)
        return distance <= (ball1.r + ball2.r)
    
    def _resolve_collision(self, ball1, ball2):
        """두 공의 충돌을 처리합니다."""
        # 충돌 벡터
        dx = ball2.x - ball1.x
        dy = ball2.y - ball1.y
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance == 0:
            return
        
        # 정규화된 충돌 벡터
        nx = dx / distance
        ny = dy / distance
        
        # 상대 속도
        dvx = ball1.vx - ball2.vx
        dvy = ball1.vy - ball2.vy
        
        # 충돌 축 방향 상대 속도
        dvn = dvx * nx + dvy * ny
        
        # 이미 멀어지고 있으면 무시
        if dvn <= 0:
            return
        
        # 충격량 계산
        impulse = 2 * dvn / (ball1.m + ball2.m)
        
        # 속도 업데이트
        ball1.vx -= impulse * ball2.m * nx
        ball1.vy -= impulse * ball2.m * ny
        ball2.vx += impulse * ball1.m * nx
        ball2.vy += impulse * ball1.m * ny
    
    def _all_stopped(self, balls):
        """모든 공이 정지했는지 확인합니다."""
        threshold = 0.01
        for ball in balls:
            if np.sqrt(ball.vx**2 + ball.vy**2) > threshold:
                return False
        return True
```

---

### 4. ranking/generator.py - 샷 후보 생성

```python
"""
샷 후보 생성기
"""
import numpy as np

class ShotGenerator:
    """샷 후보 생성 클래스"""
    
    def __init__(self, config):
        self.angle_resolution = config.get('angle_resolution', 15)  # 각도 해상도(도)
        self.power_levels = config.get('power_levels', [0.3, 0.5, 0.7, 1.0])
    
    def generate(self, balls, cue):
        """
        가능한 샷 후보를 생성합니다.
        
        Args:
            balls: 감지된 공 리스트
            cue: 큐 정보
        
        Returns:
            candidates: 샷 후보 리스트
        """
        candidates = []
        
        # 흰 공(큐볼) 찾기
        cue_ball_idx = None
        for i, ball in enumerate(balls):
            if ball['color'] == 'white':
                cue_ball_idx = i
                break
        
        if cue_ball_idx is None:
            return candidates
        
        cue_ball = balls[cue_ball_idx]
        
        # 모든 각도와 파워 조합 생성
        for angle_deg in range(0, 360, self.angle_resolution):
            angle_rad = np.deg2rad(angle_deg)
            direction = (np.cos(angle_rad), np.sin(angle_rad))
            
            for power in self.power_levels:
                candidate = {
                    'cue_ball': cue_ball_idx,
                    'direction': direction,
                    'power': power,
                    'angle': angle_deg
                }
                candidates.append(candidate)
        
        return candidates
```

---

### 5. ranking/score.py - 샷 점수 계산

```python
"""
샷 평가 및 점수 계산
"""
import numpy as np

class ShotScorer:
    """샷 점수 계산 클래스"""
    
    def __init__(self, config):
        self.weights = config.get('weights', {
            'collision': 0.5,
            'accuracy': 0.3,
            'safety': 0.2
        })
    
    def score(self, simulation_result, candidate):
        """
        샷의 점수를 계산합니다.
        
        Args:
            simulation_result: 물리 시뮬레이션 결과
            candidate: 샷 후보
        
        Returns:
            score: 점수 (0.0 ~ 1.0)
        """
        collision_score = self._score_collision(simulation_result)
        accuracy_score = self._score_accuracy(simulation_result)
        safety_score = self._score_safety(simulation_result)
        
        total_score = (
            self.weights['collision'] * collision_score +
            self.weights['accuracy'] * accuracy_score +
            self.weights['safety'] * safety_score
        )
        
        return total_score
    
    def _score_collision(self, result):
        """충돌 성공 여부 점수"""
        return 1.0 if result['success'] else 0.0
    
    def _score_accuracy(self, result):
        """정확도 점수 (경로 길이 기반)"""
        path = result['path']
        if len(path) < 2:
            return 0.0
        
        # 경로가 짧을수록 높은 점수
        path_length = 0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            path_length += np.sqrt(dx**2 + dy**2)
        
        # 정규화 (임의의 최대값 사용)
        max_length = 500
        return max(0.0, 1.0 - path_length / max_length)
    
    def _score_safety(self, result):
        """안전성 점수 (벽 근접도)"""
        final_pos = result['final_positions'][0]  # 큐볼 최종 위치
        
        # 테이블 중앙으로부터의 거리
        center_x, center_y = 127, 63.5  # 테이블 중심
        distance = np.sqrt((final_pos[0] - center_x)**2 + (final_pos[1] - center_y)**2)
        
        # 정규화
        max_distance = np.sqrt(center_x**2 + center_y**2)
        return 1.0 - distance / max_distance
```

---

### 6. feedback/overlay.py - 시각적 오버레이

```python
"""
시각적 피드백 오버레이
"""
import cv2
import numpy as np

class OverlayRenderer:
    """시각적 오버레이 렌더링 클래스"""
    
    def __init__(self):
        self.colors = {
            'white': (255, 255, 255),
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'other': (0, 255, 0)
        }
    
    def render(self, frame, balls, cue, best_shot):
        """
        전체 피드백을 렌더링합니다.
        
        Args:
            frame: 입력 프레임
            balls: 감지된 공 리스트
            cue: 큐 정보
            best_shot: 최적 샷 (candidate, result)
        
        Returns:
            frame: 오버레이가 그려진 프레임
        """
        overlay = frame.copy()
        
        # 1. 공 표시
        for ball in balls:
            color = self.colors.get(ball['color'], (0, 255, 0))
            cv2.circle(overlay, (ball['x'], ball['y']), ball['r'], color, 2)
            cv2.circle(overlay, (ball['x'], ball['y']), 2, color, -1)
        
        # 2. 최적 샷 경로 표시
        if best_shot:
            candidate, result = best_shot
            path = result['path']
            
            # 경로 선 그리기
            for i in range(1, len(path)):
                pt1 = (int(path[i-1][0]), int(path[i-1][1]))
                pt2 = (int(path[i][0]), int(path[i][1]))
                cv2.line(overlay, pt1, pt2, (0, 255, 255), 2)
            
            # 충돌 지점 표시
            for collision in result['collisions']:
                idx1, idx2, time = collision
                if len(path) > 0:
                    pos_idx = min(int(time * 1000), len(path) - 1)
                    pos = path[pos_idx]
                    cv2.circle(overlay, (int(pos[0]), int(pos[1])), 8, (0, 0, 255), -1)
            
            # 샷 정보 텍스트
            angle = candidate['angle']
            power = candidate['power']
            text = f"Angle: {angle:.1f}° | Power: {power:.2f}"
            cv2.putText(overlay, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 3. 큐 표시
        if cue:
            cv2.line(overlay, cue['start'], cue['end'], (255, 0, 255), 3)
        
        return overlay
    
    def render_balls(self, frame, balls):
        """공만 표시합니다."""
        overlay = frame.copy()
        
        for ball in balls:
            color = self.colors.get(ball['color'], (0, 255, 0))
            cv2.circle(overlay, (ball['x'], ball['y']), ball['r'], color, 2)
            cv2.circle(overlay, (ball['x'], ball['y']), 2, color, -1)
        
        return overlay
```

---

### 7. utils/logger.py - 로깅 설정

```python
"""
로깅 유틸리티
"""
import logging
import os
from datetime import datetime

def setup_logger(name='BilliardCoach', log_dir='logs'):
    """
    로거를 설정합니다.
    
    Args:
        name: 로거 이름
        log_dir: 로그 디렉토리
    
    Returns:
        logger: 설정된 로거
    """
    # 로그 디렉토리 생성
    os.makedirs(log_dir, exist_ok=True)
    
    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # 이미 핸들러가 있으면 제거
    if logger.handlers:
        logger.handlers.clear()
    
    # 파일 핸들러
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'billiard_{timestamp}.log'),
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 포맷 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
```

---

### 8. utils/config.py - 설정 로더

```python
"""
설정 파일 로더
"""
import yaml
import os

def load_config(config_path='config.yaml'):
    """
    YAML 설정 파일을 로드합니다.
    
    Args:
        config_path: 설정 파일 경로
    
    Returns:
        config: 설정 딕셔너리
    """
    if not os.path.exists(config_path):
        # 기본 설정 반환
        return get_default_config()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def get_default_config():
    """기본 설정을 반환합니다."""
    return {
        'vision': {
            'ball_detection': {
                'min_radius': 5,
                'max_radius': 30,
                'confidence_threshold': 0.7
            },
            'cue_detection': {
                'min_length': 50,
                'angle_tolerance': 5
            }
        },
        'physics': {
            'friction_coefficient': 0.02,
            'restitution': 0.95,
            'simulation_steps': 1000,
            'time_delta': 0.001,
            'table_width': 254,
            'table_height': 127
        },
        'ranking': {
            'angle_resolution': 15,
            'power_levels': [0.3, 0.5, 0.7, 1.0],
            'weights': {
                'collision': 0.5,
                'accuracy': 0.3,
                'safety': 0.2
            }
        }
    }
```

---

### 9. config.yaml - 설정 파일

```yaml
vision:
  ball_detection:
    min_radius: 5
    max_radius: 30
    confidence_threshold: 0.7
  
  cue_detection:
    min_length: 50
    angle_tolerance: 5

physics:
  friction_coefficient: 0.02
  restitution: 0.95
  simulation_steps: 1000
  time_delta: 0.001
  table_width: 254
  table_height: 127

ranking:
  angle_resolution: 15
  power_levels: [0.3, 0.5, 0.7, 1.0]
  weights:
    collision: 0.5
    accuracy: 0.3
    safety: 0.2
```

---

### 10. data/storage.py - 데이터 저장

```python
"""
데이터 저장 및 관리
"""
import sqlite3
import json
import os
from datetime import datetime

class DataStorage:
    """로컬 데이터 저장소"""
    
    def __init__(self, db_path='data/billiard.db'):
        """
        데이터베이스를 초기화합니다.
        
        Args:
            db_path: 데이터베이스 파일 경로
        """
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """데이터베이스 테이블을 생성합니다."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS shots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                balls_data TEXT NOT NULL,
                shot_candidate TEXT NOT NULL,
                simulation_result TEXT NOT NULL,
                score REAL NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                total_shots INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_shot(self, balls_data, shot_candidate, simulation_result, score):
        """
        샷 데이터를 저장합니다.
        
        Args:
            balls_data: 공 데이터
            shot_candidate: 샷 후보
            simulation_result: 시뮬레이션 결과
            score: 점수
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT INTO shots (timestamp, balls_data, shot_candidate, simulation_result, score)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            timestamp,
            json.dumps(balls_data),
            json.dumps(shot_candidate),
            json.dumps(simulation_result),
            score
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_shots(self, limit=10):
        """최근 샷 기록을 조회합니다."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM shots
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return rows
```

---

### 11. data/privacy.py - 데이터 익명화

```python
"""
프라이버시 및 데이터 익명화
"""
import hashlib
import json

class PrivacyManager:
    """데이터 프라이버시 관리자"""
    
    def __init__(self):
        self.anonymize_enabled = True
    
    def anonymize_data(self, data):
        """
        데이터를 익명화합니다.
        
        Args:
            data: 원본 데이터
        
        Returns:
            anonymized_data: 익명화된 데이터
        """
        if not self.anonymize_enabled:
            return data
        
        anonymized = data.copy()
        
        # 위치 정보 해싱
        if 'balls' in anonymized:
            for ball in anonymized['balls']:
                ball['x'] = self._hash_value(ball['x'])
                ball['y'] = self._hash_value(ball['y'])
        
        return anonymized
    
    def _hash_value(self, value):
        """값을 해시화합니다."""
        hash_object = hashlib.sha256(str(value).encode())
        return hash_object.hexdigest()[:8]
```

---

### 12. tests/test_vision.py - 비전 모듈 테스트

```python
"""
비전 모듈 테스트
"""
import pytest
import numpy as np
import cv2
from vision.core import BallDetector, CueDetector

def test_ball_detector():
    """공 감지 테스트"""
    config = {
        'min_radius': 5,
        'max_radius': 30,
        'confidence_threshold': 0.7
    }
    
    detector = BallDetector(config)
    
    # 테스트 이미지 생성 (흰 공 하나)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame, (320, 240), 20, (255, 255, 255), -1)
    
    balls = detector.detect(frame)
    
    assert len(balls) > 0
    assert balls[0]['color'] == 'white'

def test_cue_detector():
    """큐 감지 테스트"""
    config = {
        'min_length': 50,
        'angle_tolerance': 5
    }
    
    detector = CueDetector(config)
    
    # 테스트 이미지 생성 (직선)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.line(frame, (100, 100), (500, 300), (255, 255, 255), 5)
    
    balls = []
    cue = detector.detect(frame, balls)
    
    # 큐가 감지되어야 함
    assert cue is not None
```

---

### 13. tests/test_physics.py - 물리 엔진 테스트

```python
"""
물리 엔진 테스트
"""
import pytest
from physics.engine import PhysicsEngine, Ball

def test_physics_engine():
    """물리 엔진 기본 테스트"""
    config = {
        'friction_coefficient': 0.02,
        'restitution': 0.95,
        'simulation_steps': 100,
        'time_delta': 0.01,
        'table_width': 254,
        'table_height': 127
    }
    
    engine = PhysicsEngine(config)
    
    # 테스트 데이터
    balls_data = [
        {'x': 50, 'y': 50, 'r': 10, 'color': 'white'},
        {'x': 150, 'y': 150, 'r': 10, 'color': 'red'}
    ]
    
    shot_candidate = {
        'cue_ball': 0,
        'direction': (1.0, 0.0),
        'power': 50.0
    }
    
    result = engine.simulate(balls_data, shot_candidate)
    
    assert 'path' in result
    assert 'success' in result
    assert len(result['path']) > 0
```
### 15. feedback/text.py - 텍스트 피드백 생성 (의사코드)

```python
"""
텍스트 기반 피드백 생성 - 알고리즘만 제시
"""

class TextFeedbackGenerator:
    """텍스트 피드백 생성기"""
    
    def generate_success_feedback(self, shot_result):
        """
        성공 시 피드백 생성
        
        알고리즘:
        1. 샷의 정확도 점수 계산 (0-100%)
        2. 우수한 부분 식별 (각도, 파워, 타이밍)
        3. 칭찬 메시지 생성
        4. 다음 단계 제안
        """
        pass
    
    def generate_failure_feedback(self, analysis_result):
        """
        실패 시 피드백 생성
        
        알고리즘:
        1. FailureAnalyzer의 분석 결과 수신
        2. 주요 원인을 사용자 친화적 언어로 변환
        3. 단계별 개선 가이드 생성
        4. 관련 연습 방법 제안
        """
        pass
    
    def generate_progress_feedback(self, history):
        """
        진행 상황 피드백
        
        알고리즘:
        1. 최근 N개 샷의 통계 분석
        2. 향상된 부분과 정체된 부분 식별
        3. 성취도 리포트 생성
        4. 맞춤형 연습 계획 제안
        """
        pass
```

---

### 16. main.py 업데이트 - 실패 분석 통합 (완전 구현)

```python
"""
메인 실행 파일 - 실패 분석 기능 추가
"""
import argparse
import cv2
import numpy as np
from vision.core import BallDetector, CueDetector
from physics.engine import PhysicsEngine
from ranking.generator import ShotGenerator
from ranking.score import ShotScorer
from feedback.overlay import OverlayRenderer
from feedback.failure_analyzer import FailureAnalyzer  # 추가
from utils.logger import setup_logger
from utils.config import load_config

def main():
    parser = argparse.ArgumentParser(description='Billiard Shot Coaching System')
    parser.add_argument('--mode', choices=['vision', 'full', 'practice'], default='full',
                        help='실행 모드 선택')
    parser.add_argument('--camera', type=int, default=0,
                        help='카메라 디바이스 ID')
    parser.add_argument('--input', type=str, default=None,
                        help='입력 비디오 파일 경로')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='설정 파일 경로')
    parser.add_argument('--enable-failure-analysis', action='store_true',
                        help='실패 원인 분석 활성화')
    
    args = parser.parse_args()
    
    # 로거 및 설정 로드
    logger = setup_logger()
    config = load_config(args.config)
    
    # 모듈 초기화
    ball_detector = BallDetector(config['vision']['ball_detection'])
    cue_detector = CueDetector(config['vision']['cue_detection'])
    physics_engine = PhysicsEngine(config['physics'])
    shot_generator = ShotGenerator(config['ranking'])
    shot_scorer = ShotScorer(config['ranking'])
    overlay_renderer = OverlayRenderer()
    
    # 실패 분석기 초기화 (선택적)
    failure_analyzer = None
    if args.enable_failure_analysis or args.mode == 'practice':
        failure_analyzer = FailureAnalyzer(config.get('failure_analysis', {}))
        logger.info("실패 원인 분석 모드 활성화")
    
    # 비디오 소스 설정
    if args.input:
        cap = cv2.VideoCapture(args.input)
    else:
        cap = cv2.VideoCapture(args.camera)
    
    logger.info(f"시스템 시작 - 모드: {args.mode}")
    
    # 샷 추적을 위한 변수
    previous_balls = None
    shot_in_progress = False
    expected_result = None
    shot_candidate = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1. 공 감지
        current_balls = ball_detector.detect(frame)
        
        if args.mode == 'full' and len(current_balls) > 0:
            # 2. 큐 감지
            cue = cue_detector.detect(frame, current_balls)
            
            # 샷 시작 감지 (큐가 사라졌을 때)
            if previous_balls and cue is None and not shot_in_progress:
                shot_in_progress = True
                
                # 3. 샷 후보 생성 및 최적 샷 예측
                candidates = shot_generator.generate(current_balls, None)
                
                best_shot = None
                best_score = -1
                
                for candidate in candidates[:10]:
                    result = physics_engine.simulate(current_balls, candidate)
                    score = shot_scorer.score(result, candidate)
                    
                    if score > best_score:
                        best_score = score
                        best_shot = (candidate, result)
                
                if best_shot:
                    shot_candidate, expected_result = best_shot
                    logger.info(f"샷 시작 감지 - 예상 점수: {best_score:.2f}")
            
            # 샷 종료 감지 (모든 공이 정지했을 때)
            if shot_in_progress and self._balls_stopped(current_balls, previous_balls):
                shot_in_progress = False
                
                # 실패 분석 수행
                if failure_analyzer and expected_result:
                    # 실제 결과 구성
                    actual_result = {
                        'path': self._extract_cue_ball_path(previous_balls, current_balls),
                        'collisions': self._detect_collisions(previous_balls, current_balls),
                        'final_positions': [(b['x'], b['y']) for b in current_balls]
                    }
                    
                    # 득점 여부 확인
                    success = self._check_scoring(expected_result, actual_result)
                    
                    if not success:
                        # 실패 원인 분석
                        analysis = failure_analyzer.analyze_failure(
                            expected_result,
                            actual_result,
                            shot_candidate
                        )
                        
                        # 피드백 메시지 생성 및 표시
                        feedback_msg = failure_analyzer.generate_feedback_message(analysis)
                        logger.info(f"\n실패 분석 결과:\n{feedback_msg}")
                        
                        # 화면에 피드백 표시
                        frame = self._display_failure_feedback(frame, feedback_msg, analysis)
                    else:
                        logger.info("✅ 득점 성공!")
                        frame = self._display_success_message(frame)
            
            # 4. 시각적 피드백
            if expected_result and shot_candidate:
                frame = overlay_renderer.render(frame, current_balls, cue, 
                                                (shot_candidate, expected_result))
        else:
            # Vision 모드: 공만 표시
            frame = overlay_renderer.render_balls(frame, current_balls)
        
        # 화면 표시
        cv2.imshow('Billiard Coaching System', frame)
        
        # 이전 프레임 저장
        previous_balls = current_balls
        
        # 'q' 키로 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    logger.info("시스템 종료")

    def _balls_stopped(self, current, previous, threshold=2.0):
        """공들이 정지했는지 확인"""
        if not previous or len(current) != len(previous):
            return False
        
        for curr, prev in zip(current, previous):
            distance = np.sqrt((curr['x'] - prev['x'])**2 + (curr['y'] - prev['y'])**2)
            if distance > threshold:
                return False
        return True
    
    def _extract_cue_ball_path(self, start_balls, end_balls):
        """큐볼의 이동 경로 추출 - 의사코드"""
        # TODO: 프레임별 큐볼 위치 추적 필요
        # 현재는 시작과 끝 위치만 반환
        path = []
        # ... 구현 필요
        return path
    
    def _detect_collisions(self, start_balls, end_balls):
        """충돌 감지 - 의사코드"""
        # TODO: 프레임별 분석으로 충돌 지점 감지
        collisions = []
        # ... 구현 필요
        return collisions
    
    def _check_scoring(self, expected, actual):
        """득점 여부 확인 - 의사코드"""
        # TODO: 포켓에 들어간 공 확인
        # ... 구현 필요
        return False  # 현재는 항상 실패로 가정
    
    def _display_failure_feedback(self, frame, message, analysis):
        """실패 피드백을 화면에 표시"""
        overlay = frame.copy()
        
        # 반투명 배경
        cv2.rectangle(overlay, (10, 10), (600, 250), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 메시지 표시
        y_offset = 40
        for line in message.split('\n'):
            cv2.putText(frame, line, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 30
        
        return frame
    
    def _display_success_message(self, frame):
        """성공 메시지 표시"""
        cv2.putText(frame, "SUCCESS!", (frame.shape[1]//2 - 100, frame.shape[0]//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        return frame

if __name__ == '__main__':
    main()
```

---

### 17. 실패 분석 결과 예시

```
❌ 파워가 25.3% 부족했습니다

추가 문제점:
  • 큐 각도가 8.5° 벗어났습니다
  • 충돌 지점이 3.2cm 벗어났습니다

💡 개선 방안:
  💪 스트로크를 약 25% 더 강하게 해보세요
  💡 큐를 8.5° 더 오른쪽으로 향하도록 조정하세요
  📍 충돌 지점을 3.2cm 조정해야 합니다
```

---

### 18. config.yaml 업데이트 - 실패 분석 설정 추가

```yaml
vision:
  ball_detection:
    min_radius: 5
    max_radius: 30
    confidence_threshold: 0.7
  
  cue_detection:
    min_length: 50
    angle_tolerance: 5

physics:
  friction_coefficient: 0.02
  restitution: 0.95
  simulation_steps: 1000
  time_delta: 0.001
  table_width: 254
  table_height: 127

ranking:
  angle_resolution: 15
  power_levels: [0.3, 0.5, 0.7, 1.0]
  weights:
    collision: 0.5
    accuracy: 0.3
    safety: 0.2

failure_analysis:
  enabled: true
  thresholds:
    angle_error: 10.0      # 각도 오차 임계값 (도)
    power_shortage: 0.2    # 파워 부족 비율
    power_excess: 0.3      # 파워 과다 비율
    near_miss: 5.0         # 근접 실패 거리 (cm)
    cushion_error: 15.0    # 쿠션 각도 오차
  
  feedback:
    show_visual_hints: true    # 시각적 힌트 표시
    voice_feedback: false      # 음성 피드백 (미구현)
    save_failure_logs: true    # 실패 로그 저장
```

---

### 19. 실패 분석 알고리즘 상세 설명

#### 🔍 분석 프로세스

```
1. 데이터 수집 단계
   ┌─────────────────────────────────┐
   │ - 예상 결과 (시뮬레이션)        │
   │ - 실제 결과 (카메라 추적)       │
   │ - 샷 입력 정보                  │
   └─────────────────────────────────┘
                 ↓
2. 비교 분석 단계
   ┌─────────────────────────────────┐
   │ ① 각도 오차 계산                │
   │ ② 파워 오차 계산                │
   │ ③ 충돌 지점 차이 분석           │
   │ ④ 경로 편차 분석                │
   │ ⑤ 최종 위치 비교                │
   └─────────────────────────────────┘
                 ↓
3. 원인 도출 단계
   ┌─────────────────────────────────┐
   │ - 각 오차의 심각도 점수화       │
   │ - 주요 원인 / 부차 원인 분류    │
   │ - 오차 간 상관관계 분석         │
   └─────────────────────────────────┘
                 ↓
4. 피드백 생성 단계
   ┌─────────────────────────────────┐
   │ - 사용자 친화적 메시지 생성     │
   │ - 구체적 개선 방안 제시         │
   │ - 시각적 가이드 생성            │
   └─────────────────────────────────┘
```

#### 📊 원인 분석 알고리즘

**1. 각도 오차 분석**
```
입력: 예상 각도, 실제 이동 벡터
처리:
  - 실제 이동 방향을 벡터로 계산
  - arctan2로 각도 변환
  - 예상 각도와의 차이 계산
출력: 각도 오차 (도), 조정 방향
```

**2. 파워 오차 분석**
```
입력: 예상 경로, 실제 경로
처리:
  - 각 경로의 총 이동거리 계산
  - 거리 비율로 파워 오차 추정
  - 임계값과 비교하여 부족/과다 판단
출력: 파워 오차 비율, 조정량
```

**3. 충돌 지점 분석**
```
입력: 예상 충돌점, 실제 충돌점
처리:
  - 유클리드 거리 계산
  - 충돌 각도 비교
  - 충돌 실패 여부 확인
출력: 거리 오차, 각도 오차
```

**4. 쿠션 활용 분석 (미구현 - 알고리즘)**
```
알고리즘:
  1. 쿠션 충돌 횟수 카운트
     - 예상: N회
     - 실제: M회
  
  2. 각 쿠션 충돌의 입사각/반사각 계산
     - 이상적 반사각 = 180° - 입사각
     - 실제 반사각과 비교
  
  3. 쿠션 활용 실패 유형 분류
     - Type A: 쿠션 미도달
     - Type B: 쿠션 각도 오차
     - Type C: 쿠션 과도 충돌
  
  4. 원인 파악
     - 파워 부족 → Type A
     - 각도 오차 → Type B
     - 파워 과다 → Type C
```

**5. 스핀 효과 분석 (미구현 - 알고리즘)**
```
알고리즘:
  1. 타점(Impact Point) 추정
     - 큐볼 중심으로부터의 타격 위치
     - Top/Bottom/Left/Right spin 판단
  
  2. 스핀으로 인한 경로 변화 예측
     - Top spin: 쿠션 후 가속
     - Back spin: 쿠션 후 감속/역회전
     - Side spin: 쿠션 각도 변화
  
  3. 예상 vs 실제 스핀 효과 비교
     - 경로 곡률 비교
     - 쿠션 반사 각도 비교
  
  4. 스핀 오차 판단
     - 과도한 스핀
     - 불충분한 스핀
     - 잘못된 스핀 방향
```

---

### 20. 실패 분석 시각화 (의사코드)

```python
"""
실패 원인을 시각적으로 표시하는 알고리즘
"""

class FailureVisualizer:
    """실패 원인 시각화 클래스"""
    
    def visualize_angle_error(self, frame, analysis):
        """
        각도 오차 시각화
        
        알고리즘:
        1. 예상 각도를 녹색 화살표로 표시
        2. 실제 각도를 빨간색 화살표로 표시
        3. 두 화살표 사이에 호(arc)로 오차 각도 표시
        4. 수정 방향을 점선 화살표로 가이드
        """
        pass
    
    def visualize_power_error(self, frame, analysis):
        """
        파워 오차 시각화
        
        알고리즘:
        1. 예상 경로를 녹색 선으로 표시
        2. 실제 경로를 빨간색 선으로 표시
        3. 경로 끝지점에 게이지 바 표시
           - 부족: 빨간색 (↓)
           - 과다: 주황색 (↑)
        4. 필요한 파워 조정량을 텍스트로 표시
        """
        pass
    
    def visualize_collision_miss(self, frame, analysis):
        """
        충돌 미스 시각화
        
        알고리즘:
        1. 목표 충돌 지점을 녹색 원으로 표시
        2. 실제 큐볼 경로를 추적선으로 표시
        3. 목표공과 최단거리 지점 표시
        4. 보정 방향을 화살표로 가이드
        """
        pass
    
    def overlay_improvement_guide(self, frame, recommendations):
        """
        개선 가이드 오버레이
        
        알고리즘:
        1. 화면 상단에 반투명 패널 생성
        2. 주요 원인을 아이콘과 함께 표시
        3. 개선 방안을 번호 순서대로 나열
        4. 심각도에 따라 색상 구분
           - Low: 초록색
           - Medium: 노란색
           - High: 빨간색
        """
        pass
```

---

### 21. 연습 모드 구현 (알고리즘)

```python
"""
반복 연습 및 학습 모드
"""

class PracticeMode:
    """연습 모드 관리 클래스"""
    
    def __init__(self):
        self.shot_history = []
        self.improvement_tracker = {}
    
    def run_practice_session(self, target_skill):
        """
        특정 기술 집중 연습
        
        알고리즘:
        1. 목표 설정
           - 각도 정확도 향상
           - 파워 조절 연습
           - 쿠션 샷 연습
        
        2. 반복 시도
           for each attempt:
               - 샷 실행
               - 실시간 피드백
               - 성공/실패 기록
        
        3. 진행 상황 분석
           - 시도별 점수 변화 그래프
           - 주요 오차 항목 통계
           - 개선 추세 계산
        
        4. 맞춤형 조언
           - 취약점 기반 연습 과제 제안
           - 다음 난이도 단계 추천
        """
        pass
    
    def analyze_progress(self, history):
        """
        학습 진행도 분석
        
        알고리즘:
        1. 최근 N개 샷의 통계 수집
           - 평균 점수
           - 성공률
           - 주요 오차 빈도
        
        2. 시간대별 성능 변화 추적
           - 초기 vs 최근 비교
           - 학습 곡선 그래프
        
        3. 강점/약점 분석
           - 잘하는 샷 유형 식별
           - 반복되는 실수 패턴 발견
        
        4. 목표 달성도 평가
           - 설정한 목표 대비 현재 수준
           - 다음 마일스톤까지 필요한 연습량
        """
        pass
    
    def generate_custom_drill(self, weakness):
        """
        맞춤형 연습 과제 생성
        
        알고리즘:
        1. 약점 식별
           - 각도 조절 약함
           - 파워 조절 약함
           - 쿠션 활용 약함
        
        2. 난이도 조정
           - 현재 수준 평가
           - 적정 도전 과제 선정
        
        3. 연습 시나리오 구성
           - 공 배치 설정
           - 목표 달성 조건
           - 제한 시간/시도 횟수
        
        4. 점진적 난이도 증가
           - 성공 시 더 어려운 과제
           - 실패 시 기본으로 복귀
        """
        pass
```

---

### 22. 데이터 기반 학습 개선 (미래 기능 - 컨셉)

```
┌─────────────────────────────────────────┐
│      머신러닝 기반 성능 예측 모델       │
├─────────────────────────────────────────┤
│                                         │
│  입력: 사용자의 과거 샷 데이터          │
│    - 각도 오차 패턴                     │
│    - 파워 조절 경향                     │
│    - 자주 하는 실수 유형                │
│                                         │
│  처리: ML 모델                          │
│    - 패턴 인식                          │
│    - 예측 모델링                        │
│    - 클러스터링                         │
│                                         │
│  출력: 맞춤형 코칭                      │
│    - 개인화된 난이도 조정               │
│    - 예측 기반 사전 경고                │
│    - 최적 학습 경로 제안                │
│                                         │
└─────────────────────────────────────────┘
```

---

## 🖼️ 결과 화면 예시

시스템이 실행되면 다음과 같은 화면이 표시됩니다:

### 성공 시 화면
```
┌─────────────────────────────────────────────┐
│  Billiard Coaching System                   │
│  Angle: 45.0° | Power: 0.70 | Score: 0.85   │
├─────────────────────────────────────────────┤
│                                             │
│     ●  ← 흰공 (큐볼)                        │
│      \                                      │
│       \  ← 예상 경로 (노란색 선)             │
│        \                                    │
│         ●  ← 목표공 (빨간색)                 │
│                                             │
│         ⊗  ← 충돌 지점                       │
│                                             │
│    ━━━━  ← 큐 (보라색 선)                    │
│                                             │
│  ✅ SUCCESS! 득점 성공                       │
│                                             │
└─────────────────────────────────────────────┘
```

### 실패 시 화면 (피드백 포함)
```
┌─────────────────────────────────────────────┐
│  Billiard Coaching System                   │
│  Angle: 48.5° | Power: 0.55 | Score: 0.32   │
├─────────────────────────────────────────────┤
│                                             │
│  ❌ 실패 원인 분석                           │
│  ═══════════════════════════════════════    │
│                                             │
│  주요 원인:                                  │
│  💪 파워가 25.3% 부족했습니다               │
│                                             │
│  추가 문제:                                  │
│  • 큐 각도가 8.5° 벗어났습니다              │
│  • 충돌 지점이 3.2cm 벗어났습니다           │
│                                             │
│  💡 개선 방안:                               │
│  1. 스트로크를 25% 더 강하게 해보세요       │
│  2. 큐를 8.5° 더 오른쪽으로 조정하세요      │
│  3. 조준을 3.2cm 왼쪽으로 이동하세요        │
│                                             │
└─────────────────────────────────────────────┘

[로그 출력]
2024-01-15 10:30:45 - INFO - 시스템 시작
2024-01-15 10:30:46 - INFO - 공 3개 감지됨
2024-01-15 10:30:47 - INFO - 샷 실행 감지
2024-01-15 10:30:48 - WARN - 득점 실패
2024-01-15 10:30:48 - INFO - 실패 원인 분석 완료
```

---

## 🗺️ 개발 로드맵

| 단계 | 주요 목표 | 세부 내용 |
|------|-----------|-----------|
| 1️⃣ Prototype 구축 | 기본 구조 완성 | 공 추적 MVP, 큐 각도 추정 |
| 2️⃣ Physics Engine 고도화 | 물리 시뮬레이션 정확도 향상 | 충돌, 마찰, 쿠션 반사 구현 |
| 3️⃣ Ranking System 구현 | 샷 후보 생성 및 평가 | 점수화 알고리즘 및 최적화 |
| 4️⃣ Feedback UI 통합 | 시각적 오버레이 추가 | 경로 예측 및 실시간 렌더링 |
| 5️⃣ **실패 분석 시스템** ✨ | **원인 진단 및 피드백** | **각도/파워/충돌 오차 분석, 개선 방안 제시** |
| 6️⃣ 프라이버시 강화 | 로컬 저장 및 익명화 | 데이터 보호 정책 구현 |
| 7️⃣ 연습 모드 추가 | 학습 진행도 추적 | 맞춤형 연습 과제, 통계 분석 |
| 8️⃣ 다중 플랫폼 지원 | 웹/데스크탑/모바일 확장 | React 기반 웹 UI 개발 |
| 9️⃣ 성능 최적화 | 실시간 처리 개선 | GPU 가속, 프레임레이트 향상 |
| 🔟 커뮤니티 공개 | 오픈소스 배포 | 문서화, API/SDK 공개 |# 🎱 Camera-Based Real-time Billiard Shot Coaching System
---

## 🏗️ 아키텍처

### 시스템 구성도

```
┌─────────────┐
│   Camera    │
└──────┬──────┘
       │
       ▼
┌──────────────────────┐
│   Vision Module      │
│  - Ball Detection    │
│  - Cue Detection     │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Physics Simulation   │
│  - Collision         │
│  - Friction & Spin   │
│  - Cushion Bounce    │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Ranking Engine      │
│  - Generate Shots    │
│  - Score & Rank      │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Feedback Engine     │
│  - Visual Overlay    │
│  - Text Feedback     │
└──────────────────────┘
```

### 데이터 흐름

1. **카메라 입력** → 실시간 프레임 캡처
2. **비전 모듈** → 공/큐 위치 감지 및 추적
3. **샷 생성** → 가능한 샷 후보 생성 (각도 × 파워)
4. **물리 시뮬레이션** → 각 후보에 대한 경로 예측
5. **점수 계산** → 충돌/정확도/안전성 기반 평가
6. **피드백 렌더링** → 최적 샷 시각화 및 표시

---

## 🖼️ 결과 화면 예시

시스템이 실행되면 다음과 같은 화면이 표시됩니다:

```
┌─────────────────────────────────────────────┐
│  Billiard Coaching System                   │
│  Angle: 45.0° | Power: 0.70                 │
├─────────────────────────────────────────────┤
│                                             │
│     ●  ← 흰공 (큐볼)                        │
│      \                                      │
│       \  ← 예상 경로 (노란색 선)             │
│        \                                    │
│         ●  ← 목표공 (빨간색)                 │
│                                             │
│         ⊗  ← 충돌 지점                       │
│                                             │
│    ━━━━  ← 큐 (보라색 선)                    │
│                                             │
└─────────────────────────────────────────────┘

[로그 출력]
2024-01-15 10:30:45 - INFO - 시스템 시작 - 모드: full
2024-01-15 10:30:46 - INFO - 공 3개 감지됨
2024-01-15 10:30:46 - INFO - 최적 샷 계산 완료 (점수: 0.85)
```

### 화면 구성 요소

- **흰색 원**: 큐볼 (흰 공)
- **빨간색/파란색 원**: 목표공들
- **노란색 선**: 예상 충돌 경로
- **빨간색 점**: 충돌 예상 지점
- **보라색 선**: 큐 스틱 위치
- **상단 텍스트**: 추천 각도 및 파워

---


## 🤝 기여하기

기여를 환영합니다! 다음 절차를 따라주세요:

1. 이 저장소를 Fork 합니다
2. Feature 브랜치를 생성합니다 (`git checkout -b feature/AmazingFeature`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치에 Push 합니다 (`git push origin feature/AmazingFeature`)
5. Pull Request를 생성합니다

### 기여 가능한 영역

- ✅ 이미 구현된 기능 개선
  - 공/큐 감지 정확도 향상
  - 물리 엔진 최적화
  - 실패 분석 알고리즘 고도화
  
- 🔨 구현 필요 (의사코드만 제공됨)
  - 쿠션 활용 분석 (`feedback/failure_analyzer.py`)
  - 스핀 효과 분석 (`feedback/failure_analyzer.py`)
  - 실패 시각화 (`feedback/failure_visualizer.py`)
  - 연습 모드 (`practice/practice_mode.py`)
  - 학습 진행도 추적 시스템


자세한 내용은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참조하세요.

---

## 📄 라이선스

본 프로젝트의 코드는 **MIT License**를 따릅니다.

---

## 💡 향후 개발 방향

### 단기 목표 (1개월)
- [ ] 쿠션 반사 완전 구현
- [ ] 스핀 효과 시뮬레이션
- [ ] 연습 모드 베타 버전
- [ ] 성능 최적화 (60fps 안정화)

### 중기 목표 (1년)
- [ ] 웹 기반 UI 개발
- [ ] 머신러닝 기반 샷 예측
- [ ] 다양한 게임 룰 지원
- [ ] 모바일 앱 프로토타입

### 장기 목표 (1년+)
- [ ] AR 기반 실시간 가이드
- [ ] 온라인 코칭 플랫폼
- [ ] 프로 선수 데이터 분석
- [ ] 국제 대회 활용

---

## 🙏 참고 자료

### 관련 프로젝트
- [Open Shot Detector](https://github.com/honey-da/osd.git) - 멀티 AI 모델 라우팅 시스템

### 기술 문서
- [OpenCV 문서](https://docs.opencv.org/)
- [당구 물리학](https://billiards.colostate.edu/)
- [Hough Transform](https://en.wikipedia.org/wiki/Hough_transform)

### 논문 및 연구
- "Computer Vision for Billiard Analysis" (2023)
- "Real-time Physics Simulation for Sports Coaching" (2022)

---

## 🎓 사용 예시 및 튜토리얼

### 기본 사용법
```bash
# 1. 저장소 클론
git clone https://github.com/your-username/billiard-coaching-system.git
cd billiard-coaching-system

# 2. 환경 설정
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. 기본 실행
python main.py --mode full --camera 0

# 4. 실패 분석 모드로 실행
python main.py --mode practice --enable-failure-analysis

# 5. 비디오 파일로 테스트
python main.py --mode full --input examples/sample_videos/test.mp4
```

### 고급 설정
```bash
# 커스텀 설정 파일 사용
python main.py --config my_config.yaml

# 디버그 모드
DEBUG_MODE=true python main.py --mode full
```

---

## ⚠️ 알려진 제한사항

현재 버전의 제한사항:

- 조명 조건에 따라 공 감지 정확도가 변동될 수 있습니다
- 복잡한 스핀 효과는 아직 시뮬레이션되지 않습니다
- 실시간 경로 추적 기능이 미구현 상태입니다 (프레임 간 보간 필요)
- 쿠션 반사 물리 모델이 단순화되어 있습니다

개선 계획은 [로드맵](#-개발-로드맵)을 참조하세요.

---

## 🔧 트러블슈팅

### 공이 감지되지 않는 경우
```yaml
# config.yaml 조정
vision:
  ball_detection:
    min_radius: 3        # 더 작은 값으로 조정
    max_radius: 40       # 더 큰 값으로 조정
    confidence_threshold: 0.5  # 낮춰서 감도 증가
```

### FPS가 낮은 경우
```bash
# 해상도 낮추기
python main.py --camera 0 --resolution 640x480

# 시뮬레이션 스텝 줄이기
# config.yaml에서 simulation_steps: 500 으로 조정
```

### 실패 분석이 작동하지 않는 경우
```bash
# 명시적으로 활성화
python main.py --mode practice --enable-failure-analysis

# 로그 확인
tail -f logs/billiard_*.log
```

---
