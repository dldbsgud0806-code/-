# 🎱 Camera-Based Real-time Billiard Shot Coaching System

> 카메라를 활용한 실시간 당구 샷 분석 및 코칭 시스템

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-alpha-orange)]()
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

---

## 📚 목차

- [소개](#-소개)
- [특징](#-특징)
- [설치](#-설치)
- [빠른 시작](#-빠른-시작)
- [사용법](#-사용법)
- [설정](#-설정)
- [프로젝트 구조](#-프로젝트-구조)

---

## 🎯 소개

본 프로젝트는 카메라를 활용하여 당구대 위의 공과 큐의 위치 및 각도를 실시간으로 추적하고, 물리 모델 기반 시뮬레이션을 통해 최적의 샷 후보를 제시하는 오픈소스 코칭 시스템입니다.

### 핵심 목표

- 실시간 공 위치 및 큐 각도 추적 (30-60fps)
- 물리 기반 경로 예측 및 최적 샷 추천
- 득점 실패 시 원인 분석 및 개선 방안 제시
- 시각적 피드백을 통한 학습 효과 극대화

---

## ✨ 특징

### 실시간 분석
- 30-60fps로 공의 위치와 큐의 각도를 추적
- OpenCV 기반 실시간 영상 처리

### 물리 기반 시뮬레이션
- 충돌, 마찰, 쿠션 반사를 정확히 계산
- 2D 물리 엔진으로 경로 예측

### 시각적 피드백
- 예상 경로를 화면에 오버레이로 표시
- 최적 타점과 큐 각도 실시간 안내

### 실패 원인 분석 ⭐
- 득점 실패 시 자동 원인 진단
  - 각도 오차 분석 (예: 8.5° 벗어남)
  - 파워 오차 분석 (예: 25% 부족)
  - 충돌 지점 분석 (예: 3.2cm 오차)
- 구체적인 개선 방안 제시
  - "스트로크를 25% 더 강하게"
  - "큐를 8.5° 오른쪽으로 조정"

### 프라이버시 보호
- 모든 데이터는 로컬에서 처리
- 익명화 옵션 제공

---

## 🚀 설치

### 시스템 요구사항

```
- Python 3.8 이상
- OpenCV 4.5 이상
- NumPy 1.20 이상
```

### 설치 방법

```bash
# 저장소 클론
git clone https://github.com/your-username/billiard-coaching-system.git
cd billiard-coaching-system

# 가상환경 생성 (Windows)
python -m venv venv
venv\Scripts\activate

# 가상환경 생성 (Mac/Linux)
python3 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### requirements.txt

```txt
opencv-python>=4.5.0
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.3.0
pandas>=1.3.0
python-dotenv>=0.19.0
pyyaml>=5.4.0
pytest>=6.2.0
```

---

## ⚡ 빠른 시작

### 기본 실행

```bash
# 전체 코칭 시스템 실행
python main.py --mode full --camera 0

# 비전 모듈만 테스트
python main.py --mode vision --input test_video.mp4

# 실패 분석 모드
python main.py --mode practice --enable-failure-analysis
```

### 실행 화면

```
[2024-01-15 10:30:45] INFO - 시스템 시작 - 모드: full
[2024-01-15 10:30:46] INFO - 공 3개 감지됨
[2024-01-15 10:30:47] INFO - 최적 샷 계산 완료 (점수: 0.85)
```

---

## 📖 사용법

### 1️⃣ 비디오 파일 분석

```bash
python main.py --mode vision --input sample_video.mp4
```

### 2️⃣ 실시간 카메라 코칭

```bash
python main.py --mode full --camera 0
```

### 3️⃣ 결과 확인

**성공 시**
- 예상 경로와 실제 결과 비교
- 성공 메시지 표시

**실패 시**
- 실패 원인 자동 분석
- 구체적 개선 방안 제시

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

### 4️⃣ 종료

- OpenCV 윈도우에서 `q` 키 누르기
- 터미널에서 `Ctrl + C`
- 가상환경 종료: `deactivate`

---

## ⚙️ 설정

### config.yaml

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
    angle_error: 10.0
    power_shortage: 0.2
    power_excess: 0.3
```

---

## 📁 프로젝트 구조

```
billiard-coaching-system/
│
├── main.py                    # 메인 실행 파일
├── requirements.txt           # 의존성
├── config.yaml               # 설정 파일
│
├── vision/                   # 비전 모듈
│   ├── core.py              # 공/큐 감지
│   └── tracking.py          # 객체 추적
│
├── physics/                 # 물리 시뮬레이션
│   ├── engine.py           # 시뮬레이션 엔진
│   └── collision.py        # 충돌 처리
│
├── ranking/                # 샷 평가
│   ├── generator.py       # 후보 생성
│   └── score.py           # 점수 계산
│
├── feedback/               # 피드백
│   ├── overlay.py         # 시각적 오버레이
│   └── failure_analyzer.py # 실패 분석
│
├── utils/                 # 유틸리티
│   ├── logger.py         # 로깅
│   └── config.py         # 설정 로더
│
└── examples/             # 예제
    └── sample_videos/    # 샘플 비디오
```

---

## 🏗️ 시스템 아키텍처

### 전체 구조

```
┌─────────────────┐
│    Camera       │  카메라 입력
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vision Module   │  공/큐 감지
│ - Ball Detection│
│ - Cue Detection │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Physics Engine   │  물리 시뮬레이션
│ - Collision     │
│ - Friction      │
│ - Cushion       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Ranking Engine  │  샷 평가
│ - Generate      │
│ - Score & Rank  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Feedback Engine  │  피드백
│ - Visual        │
│ - Analysis      │
└─────────────────┘
```

---

## 🔍 핵심 알고리즘

### 1. 공 감지 알고리즘

```
입력: 카메라 프레임 (RGB 이미지)

처리:
1. Grayscale 변환
2. Gaussian Blur로 노이즈 제거
3. Hough Circle Transform으로 원형 객체 감지
4. 색상 분석으로 공 종류 분류 (흰색/빨간색/파란색)

출력: [(x, y, radius, color), ...]
```

### 2. 큐 감지 알고리즘

```
입력: 카메라 프레임

처리:
1. Edge Detection (Canny)
2. Hough Line Transform으로 직선 감지
3. 가장 긴 직선을 큐로 판단
4. 큐 각도 계산

출력: (start_point, end_point, angle)
```

### 3. 물리 시뮬레이션

```
입력: 공 위치, 샷 각도, 파워

처리:
for each time_step:
    1. 속도 업데이트 (마찰력 적용)
    2. 위치 업데이트
    3. 충돌 감지
        - 공-공 충돌
        - 공-벽 충돌
    4. 충돌 처리 (운동량 보존)
    
출력: 경로 좌표 리스트, 충돌 지점
```

### 4. 실패 원인 분석 알고리즘

```
입력: 예상 결과, 실제 결과, 샷 정보

분석:
1. 각도 오차 계산
   - 실제 이동 방향 계산
   - 예상 각도와 비교
   - 오차가 임계값(10°) 초과 시 원인으로 식별

2. 파워 오차 계산
   - 경로 총 길이 비교
   - 부족/과다 판단 (±20% 이상)

3. 충돌 지점 오차
   - 예상 vs 실제 충돌 지점 거리
   - 5cm 이상 차이 시 원인으로 식별

4. 심각도 평가
   - 각 오차에 점수 부여
   - 주요 원인 선정

출력: 
- 주요 원인
- 부차적 원인들
- 개선 방안 리스트
- 심각도 (low/medium/high)
```

### 5. 샷 후보 생성

```
입력: 현재 공 배치

처리:
for angle in [0°, 15°, 30°, ..., 345°]:
    for power in [0.3, 0.5, 0.7, 1.0]:
        candidate = {
            angle: angle,
            power: power
        }
        candidates.append(candidate)

출력: 96개의 샷 후보
```

---

## 🖼️ 결과 화면

### 시스템 데모 화면

<p align="center">
  <img src="https://via.placeholder.com/800x400/0f7a4a/ffffff?text=Billiard+Coaching+System+Demo" alt="시스템 데모 화면" width="800">
</p>

*실제 구현 시 당구대, 공, 예상 경로가 실시간으로 화면에 오버레이됩니다.*

### 화면 구성

| 요소 | 설명 | 색상 |
|------|------|------|
| 🔵 큐볼 | 흰 공 (플레이어가 치는 공) | 흰색 원 |
| 🔴 목표공 | 맞춰야 할 공들 | 빨강/파랑 원 |
| 📏 예상 경로 | 최적 샷의 이동 경로 | 노란색 선 |
| 🎯 충돌 지점 | 공이 부딪히는 위치 | 빨간색 점 |
| 🎱 큐 스틱 | 플레이어의 큐 위치 | 보라색 선 |
| 📊 정보 패널 | 각도, 파워, 점수 | 상단 텍스트 |

---

## 📊 개발 로드맵

| 단계 | 기능 | 상태 |
|------|------|------|
| 1️⃣ | 공 추적 MVP 
| 2️⃣ | 큐 각도 추정 
| 3️⃣ | 물리 시뮬레이션 
| 4️⃣ | 샷 후보 생성/평가 
| 5️⃣ | **실패 원인 분석** 
| 6️⃣ | 시각적 오버레이 
| 7️⃣ | 쿠션 반사 고도화 
| 8️⃣ | 스핀 효과 구현 
| 9️⃣ | 웹 UI 개발 
| 🔟 | 모바일 앱 

---

## 🤝 기여하기

이 프로젝트는 오픈소스로 운영되며 커뮤니티의 기여를 환영합니다.

### 기여 방법

1. 저장소 Fork
2. Feature 브랜치 생성 (`git checkout -b feature/AmazingFeature`)
3. 변경사항 커밋 (`git commit -m 'Add AmazingFeature'`)
4. 브랜치에 Push (`git push origin feature/AmazingFeature`)
5. Pull Request 생성

### 기여 가능 영역

**완료된 기능 개선**
- 공/큐 감지 정확도 향상
- 물리 엔진 최적화
- 실패 분석 알고리즘 고도화

**신규 기능 구현**
- 쿠션 반사 분석 (알고리즘 제공됨)
- 스핀 효과 계산 (알고리즘 제공됨)
- 실시간 경로 추적
- 연습 모드 UI

---

## 📄 라이선스

본 프로젝트는 **MIT License** 하에 배포됩니다.

---

## 💡 기술 스택

- **언어**: Python 3.8+
- **비전**: OpenCV, NumPy
- **물리**: SciPy, 커스텀 2D 엔진
- **시각화**: Matplotlib, OpenCV
- **설정**: YAML, python-dotenv
- **테스트**: pytest

---
