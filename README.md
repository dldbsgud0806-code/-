# 카메라 기반 실시간 당구 샷 코칭 시스템

개요
- 본 프로젝트는 카메라를 활용해 당구대 위의 공과 큐의 위치/각도를 실시간으로 추정하고, 물리 모델을 기반으로 최적의 샷 후보를 제시하는 오픈소스 코칭 시스템입니다.
- 실시간 시각적 오버레이와 텍스트/음성 피드백으로 학습 효과를 높이고, 로컬 처리 원칙과 익명화 옵션으로 프라이버시를 보장합니다.
- 모듈화된 아키텍처로 웹/데스크탑/모바일 환경에 이식 가능하며, 외부 애플리케이션과의 연동을 위한 API/SDK를 제공합니다.

목차
- 개요 및 목표
- 아키텍처 개요
- 모듈 상세 설명
  - 비전 모듈
  - 물리 시뮬레이션 모듈
  - 샷 평가/랭킹 엔진
  - 피드백 엔진
  - 데이터 관리 및 프라이버시
  - API/SDK
- 데이터 포맷 및 예시
- 구현 예제 코드 스니펫
- 구현 로드맷( MVP 포함)
- 테스트 전략
- 배포 및 운영
- 오픈소스 운영 가이드
- 라이선스
- 기여 방법
- 참고 자료 및 레퍼런스
- 실행 방법


1. 개요 및 목표
- 목표: 실시간으로 큐 거리, 타점, 스트로크 세기, 샷 방향 등 핵심 파라미터를 추정하고, 물리 모델로 최적의 샷 후보를 제시하는 교육용 코칭 도구를 제공합니다.
- 학습 효과: 시각적 오버레이와 텍스트/음성 피드백으로 학습 비용을 감소시키고, 반복 학습의 효율을 높입니다.
- 프라이버시: 로컬 처리 우선 정책을 기본으로 하며, 필요 시 익명화된 데이터만 전송합니다.
- 커뮤니티: 모듈화된 아키텍처와 명확한 기여 가이드로 오픈소스 커뮤니티에 쉽게 기여할 수 있도록 합니다.

2. 아키텍처 개요
- 주요 컴포넌트
  - 비전 모듈: 공 위치 추적, 큐 위치/각도 추정, 환경 인자 인식
  - 물리 시뮬레이션 모듈: 2D 물리 모델로 충돌, 마찰, 쿠션 반사를 시뮬레이션
  - 샷 평가/랭킹 엔진: 상태공간에서 후보 샷 생성 및 점수화
  - 피드백 엔진: 시각적 오버레이, 텍스트/음성 피드백
  - 데이터 관리 모듈: 로컬 저장소, 익명화 처리, 정책 준수 로깅
  - API/SDK 모듈: 외부 애플리케이션 연동 인터페이스
- 데이터 흐름
  - 카메라 입력 → 비전 모듈 → 좌표계 변환 → 물리 시뮬레이션 → 샷 후보 랭킹 → 피드백 UI/로그 → 로컬 저장
- 기술 스택 제안
  - 비전: OpenCV, MediaPipe(공 추적 중심의 경량화)
  - 물리 시뮬레이션: 2D PyBullet, NumPy 기반 커스텀 엔진
  - 최적화/ML 보조: SciPy.optimize, PyTorch(경량 모델)
  - UI: 웹(WebGL/Canvas, React) 또는 Desktop(Qt/ImGui)
  - 배포: Docker + GitHub Actions

3. 모듈 상세 설명
3.1 비전 모듈
- 기능
  - 공 위치 추정(다수의 공 좌표 추정, 월드 좌표 매핑)
  - 큐 위치/각도 추정(시작점, 끝점, 스윙 방향)
  - 환경 인자 인식(조도, 벽/쿠션 위치)
- 간단한 구현 예시
  ```python
  # vision/core.py
  import cv2
  import numpy as np

  def detect_balls(frame: np.ndarray, min_r=5, max_r=30):
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      blurred = cv2.medianBlur(gray, 5)
      circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                                 param1=50, param2=30, minRadius=min_r, maxRadius=max_r)
      balls = []
      if circles is not None:
          for c in circles[0, :]:
              x, y, r = map(int, c)
              balls.append({'x': x, 'y': y, 'r': r})
      return balls
  ```
3.2 물리 시뮬레이션 모듈
- 기능
  - 2D 평면에서 큐/볼의 위치를 시간 축에 따라 시뮬레이션
  - 충돌, 마찰, 쿠션 반사 적용
- 예시 구현 아이디어
  ```python
  # physics/engine.py
  import numpy as np

  class Ball:
      def __init__(self, x, y, vx, vy, r=2.25, m=0.17):
          self.x, self.y, self.vx, self.vy, self.r, self.m = x, y, vx, vy, r, m

  def step(balls, dt, w, h, mu=0.02):
      for b in balls:
          b.x += b.vx * dt
          b.y += b.vy * dt
          b.vx *= max(0, 1 - mu * dt)
          b.vy *= max(0, 1 - mu * dt)
          if b.x - b.r = w:
              b.vx *= -1
          if b.y - b.r = h:
              b.vy *= -1
  ```
3.3 샷 평가/랭킹 엔진
- 기능
  - 현재 상태에서 가능한 샷 후보를 생성
  - 물리 시뮬레이션 결과를 바탕으로 점수화 및 랭킹
- 예시 스켈레톤
  ```python
  # ranking/score.py
  import numpy as np

  def score_shot(state, candidate):
      distance_penalty = np.linalg.norm(np.array(state['cue']) - np.array(state['target'])) * 0.1
      contact_quality = 1.0 - abs(candidate['tp'] - 0.5)
      return max(0.0, 1.0 - distance_penalty) * contact_quality
  ```
3.4 피드백 엔진
- 기능
  - 시각적 오버레이: 경로, 타점, 큐 위치, 예상 충돌 경로를 화면에 렌더링
  - 텍스트/음성 피드백
- 간단한 웹 예시
  ```tsx
  // ui/Overlay.jsx
  import React from 'react';
  export default function Overlay({ paths, cue, target, predicted }) {
    return (
      
        {paths.map((p, i) => (
           `${pt.x},${pt.y}`).join(' ')} stroke="orange" fill="none" strokeWidth="2" />
        ))}
        
        
      
    );
  }
  ```
3.5 데이터 관리 및 프라이버시
- 로컬 저장 우선 정책 예시
  ```python
  # data/storage.py
  import sqlite3

  def init_db(db_path='local.db'):
      conn = sqlite3.connect(db_path)
      c = conn.cursor()
      c.execute('CREATE TABLE IF NOT EXISTS logs (timestamp TEXT, data TEXT)')
      conn.commit()
      return conn
  ```
3.6 API/SDK
- 외부 앱 연동 인터페이스 설계 예시
  - REST/GraphQL 엔드포인트 또는 로컬 라이브러리 형태
  - 인증/권한 관리 및 샘플 요청/응답 포맷

4. 구현 로드맷( MVP 포함)
- 0~1개월: 공 위치 추적 MVP + 큐 추정
- 1~2개월: 샷 후보 생성/랭킹 + 시각적 오버레이
- 2~3개월: 쿠션 반사 포함 경로 예측, 로컬 저장/익명화 옵션
- 3개월 이상: API/SDK 공개, 다중 환경 검증

5. 테스트 전략
- 단위 테스트: 비전/시뮬레이션 모듈별 테스트
- 통합 테스트: 시나리오 기반 엔드투엔드 테스트
- 성능 테스트: 프레임레이트 30/60fps 달성 여부, 지연 시간 측정
- 사용성 테스트: 피드백 해석성/학습 효과 설문

6. 배포 및 운영
- 배포 패키지: Docker 기반 컨테이너 또는 플랫폼별 패키지
- CI/CD: GitHub Actions를 이용한 빌드/테스트/배포 파이프라인
- 문서화: CONTRIBUTING.md, CODE_OF_CONDUCT.md, CHANGELOG

7. 오픈소스 운영 가이드
- 라이선스: MIT 또는 Apache 2.0
- 코드 품질: PEP8, 테스트 커버리지 목표
- 문서화: API 문서, 개발자 가이드, 기여 예제 포함
- 이슈/PR 관리: 템플릿 및 리뷰 프로세스 정의

8. 라이선스
- 본 프로젝트는 오픈소스 커뮤니티에 기여하기 위한 MIT 라이선스(또는 Apache 2.0) 하에 배포됩니다.

9. 기여 방법
- 기여 절차: 새로운 기능 추가, 버그 수정, 문서 개선 등
- 브랜치 전략: main(배포) / dev(개발) / feature/xxx
- PR 템플릿 및 이슈 템플릿 제공

10. 참고 자료 및 레퍼런스
- 기존 사례: https://github.com/honey-da/osd.git 등
- 관련 논문/블로그 및 표준 포맷 예시


11. 실행 방법 요약
- 로컬 개발 환경 세팅
- 의존성 설치: pip install -r requirements.txt, npm install
- MVP 실행: python main.py --mode vision
- 테스트 데이터로 검증

