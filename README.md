# Auto Map Matching

카메라 이미지와 지도 데이터를 매칭하여 카메라 자세를 조정하고, 특징점 기반으로 이미지 간 상대 자세를 추정하며, 차선 라벨링을 통한 자동 자세 피팅을 지원하는 도구

## 프로젝트 구조

```
auto-map-matching/
├── run.py                     # 메인 실행 파일
├── requirements.txt           # Python 패키지 의존성
├── app/                       # 애플리케이션 패키지
│   ├── ui/                   # UI 관련 모듈
│   │   ├── main_window.py   # 메인 윈도우
│   │   ├── matcher_dialog.py     # SIFT 특징점 매칭 다이얼로그
│   │   ├── matcher_dialog_orb.py # ORB 특징점 매칭 다이얼로그
│   │   ├── lane_labeling_dialog.py # 차선 라벨링 및 정답지 생성
│   │   └── lane_detection_sam3_dialog.py # SAM3 차선 검출 다이얼로그
│   └── core/                 # 핵심 로직
│       ├── feature_matcher.py    # 특징점 매칭 및 자세 추정
│       ├── auto_fitter.py        # 차선 기반 자동 자세 피팅 (Powell, NM, LM 등)
│       ├── lane_detector_sam3.py # SAM3 기반 차선 검출
│       ├── skeleton_lane_processor.py # 스켈레톤 기반 차선 후처리
│       └── geometry.py           # 기하학 변환 유틸리티
├── packages/
│   └── u1gis_geovision/      # GIS 투영 패키지 (git submodule)
├── models/                    # AI 모델 파일
│   └── sam3.pt               # SAM3 세그먼테이션 모델
├── camera/                    # 카메라 설정 폴더
├── image/                     # 이미지 폴더
├── lane_gt/                   # 차선 라벨링 데이터 (JSON)
└── shp/                       # Shapefile 지도 데이터
```

## 설치

### 1. 저장소 클론 (submodule 포함)

```bash
git clone --recursive https://github.com/jjongmae/auto-map-matching.git
cd auto-map-matching
```

이미 클론한 경우 submodule 초기화:
```bash
git submodule update --init
```

### 2. 가상환경 생성 및 활성화

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

## 사용 방법

### 실행

```bash
python run.py
```

### UI 사용법

#### 1. 맵 매칭 기본 기능

1. **카메라 선택**: 좌측 상단에서 카메라 폴더 선택
   - `camera/` 폴더 내의 `base.yaml`을 가진 폴더가 자동으로 스캔됩니다
2. **이미지 선택**: 이미지 목록에서 원하는 이미지 클릭
   - 선택한 이미지에 맵 오버레이가 자동으로 표시됩니다
3. **자세 조정**: Position과 Orientation 슬라이더/스핀박스로 조정
   - **Angle Normalization**: Yaw, Roll은 -180에서 180, Pitch는 -90에서 90 범위로 자동 정규화 및 UI 래핑이 적용됩니다.
4. **저장/리셋**: Save 버튼으로 현재 이미지 설정 저장, Reset 버튼으로 초기화

#### 2. Auto Fitting (자동 자세 피팅)

이미지에 차선 라벨링 데이터가 있는 경우, 지도의 차선과 정렬되도록 파라미터를 자동 최적화합니다.

1. **Lane Annotation**: 'Lane Annotation' 버튼을 클릭하여 현재 이미지에 대한 차선 라벨링을 수행하거나 편집합니다.
2. **Algorithm 선택**: UI 하단의 Auto Fit 그룹에서 알고리즘 선택
   - **Powell**: 방향 탐색 기반 최적화
   - **NM (Nelder-Mead)**: 심플렉스 기반, 노이즈에 강함
   - **LM (Levenberg-Marquardt)**: 비선형 최소제곱법, 빠른 수렴 (Robotics 표준)
   - **DE (Differential Evolution)**: 진화 알고리즘 기반 글로벌 최적화 (가장 강력함)
3. **Fit 실행**: 각 알고리즘 버튼을 클릭하면 최적화가 수행되고 결과가 즉시 반영됩니다.

#### 3. SAM3 차선 검출 (AI 자동 검출)

AI 모델(SAM3)을 사용하여 이미지에서 차선을 자동으로 검출합니다.

1. **SAM3 검출 시작**: 'SAM3 Lane Detection' 버튼을 클릭하여 차선 검출 다이얼로그 열기
2. **검출 설정**:
   - **프롬프트**: 검출할 객체를 설명하는 텍스트 (기본: "lane line")
   - **신뢰도**: 검출 신뢰도 임계값 (0.1 ~ 1.0, 기본: 0.5)
3. **검출 실행**: '검출 시작' 버튼을 클릭하면 SAM3 모델이 차선을 자동으로 검출
4. **결과 확인**: 
   - **좌측**: SAM3가 검출한 Raw 마스크 (세그먼테이션 결과)
   - **우측**: 후처리된 폴리라인 결과 (스켈레톤 + DBSCAN + B-spline)
   - 줌 인/아웃 및 스크롤로 세부 확인 가능
5. **저장**: 검출 결과를 `lane_gt/` 폴더에 JSON 형식으로 저장하여 Auto Fitting에 활용

#### 4. Feature Comparison (특징점 매칭)

이미지 목록에서 두 개의 이미지를 선택한 후 Compare 버튼을 클릭하면:

- **SIFT/ORB 매칭**: 선택한 알고리즘으로 두 이미지 간 특징점을 매칭
- **상대 자세 추정**: Essential Matrix를 통한 카메라 간 상대적 회전/이동 추정
- **자동 적용**: 추정된 자세를 두 번째 이미지에 자동으로 적용하여 맵 정렬 상태를 제안합니다.

## 카메라 YAML 파일 형식

```yaml
# Extrinsic Parameters (world coordinates)
x: 376841.249
y: 4116956.264
z: 135.592
yaw: -179.1
pitch: -16.725
roll: 0.8

# Intrinsic Parameters (pixels)
fx: 1788.9
fy: 1790.96
cx: 927.1627
cy: 541.8689

# Distortion Coefficients
k1: -0.19563
k2: 0.145216
p1: -0.00062
p2: -0.00204
k3: 0

# Resolution
resolution_width: 1920
resolution_height: 1080
```

## 폴더 명명 규칙

카메라 관련 폴더명은 모두 일치해야 합니다:

- `camera/여주시험도로_001/base.yaml` (필수)
- `camera/여주시험도로_001/img001.yaml` (선택, 이미지별 설정)
- `image/여주시험도로_001/img001.jpg`
- `lane_gt/여주시험도로_001/img001.json` (선택, 차선 라벨링 데이터)
- `shp/여주시험도로_001/`

**이미지별 설정 파일:**
- 이미지 파일명과 동일한 이름의 YAML 파일을 생성하면 해당 이미지에 대한 개별 설정으로 사용됩니다
- 예: `img001.jpg` → `img001.yaml`
- 없으면 자동으로 `base.yaml` 사용

## 주요 기능

### 맵 매칭 & 자세 조정
- 카메라 폴더 자동 스캔 및 실시간 맵 오버레이
- Yaw/Pitch/Roll 각도 범위 자동 정규화 및 UI 스핀박스 래핑 지원
- 이미지별 조정된 파라미터 저장 및 리셋

### 자동 자세 피팅 (Auto Fitting)
- 차선 라벨링 데이터를 활용한 다중 최적화 알고리즘 지원
- **Powell, Nelder-Mead(NM), Levenberg-Marquardt(LM), Differential Evolution(DE)**
- 투영된 지도 차선과 라벨링된 차선 간의 거리 최소화 방식

### 특징점 기반 상대 자세 추정
- SIFT/ORB 기반 특징점 검출 및 Robust 매칭 (Cross-check, Ratio test)
- Essential Matrix 및 RANSAC 기반의 카메라 간 상대 자세 추구
- 추정된 자세를 인접 이미지에 자동 적용 및 시각화

### SAM3 기반 자동 차선 검출
- Ultralytics SAM3 모델을 활용한 AI 기반 차선 자동 검출
- 텍스트 프롬프트(예: "lane line")를 통한 세그먼테이션 기반 차선 인식
- **두 가지 후처리 방식 지원:**
  - **다항식 피팅**: 2차/3차 다항식으로 차선 스무딩 (빠르고 간단)
  - **스켈레톤 기반**: 노이즈 제거 → 스켈레톤화 → DBSCAN 클러스터링 → B-spline 피팅 (고정밀)
- Raw 마스크와 후처리 결과를 동시에 표시하여 디버깅 용이
- 검출된 차선을 `lane_gt` 포맷으로 저장하여 자동 피팅에 즉시 활용 가능

### 차선 라벨링 도구
- 이미지 상의 차선을 클릭하여 정답지(`lane_gt`) 생성 및 편집 도구 내장
- 최적화 알고리즘의 입력 데이터로 즉시 활용 가능
