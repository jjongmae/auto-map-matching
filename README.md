# Auto Map Matching

카메라 이미지와 지도 데이터를 매칭하여 카메라 자세를 조정하고, 특징점 기반으로 이미지 간 상대 자세를 추정하는 도구

## 프로젝트 구조

```
auto-map-matching/
├── run.py                     # 메인 실행 파일
├── requirements.txt           # Python 패키지 의존성
├── app/                       # 애플리케이션 패키지
│   ├── ui/                   # UI 관련 모듈
│   │   ├── main_window.py   # 메인 윈도우
│   │   ├── matcher_dialog.py     # SIFT 특징점 매칭 다이얼로그
│   │   └── matcher_dialog_orb.py # ORB 특징점 매칭 다이얼로그
│   └── core/                 # 핵심 로직
│       ├── feature_matcher.py    # 특징점 매칭 및 자세 추정
│       └── geometry.py           # 기하학 변환 유틸리티
├── packages/
│   └── vaid_gis/             # VAID GIS 패키지 (로컬)
├── camera/                    # 카메라 설정 폴더
│   └── 여주시험도로_001/      # 카메라별 설정 폴더
│       ├── base.yaml         # 기본 설정 (필수)
│       ├── img001.yaml       # 이미지별 개별 설정 (선택)
│       └── img002.yaml
├── image/                     # 이미지 폴더
│   └── 여주시험도로_001/      # 카메라별 이미지 폴더
│       ├── img001.jpg
│       └── ...
└── shp/                       # Shapefile 지도 데이터
    └── 여주시험도로_001/      # 카메라별 SHP 폴더
        ├── *.shp
        └── ...
```

## 설치

### 1. 가상환경 생성 및 활성화

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
pip install -e packages/vaid_gis
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
   - 선택 시 `base.yaml`이 로드되고 맵 데이터와 이미지 목록이 표시됩니다

2. **이미지 선택**: 이미지 목록에서 원하는 이미지 클릭
   - `image/{카메라명}/` 폴더의 이미지가 표시됩니다
   - 해당 이미지명과 같은 YAML 파일이 있으면 자동으로 로드됩니다
   - 없으면 `base.yaml`의 설정을 사용합니다
   - 선택한 이미지에 맵 오버레이가 자동으로 표시됩니다

3. **자세 조정**: Position과 Orientation 슬라이더로 조정
   - **Position**: X, Y, Z (카메라 위치 좌표)
   - **Orientation**: Yaw, Pitch, Roll (카메라 회전 각도)
   - 조정 시 실시간으로 맵 오버레이가 업데이트됩니다

4. **저장/리셋**:
   - **Save**: 현재 이미지명으로 YAML 파일 저장 (예: `img001.yaml`)
   - **Reset**: 현재 이미지의 YAML 파일에서 값을 다시 로드 (있으면 이미지명.yaml, 없으면 base.yaml)

#### 2. Feature Comparison 기능

이미지 목록에서 두 개의 이미지를 선택한 후 Compare 버튼을 클릭하면:

- **SIFT 특징점 매칭**: SIFT 알고리즘을 사용하여 두 이미지 간 특징점을 매칭
- **ORB 특징점 매칭**: ORB 알고리즘을 사용한 빠른 특징점 매칭
- **상대 자세 추정**: Essential Matrix를 통한 카메라 간 상대적 회전/이동 추정
- **시각화**: 매칭된 특징점과 추정된 자세 파라미터를 시각적으로 표시

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
- `shp/여주시험도로_001/`

**이미지별 설정 파일:**
- 이미지 파일명과 동일한 이름의 YAML 파일을 생성하면 해당 이미지에 대한 개별 설정으로 사용됩니다
- 예: `img001.jpg` → `img001.yaml`
- 없으면 자동으로 `base.yaml` 사용

## 주요 기능

### 맵 매칭
- 카메라 폴더 자동 스캔 및 `base.yaml` 로드
- 이미지별 개별 설정 지원 (이미지명.yaml)
- 이미지 기반 맵 매칭 실시간 시각화
- 실시간 자세 조정 및 프리뷰
- 이미지별 조정된 파라미터 저장

### Feature Comparison & Pose Estimation
- SIFT/ORB 기반 특징점 검출 및 매칭
- Cross-check 및 Lowe's ratio test를 통한 robust 매칭
- Essential Matrix를 통한 상대 자세 추정
- RANSAC 기반 outlier 제거
- 매칭 결과 및 자세 파라미터 시각화
- 추정된 자세를 두 번째 이미지에 자동 적용
