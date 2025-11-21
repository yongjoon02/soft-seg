# Diffusion 모델을 위한 소프트 라벨 구현

## 개요
이 문서는 diffusion 기반 분할 모델을 위한 소프트 라벨 생성 시스템을 설명합니다. 소프트 라벨은 학습 중에 동적으로 생성되며 forward diffusion 과정에서 디노이징 타겟(x_0)으로 사용됩니다.

## 핵심 개념

### 소프트 라벨 vs 이진 라벨
- **이진 라벨(Binary Label)**: 하드 분할 마스크 (0 또는 1)
- **소프트 라벨(Soft Label)**: 경계에서 불확실성을 가진 연속적인 확률 맵
  - `boundary`: 혈관 경계에서의 거리 기반 불확실성
  - `thickness`: 혈관 두께 기반 불확실성
  - `sauna`: 경계 + 두께 결합 불확실성

### 목적
소프트 라벨은 forward diffusion 과정에서 **디노이징 타겟**으로 사용됩니다:
- `x_0` = 소프트 라벨 (이진 라벨이 아님)
- 모델은 `x_t`에서 소프트 라벨로 디노이징하는 것을 학습
- 이는 소프트 라벨을 샘플링 가이드로 사용하는 것(MaskDiff의 prob_img)과는 다름

## 구현

### 1. SoftLabelGenerator 클래스
**위치**: `src/data/transforms/soft_label.py`

```python
from src.data.transforms import SoftLabelGenerator

# 생성기 초기화
generator = SoftLabelGenerator(
    method='sauna',           # 옵션: 'none', 'boundary', 'thickness', 'sauna'
    cache=True,               # sample_id로 캐싱 활성화
    fg_max=11,                # 경계 불확실성의 최대 거리
    thickness_max=13,         # 두께 불확실성의 최대 두께
    kernel_ratio=0.1,         # 스무딩을 위한 커널 크기 비율
)

# 소프트 라벨 생성
soft_labels = generator(binary_labels, sample_ids=sample_names)
# 반환: [B, 1, H, W] 텐서로 소프트 라벨 반환
```

**주요 기능**:
- 4가지 방법 지원: `'none'` (이진), `'boundary'`, `'thickness'`, `'sauna'`
- 재생성을 피하기 위한 샘플 ID 기반 선택적 캐싱
- 효율성을 위한 배치 처리
- 자동 값 범위 처리 (boundary: [-1,1], thickness: [0,1], sauna: [-1,1])

### 2. DiffusionModel 통합
**위치**: `src/archs/diffusion_model.py`

**변경 사항**:
1. `__init__()`에 소프트 라벨 파라미터 추가:
   ```python
   def __init__(
       self,
       ...,
       soft_label_type: str = 'none',
       soft_label_cache: bool = True,
       soft_label_fg_max: int = 11,
       soft_label_thickness_max: int = 13,
       soft_label_kernel_ratio: float = 0.1,
   ):
   ```

2. `__init__()`에서 SoftLabelGenerator 초기화:
   ```python
   from src.data.transforms import SoftLabelGenerator
   self.soft_label_generator = SoftLabelGenerator(
       method=soft_label_type,
       cache=soft_label_cache,
       fg_max=soft_label_fg_max,
       thickness_max=soft_label_thickness_max,
       kernel_ratio=soft_label_kernel_ratio,
   )
   ```

3. 소프트 라벨을 생성하고 사용하도록 `training_step()` 수정:
   ```python
   def training_step(self, batch, batch_idx):
       images, labels = batch['image'], batch['label']
       
       # 디노이징 타겟으로 소프트 라벨 생성
       sample_ids = batch.get('name', None)
       soft_labels = self.soft_label_generator(labels, sample_ids)
       
       # diffusion forward 과정에서 x_0 타겟으로 소프트 라벨 사용
       target_labels = soft_labels
       
       # 소프트 라벨을 타겟으로 diffusion loss 계산
       loss = self(target_labels, images, prob_img)
       ...
   ```

### 3. 설정 파일
**위치**: 
- `configs/octa500_3m_diffusion_models.yaml`
- `configs/octa500_6m_diffusion_models.yaml`
- `configs/rossa_diffusion_models.yaml`

**추가된 파라미터**:
```yaml
model:
  arch_name: segdiff
  ...
  # 소프트 라벨 생성 설정 (forward diffusion에서 디노이징 타겟으로 사용)
  # 옵션: 'none' (이진), 'boundary', 'thickness', 'sauna' (경계+두께)
  soft_label_type: none
  soft_label_cache: true
  soft_label_fg_max: 11
  soft_label_thickness_max: 13
  soft_label_kernel_ratio: 0.1
```

### 4. 데이터셋 간소화
**위치**: 
- `src/data/octa500.py`
- `src/data/rossa.py`
- `src/data/base_dataset.py`

**변경 사항**:
1. `get_data_fields()`에서 `label_prob`와 `label_sauna` 제거:
   ```python
   def get_data_fields(self) -> list:
       """로드할 데이터 필드 목록 반환 (label_prob, label_sauna 제외).
       소프트 라벨은 학습 중에 동적으로 생성됩니다.
       """
       return ['image', 'label']
   ```

2. `FIELD_SCALE_CONFIG`를 `image`와 `label`만 포함하도록 업데이트:
   ```python
   FIELD_SCALE_CONFIG = {
       "image": (-1.0, 1.0),
       "label": (0.0, 1.0),
   }
   ```

**장점**:
- 메모리 사용량 감소 (더 이상 label_prob/label_sauna를 로드하지 않음)
- 더 깔끔한 데이터셋 인터페이스
- 더 유연한 실험 (config를 통해 소프트 라벨 방법 변경)

## 사용법

### 소프트 라벨로 학습하기

#### 1. 이진 라벨 (베이스라인)
```bash
python script/train_diffusion_models.py fit \
    --config configs/octa500_3m_diffusion_models.yaml \
    --model.arch_name segdiff \
    --model.soft_label_type none
```

#### 2. 경계 불확실성
```bash
python script/train_diffusion_models.py fit \
    --config configs/octa500_3m_diffusion_models.yaml \
    --model.arch_name segdiff \
    --model.soft_label_type boundary \
    --model.soft_label_fg_max 11
```

#### 3. 두께 불확실성
```bash
python script/train_diffusion_models.py fit \
    --config configs/octa500_3m_diffusion_models.yaml \
    --model.arch_name segdiff \
    --model.soft_label_type thickness \
    --model.soft_label_thickness_max 13
```

#### 4. SAUNA (경계 + 두께)
```bash
python script/train_diffusion_models.py fit \
    --config configs/octa500_3m_diffusion_models.yaml \
    --model.arch_name segdiff \
    --model.soft_label_type sauna \
    --model.soft_label_fg_max 11 \
    --model.soft_label_thickness_max 13 \
    --model.soft_label_kernel_ratio 0.1
```

### 파라미터 가이드

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `soft_label_type` | str | `'none'` | 방법: `'none'`, `'boundary'`, `'thickness'`, `'sauna'` |
| `soft_label_cache` | bool | `True` | sample_id로 캐싱 활성화 |
| `soft_label_fg_max` | int | `11` | 경계 불확실성의 최대 거리 |
| `soft_label_thickness_max` | int | `13` | 두께 불확실성의 최대 두께 |
| `soft_label_kernel_ratio` | float | `0.1` | 스무딩을 위한 커널 크기 비율 |

## 기술적 세부사항

### 소프트 라벨 생성 방법

#### 1. None (이진)
- 이진 라벨을 변경하지 않고 반환
- 비교를 위한 베이스라인으로 사용

#### 2. Boundary (경계)
- **함수**: `extract_boundary_uncertainty_map()`
- **출력 범위**: [-1, 1]
- **과정**:
  1. 혈관 경계로부터 거리 변환 계산
  2. `fg_max`로 정규화 (기본값: 11)
  3. tanh 스무딩 적용: `tanh((dist/fg_max - 0.5) * 2π)`
  4. [-1, 1]로 매핑: 혈관 내부 = 1, 외부 = -1, 경계 = 부드러운 전환

#### 3. Thickness (두께)
- **함수**: `extract_thickness_uncertainty_map()`
- **출력 범위**: [0, 1]
- **과정**:
  1. 거리 변환을 사용하여 혈관 두께 계산
  2. `thickness_max`로 정규화 (기본값: 13)
  3. 가우시안 스무딩 적용 (kernel_size = H * kernel_ratio)
  4. [0, 1]로 클립

#### 4. SAUNA (결합)
- **함수**: `extract_combined_uncertainty_map()`
- **출력 범위**: [-1, 1]
- **과정**:
  1. 경계 맵 생성 ([-1, 1])
  2. 두께 맵 생성 ([0, 1])
  3. 결합: `boundary_map * thickness_map`
  4. 결과: 경계와 두께 정보를 모두 가진 소프트 라벨

### 캐싱 전략
- **언제**: `cache=True`이고 `sample_ids`가 제공될 때
- **키**: 샘플 ID (예: 파일명)
- **저장**: 메모리 내 딕셔너리 `{sample_id: soft_label_tensor}`
- **장점**: 
  - 에폭 간 재생성 방지
  - 더 빠른 학습
  - 샘플당 일관된 소프트 라벨

### 메모리 고려사항
- 캐싱은 속도를 위해 메모리를 사용
- 대용량 데이터셋의 경우 `cache=False` 고려
- `generator.clear_cache()`로 캐시 삭제

## 비교: 소프트 라벨 vs MaskDiff 확률 맵

### 소프트 라벨 (이번 구현)
- **목적**: forward diffusion에서 디노이징 타겟
- **사용**: `q_sample(x_0, t)`에서 `x_0 = soft_label`
- **효과**: 모델이 소프트 확률 맵으로 디노이징하는 것을 학습
- **위치**: `training_step()`의 `target_labels`

### MaskDiff 확률 맵
- **목적**: reverse diffusion에서 샘플링 가이드
- **사용**: 샘플링 과정을 가이드하는 추가 입력
- **효과**: 불확실한 영역으로 샘플링을 편향
- **위치**: `training_step()`의 `prob_img`

**핵심 인사이트**: 이들은 상호보완적인 접근법입니다!
- 소프트 라벨: 모델이 학습하는 것 (타겟)
- 확률 맵: 모델이 샘플링하는 방법 (가이드)

## 테스트

### 빠른 테스트
```bash
# 이진 라벨로 테스트 (이전과 같이 작동해야 함)
python script/train_diffusion_models.py fit \
    --config configs/octa500_3m_diffusion_models.yaml \
    --model.arch_name segdiff \
    --trainer.max_epochs 1 \
    --trainer.limit_train_batches 2 \
    --trainer.limit_val_batches 2

# sauna 소프트 라벨로 테스트
python script/train_diffusion_models.py fit \
    --config configs/octa500_3m_diffusion_models.yaml \
    --model.arch_name segdiff \
    --model.soft_label_type sauna \
    --trainer.max_epochs 1 \
    --trainer.limit_train_batches 2 \
    --trainer.limit_val_batches 2
```

### 예상 동작
- 첫 번째 테스트: 이진 라벨로 정상 학습
- 두 번째 테스트: 실시간으로 소프트 라벨 생성
- 소프트 라벨 생성 메시지가 로그에 나타나는지 확인
- 라벨 shape이나 값과 관련된 오류가 없는지 확인

## 향후 확장

### Flow Matching 지원
동일한 소프트 라벨 인프라를 flow matching 모델에 적용할 수 있습니다:
1. `flow_matching_model.py` 생성 (`diffusion_model.py`와 유사)
2. `__init__()`에 `SoftLabelGenerator` 추가
3. flow matching forward 과정에서 `x_0`로 소프트 라벨 사용
4. 컴포넌트는 이미 `src/archs/components/flow_matching.py`에 존재

### 추가 방법
잠재적인 새로운 소프트 라벨 방법:
- **Skeleton 기반**: 혈관 위상에 기반한 불확실성
- **Multi-scale**: 다른 스케일에서 다른 불확실성
- **Learned**: 신경망 기반 소프트 라벨 생성

## 참고 자료

### 코드 파일
- 소프트 라벨 생성기: `src/data/transforms/soft_label.py`
- Diffusion 모델: `src/archs/diffusion_model.py`
- 불확실성 함수: `src/utils/generate_uncertainty.py`
- 설정 파일: `configs/*_diffusion_models.yaml`

### 관련 스크립트
- 학습: `script/train_diffusion_models.py`
- 평가: `script/evaluate_diffusion_models.py`
- SAUNA 맵 생성: `script/create_sauna_maps_v2.py`
- 테스트: `script/test_soft_labels.sh`

## 문제 해결

### 문제: 소프트 라벨이 생성되지 않음
**확인**: config에서 `soft_label_type`이 `'none'`이 아닌지 확인

### 문제: 캐시가 너무 많은 메모리 사용
**해결책**: config에서 `soft_label_cache: false` 설정

### 문제: 잘못된 값 범위
**확인**: 불확실성 함수가 올바른 범위를 반환하는지 확인:
- boundary: [-1, 1]
- thickness: [0, 1]
- sauna: [-1, 1]

### 문제: 예상보다 느린 학습
**확인**: `soft_label_cache: true`로 캐싱 활성화

## 요약
이 구현은 diffusion 기반 분할에서 소프트 라벨 생성을 위한 유연하고 config 기반의 시스템을 제공합니다. 여러 불확실성 방법을 지원하며 flow matching 및 기타 생성 모델로 쉽게 확장할 수 있습니다.
