# 🎯 리팩토링 완료: Thin Scripts, Fat Library

## ✅ 변경 사항 요약

### 구조 개선
- **scripts/**: 매우 간단 (CLI만, ~50 lines)
- **src/**: 모든 로직 (registry, runner, experiment tracker)
- **TensorBoard**: 풍부한 로깅 (메트릭 + 이미지 + 하이퍼파라미터)

### 새로 추가된 모듈

```
src/
├── registry/              # 모델/데이터셋 메타데이터 관리
│   ├── models.py          # MODEL_REGISTRY + 정보
│   └── datasets.py        # DATASET_REGISTRY + 정보
│
├── experiment/            # 실험 추적 및 로깅
│   ├── tracker.py         # 자동 실험 관리 (experiments.json)
│   └── logger.py          # TensorBoard 통합 로거
│
└── runner/                # 실행 로직
    └── train_runner.py    # 모든 학습 로직
```

---

## 🚀 사용법

### 1. 기본 학습

```bash
# 단순한 명령어로 학습 시작
uv run python scripts/train.py --model csnet --data octa500_3m
```

**자동으로 수행되는 작업:**
- ✅ Registry에서 모델/데이터 정보 로드
- ✅ Default 하이퍼파라미터 적용
- ✅ Experiment ID 생성 및 디렉토리 구성
- ✅ Git hash 저장 (재현성)
- ✅ Config 자동 저장 (`experiments/.../config.yaml`)
- ✅ TensorBoard 로깅 시작
- ✅ 학습 완료 후 metrics 자동 저장

### 2. 하이퍼파라미터 커스터마이징

```bash
uv run python scripts/train.py --model csnet --data octa500_3m \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.001 \
    --gpu 1
```

### 3. Diffusion 모델 + Soft Labels

```bash
uv run python scripts/train.py --model medsegdiff --data octa500_3m \
    --soft-label thickness \
    --soft-label-thickness-max 13 \
    --timesteps 1000 \
    --ensemble 5
```

### 4. 디버그 모드 (빠른 테스트)

```bash
# 2 epochs, 10 batches만 (빠른 검증)
uv run python scripts/train.py --model csnet --data octa500_3m --debug
```

### 5. 학습 재개

```bash
uv run python scripts/train.py --model csnet --data octa500_3m \
    --resume experiments/csnet/octa500_3m/20250124_150000/checkpoints/last.ckpt
```

---

## 📊 TensorBoard로 실험 모니터링

### TensorBoard 실행

```bash
# 모든 실험 보기
tensorboard --logdir experiments/

# 특정 모델만
tensorboard --logdir experiments/csnet/

# 특정 데이터셋만
tensorboard --logdir experiments/*/octa500_3m/
```

### TensorBoard에 기록되는 내용

1. **Scalars (메트릭)**
   - `train/loss`
   - `val/dice`, `val/cldice`, `val/betti_0_error`, `val/betti_1_error`
   - `val/precision`, `val/recall`, `val/specificity`, `val/iou`
   - Learning rate

2. **Images**
   - `predictions/comparison`: 입력 / 예측 / 정답 비교
   - 주기적인 validation 시각화

3. **Hparams**
   - 모든 하이퍼파라미터
   - 최종 메트릭과 함께 비교 가능

4. **Graph** (선택적)
   - 모델 구조 시각화

---

## 🗂️ 실험 디렉토리 구조

```
experiments/
├── experiments.json              # 모든 실험 메타데이터 DB
├── csnet/
│   ├── octa500_3m/
│   │   └── csnet_octa500_3m_20250124_150000/
│   │       ├── config.yaml       # 재현용 설정
│   │       ├── git_info.txt      # Git hash, branch
│   │       ├── checkpoints/
│   │       │   ├── best.ckpt     # Best validation
│   │       │   └── last.ckpt     # Latest
│   │       ├── tensorboard/      # TensorBoard logs
│   │       │   └── events.out.tfevents...
│   │       └── summary.json      # 최종 결과
│   └── octa500_6m/
└── medsegdiff/
    └── octa500_3m/
```

---

## 🔧 새 모델 추가 (매우 간단!)

### Step 1: 모델 구현

```python
# src/archs/components/new_model.py
class NewModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        # Your implementation
    
    def forward(self, x):
        return x
```

### Step 2: Registry 등록 (단 하나의 entry!)

```python
# src/registry/models.py
from src.archs.components.new_model import NewModel

MODEL_REGISTRY['newmodel'] = ModelInfo(
    name='newmodel',
    class_ref=NewModel,
    task='supervised',
    params=10_000_000,
    speed='fast',
    description='Your new model',
    default_lr=1e-3,
    default_epochs=300,
)
```

### Step 3: 즉시 사용!

```bash
uv run python scripts/train.py --model newmodel --data octa500_3m
```

**끝! 다른 파일 수정 불필요!**

---

## 📦 새 데이터셋 추가

### Step 1: DataModule 구현

```python
# src/data/new_dataset.py
from src.data.base_dataset import BaseOCTDataset, BaseOCTDataModule

class NewDataset(BaseOCTDataset):
    def get_data_fields(self):
        return ['image', 'label']

class NewDataModule(BaseOCTDataModule):
    dataset_class = NewDataset
```

### Step 2: Registry 등록

```python
# src/registry/datasets.py
DATASET_REGISTRY['new_dataset'] = DatasetInfo(
    name='new_dataset',
    class_ref=NewDataModule,
    modality='OCTA',
    resolution=(512, 512),
    num_train=100,
    num_val=20,
    num_test=20,
    description='New dataset',
    default_train_dir='data/NEW_DATASET/train',
    default_val_dir='data/NEW_DATASET/val',
    default_test_dir='data/NEW_DATASET/test',
)
```

### Step 3: 즉시 사용!

```bash
uv run python scripts/train.py --model csnet --data new_dataset
```

---

## 🎁 핵심 장점

### 1. Scripts는 매우 간단
- `scripts/train.py`: ~120 lines (대부분 argparse)
- 모든 로직은 `src/`에 있음
- 유지보수 쉬움

### 2. 확장성
- 새 모델: Registry에 1 entry 추가
- 새 데이터셋: Registry에 1 entry 추가
- Scripts 수정 불필요

### 3. 자동 추적
- 모든 실험 자동 기록 (`experiments.json`)
- Git hash 저장 (완벽한 재현)
- Config 자동 저장
- TensorBoard 자동 통합

### 4. TensorBoard 활용
- 실시간 메트릭 모니터링
- 예측 시각화
- 하이퍼파라미터 비교
- 모델 구조 시각화

### 5. 연구 친화적
- 디버그 모드로 빠른 테스트
- Checkpoint 자동 관리
- 실험 비교 용이
- 논문용 재현 쉬움

---

## 📝 비교: Before vs After

### Before (기존)

```bash
# 24개 스크립트 파일
train_supervised_octa_3m.sh
train_supervised_octa_6m.sh
train_diffusion_octa_3m.sh
...

# 8개 YAML 파일
configs/octa500_3m_supervised_models.yaml
...

# 수동 체크포인트 관리
# 수동 로그 확인
# 실험 비교 어려움
```

### After (현재)

```bash
# 1개 스크립트 (모든 것을 할 수 있음)
scripts/train.py

# Registry 기반 자동 설정
src/registry/models.py
src/registry/datasets.py

# 자동 실험 추적
experiments/experiments.json

# TensorBoard 통합
tensorboard --logdir experiments/
```

---

## 🚀 다음 단계

이제 다음 작업을 진행할 수 있습니다:

1. **평가 스크립트** (`scripts/evaluate.py`)
2. **실험 비교 도구** (`scripts/experiment.py`)
3. **단위 테스트** 추가

어떤 것부터 진행하시겠습니까?
