# 새로운 학습/평가 시스템 가이드

리팩토링된 시스템은 **데이터셋 중심**으로 구성되어 있습니다. 하나의 명령으로 모든 모델을 학습하거나 평가할 수 있습니다.

## 📁 디렉토리 구조

```
soft-seg/
├── src/
│   ├── registry/          # 모델/데이터셋 메타데이터
│   │   ├── models.py      # MODEL_REGISTRY (4개 모델)
│   │   └── datasets.py    # DATASET_REGISTRY (3개 데이터셋)
│   ├── experiment/        # 실험 추적 시스템
│   │   ├── tracker.py     # ExperimentTracker
│   │   └── logger.py      # EnhancedTensorBoardLogger
│   └── runner/            # 학습/평가 실행기
│       ├── train_runner.py  # TrainRunner
│       └── eval_runner.py   # EvalRunner
├── scripts/               # 실행 스크립트
│   ├── train.py          # 학습 CLI
│   ├── evaluate.py       # 평가 CLI
│   ├── train_octa500_3m.sh   # OCTA500 3M 전체 학습
│   ├── train_octa500_6m.sh   # OCTA500 6M 전체 학습
│   ├── train_rossa.sh        # ROSSA 전체 학습
│   ├── eval_octa500_3m.sh    # OCTA500 3M 전체 평가
│   ├── eval_octa500_6m.sh    # OCTA500 6M 전체 평가
│   └── eval_rossa.sh         # ROSSA 전체 평가
├── experiments/           # 실험 결과
│   ├── experiments.json   # 실험 데이터베이스
│   └── {model}/{dataset}/{run_id}/
│       ├── config.yaml
│       ├── git_info.txt
│       ├── checkpoints/
│       ├── tensorboard/
│       └── summary.json
└── script_legacy/         # 구 스크립트 백업
```

## 🚀 사용 방법

### 학습 (Training)

#### 데이터셋별 전체 모델 학습 (권장)

```bash
# OCTA500 3M: csnet, dscnet, medsegdiff, berdiff 동시 학습
./scripts/train_octa500_3m.sh

# OCTA500 6M: 전체 모델 학습
./scripts/train_octa500_6m.sh

# ROSSA: 전체 모델 학습
./scripts/train_rossa.sh
```

각 스크립트는:
- 4개 모델을 각각 다른 GPU(0,1,2,3)에 자동 할당
- 백그라운드에서 병렬 실행
- 로그를 `logs/train_{dataset}_{model}.log`에 저장

#### 개별 모델 학습

```bash
# 특정 모델만 학습
uv run python scripts/train.py --model csnet --data octa500_3m --gpu 0

# 특정 GPU 사용
uv run python scripts/train.py --model medsegdiff --data octa500_6m --gpu 2

# 배치 크기 조정
uv run python scripts/train.py --model berdiff --data rossa --batch-size 8
```

#### 모니터링

```bash
# TensorBoard
tensorboard --logdir experiments/ --port 6006 --host 0.0.0.0
# 접속: http://localhost:6006

# GPU 사용량
watch -n 1 nvidia-smi

# 로그 확인
tail -f logs/train_octa500_3m_*.log

# 학습 중단
pkill -f train.py
```

### 평가 (Evaluation)

#### 데이터셋별 전체 모델 평가 (권장)

```bash
# OCTA500 3M: 전체 모델 평가
./scripts/eval_octa500_3m.sh

# OCTA500 6M: 전체 모델 평가
./scripts/eval_octa500_6m.sh

# ROSSA: 전체 모델 평가
./scripts/eval_rossa.sh
```

결과는 `results/{dataset}/evaluation_{dataset}.csv`에 저장됩니다.

#### 개별 모델 평가

```bash
# 특정 모델만 평가
uv run python scripts/evaluate.py --data octa500_3m --models csnet,dscnet

# 예측 이미지 저장
uv run python scripts/evaluate.py --data octa500_6m --save-predictions

# 특정 GPU 사용
uv run python scripts/evaluate.py --data rossa --gpu 1

# 커스텀 출력 디렉토리
uv run python scripts/evaluate.py --data octa500_3m --output results/my_eval
```

## 📊 등록된 모델 및 데이터셋

### 모델 (4개)

| 모델 | 타입 | 파라미터 | 설명 |
|------|------|---------|------|
| csnet | supervised | 8.4M | CS-Net (Channel & Spatial attention) |
| dscnet | supervised | 5.8M | DSCNet (Dual-stage cascaded) |
| medsegdiff | diffusion | 16.2M | MedSegDiff (Medical segmentation diffusion) |
| berdiff | diffusion | 9.3M | BerDiff (Bernoulli diffusion) |

### 데이터셋 (3개)

| 데이터셋 | 이미지 크기 | Train/Val/Test | 설명 |
|---------|-----------|---------------|------|
| octa500_3m | 304x304 | 200/50/50 | OCTA-500 3×3mm |
| octa500_6m | 400x400 | 200/50/50 | OCTA-500 6×6mm |
| rossa | 304x304 | 35/9/9 | ROSSA dataset |

## 🔧 주요 기능

### 1. 자동 실험 추적
- 모든 실험이 `experiments/`에 자동 기록
- Git commit hash, 설정, 메트릭 자동 저장
- Best checkpoint 자동 저장

### 2. TensorBoard 통합
- 학습 곡선, 메트릭 자동 로깅
- 검증 이미지 주기적 저장
- 하이퍼파라미터 기록

### 3. 통합된 인터페이스
- Supervised와 Diffusion 모델 동일한 방식으로 사용
- 데이터셋 자동 전환
- GPU 자동 할당

### 4. 평가 자동화
- Best checkpoint 자동 탐색
- 전체 메트릭 계산 (Dice, IoU, Precision, Recall 등)
- CSV 형식으로 결과 저장

## 💡 워크플로우 예시

```bash
# 1. OCTA500 3M 전체 학습
./scripts/train_octa500_3m.sh

# 2. TensorBoard로 모니터링
tensorboard --logdir experiments/ --port 6006 --bind_all

# 3. 학습 완료 후 평가
./scripts/eval_octa500_3m.sh

# 4. 결과 확인
cat results/octa500_3m/evaluation_octa500_3m.csv

# 5. 다른 데이터셋으로 반복
./scripts/train_octa500_6m.sh
./scripts/eval_octa500_6m.sh
```

## 🎯 이전 시스템과 차이점

### 이전 (script/)
```bash
# Supervised와 Diffusion 모델 별도 스크립트
python script/train_supervised_models.py --models csnet,dscnet
python script/train_diffusion_models.py --models medsegdiff,berdiff

# 평가도 별도
python script/evaluate_supervised_models.py
python script/evaluate_diffusion_models.py
```

### 현재 (scripts/)
```bash
# 모든 모델 통합
./scripts/train_octa500_3m.sh  # 4개 모델 전체 학습
./scripts/eval_octa500_3m.sh   # 4개 모델 전체 평가
```

## 📝 추가 정보

- **실험 데이터베이스**: `experiments/experiments.json`
- **로그 디렉토리**: `logs/`
- **결과 디렉토리**: `results/`
- **레거시 스크립트**: `script_legacy/` (백업용)

## 🛠️ 트러블슈팅

### 학습이 시작되지 않음
```bash
# GPU 사용 가능 확인
nvidia-smi

# 프로세스 확인
ps aux | grep train.py

# 로그 확인
tail -f logs/train_*.log
```

### 체크포인트를 찾을 수 없음
```bash
# 실험 디렉토리 확인
ls -la experiments/{model}/{dataset}/

# best.ckpt 존재 확인
find experiments/ -name "best.ckpt"
```

### 메모리 부족
```bash
# 배치 크기 줄이기
uv run python scripts/train.py --model csnet --data octa500_3m --batch-size 8

# GPU 하나씩 학습
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py --model csnet --data octa500_3m
```
