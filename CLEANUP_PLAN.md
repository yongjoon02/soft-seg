# 🧹 기존 로그 및 결과 정리 계획

## 📊 현재 상태 (총 53.2GB)

```
lightning_logs/  : 51GB
  - octa500_3m/  : 17GB
  - octa500_6m/  : 17GB
  - rossa/       : 9.8GB
  - xca/         : 5.5GB
  - xca_backup/  : 2.1GB

logs/            : 2.2GB
results/         : 15MB
```

## 🎯 정리 대상

### 1. 삭제된 모델들의 로그 (~13GB)
삭제된 9개 모델의 체크포인트 및 로그:
- aacaunet, cenet, transunet, unet3plus, vesselnet (Supervised)
- segdiff, colddiff, maskdiff, maskdiff_v2 (Diffusion)

**위치:**
- `lightning_logs/octa500_3m/{deleted_models}/`
- `lightning_logs/octa500_6m/{deleted_models}/`
- `lightning_logs/rossa/{deleted_models}/`
- `lightning_logs/xca/{deleted_models}/`

### 2. 구버전 로그 파일 (~2.2GB)
`logs/` 디렉토리의 오래된 학습/평가 로그:
- `train_octa500_3m_*.log` (삭제된 모델 포함)
- `evaluate_*.log`

### 3. XCA 백업 (~2.1GB)
- `lightning_logs/xca_backup_wrong_intensity/`

### 4. 빈 results 디렉토리
- `results/octa500_3m/`, `results/octa500_6m/`, etc.

---

## 📦 정리 방법

### Option 1: 완전 삭제 (추천)
```bash
# 1. 삭제된 모델 로그 제거 (~13GB 절약)
rm -rf lightning_logs/*/aacaunet
rm -rf lightning_logs/*/cenet
rm -rf lightning_logs/*/transunet
rm -rf lightning_logs/*/unet3plus
rm -rf lightning_logs/*/vesselnet
rm -rf lightning_logs/*/segdiff
rm -rf lightning_logs/*/colddiff
rm -rf lightning_logs/*/maskdiff
rm -rf lightning_logs/*/maskdiff_v2

# 2. 구버전 로그 제거 (~2.2GB 절약)
rm -rf logs/

# 3. XCA 백업 제거 (~2.1GB 절약)
rm -rf lightning_logs/xca_backup_wrong_intensity/

# 4. 빈 디렉토리 제거
find results/ -type d -empty -delete

# 총 절약: ~17.3GB
```

### Option 2: 선택적 백업 후 삭제
```bash
# 1. 중요한 체크포인트만 백업
mkdir -p archive/old_experiments_backup_20251124

# Best checkpoints만 보관 (선택적)
for model in csnet dscnet medsegdiff berdiff; do
    for dataset in octa500_3m octa500_6m rossa; do
        if [ -d "lightning_logs/${dataset}/${model}/checkpoints" ]; then
            mkdir -p "archive/old_experiments_backup_20251124/${dataset}/${model}"
            cp lightning_logs/${dataset}/${model}/checkpoints/best.ckpt \
               archive/old_experiments_backup_20251124/${dataset}/${model}/ 2>/dev/null
        fi
    done
done

# 2. 나머지 완전 삭제
rm -rf lightning_logs/
rm -rf logs/
rm results/ -rf

# 3. 새 디렉토리 생성
mkdir logs
mkdir results
```

### Option 3: 아카이브 압축 (보관)
```bash
# 전체 압축 (시간 오래 걸림)
tar -czf archive/old_experiments_20251124.tar.gz \
    lightning_logs/ logs/ results/ \
    --exclude='lightning_logs/*/*/tensorboard/*'

# 압축 후 삭제
rm -rf lightning_logs/ logs/
mkdir logs
```

---

## ✅ 권장 조치

### 단계별 실행

**Step 1: 백업 (안전)**
```bash
# 현재 유지할 모델(csnet, dscnet, medsegdiff, berdiff)의 best checkpoint만 백업
bash scripts/backup_best_checkpoints.sh
```

**Step 2: 삭제된 모델 로그 제거 (~13GB)**
```bash
bash scripts/cleanup_deleted_models.sh
```

**Step 3: 구버전 로그 제거 (~2.2GB)**
```bash
rm -rf logs/*.log
```

**Step 4: 불필요한 백업 제거 (~2.1GB)**
```bash
rm -rf lightning_logs/xca_backup_wrong_intensity/
```

**Step 5: 빈 디렉토리 정리**
```bash
find results/ -type d -empty -delete
```

---

## 🎁 정리 후 구조

```
soft-seg/
├── lightning_logs/           # ~38GB (51GB → 38GB)
│   ├── octa500_3m/
│   │   ├── csnet/           # 유지
│   │   ├── dscnet/          # 유지
│   │   ├── medsegdiff/      # 유지
│   │   └── berdiff/         # 유지
│   ├── octa500_6m/
│   ├── rossa/
│   └── xca/
├── logs/                     # 비어있음 (새 로그용)
├── results/                  # 기존 분석 결과 유지
└── archive/                  # 백업 (선택적)
    └── old_experiments_backup_20251124/
        └── {best checkpoints}
```

---

## ⚠️ 주의사항

1. **백업 확인**: 중요한 체크포인트가 있는지 확인
2. **디스크 공간**: 백업 시 추가 공간 필요
3. **실행 전 확인**: dry-run 먼저 수행
4. **Git 상태**: 코드 변경사항은 이미 커밋됨 (안전)

---

## 🚀 자동화 스크립트 생성

실행하시겠습니까? 다음 스크립트를 생성합니다:
1. `scripts/backup_best_checkpoints.sh` - Best checkpoint 백업
2. `scripts/cleanup_deleted_models.sh` - 삭제된 모델 로그 제거
3. `scripts/cleanup_all.sh` - 전체 정리 자동화
