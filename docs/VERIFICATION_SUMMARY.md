# 네트워크 검증 요약표

## 🎯 Quick Reference

### Supervised Networks (8개)

| # | 네트워크 | 신뢰도 | 상태 | 논문 비교 가능? |
|---|----------|--------|------|----------------|
| 1 | **CS-Net** | ⭐⭐⭐⭐⭐ | ✅ 완벽 | YES |
| 2 | **UNet3Plus** | ⭐⭐⭐⭐⭐ | ✅ 완벽 | YES |
| 3 | **DSCNet** | ⭐⭐⭐⭐ | ⚠️ 95% | YES (근사) |
| 4 | **AACA-UNet** | ⭐⭐⭐⭐ | ⚠️ 80% | 참고용 |
| 5 | **CENet** | ⭐⭐⭐ | ⚠️ 70% | 참고용 |
| 6 | **TransUNet** | ⭐⭐⭐ | ⚠️ 70% | 참고용 |
| 7 | **VesselNet** | N/A | 🔧 Custom | NO |
| 8 | **OCT2Former** | N/A | 🔧 Custom | NO |

### Diffusion Models (6개)

| # | 모델 | 신뢰도 | 상태 | 논문 비교 가능? |
|---|------|--------|------|----------------|
| 1 | **Gaussian (SegDiff)** | ⭐⭐⭐⭐⭐ | ✅ 완벽 | YES |
| 2 | **Gaussian (MedSegDiff)** | ⭐⭐⭐⭐⭐ | ✅ 완벽 | YES |
| 3 | **Cold Diffusion** | ⭐⭐⭐⭐⭐ | ✅ 완벽 | YES |
| 4 | **Binomial (BerDiff)** | ⭐⭐⭐⭐⭐ | ✅ 완벽 | YES |
| 5 | **FlowSDF** | ⭐⭐⭐⭐ | ⚠️ 85% | YES (근사) |
| 6 | **Proposed v1/v2** | N/A | 🔧 Research | NO |

---

## 📊 통계

- **완벽 구현**: 5개 (35.7%)
- **근사/간소화**: 8개 (57.1%)
- **Custom/Research**: 1개 (7.1%)

---

## 🎯 실험 사용 가이드

### Paper 재현 실험
```python
# 100% 신뢰 가능
models = ['csnet', 'unet3plus', 'segdiff', 'medsegdiff', 'colddiff', 'berdiff']
```

### 비교 실험 (참고용)
```python
# 95%+ 신뢰
models = ['dscnet', 'flowsdf']

# 70-80% 신뢰
models = ['aacaunet', 'cenet', 'transunet']
```

### Custom 모델
```python
# 프로젝트 특화
models = ['vesselnet', 'oct2former', 'proposed', 'proposed_v2']
```

---

## ⚠️ 주의사항

1. **DSCNet**: DSConv가 간소화됨 (deformable conv 없음)
2. **CENet**: ASPP 모듈 빠짐
3. **TransUNet**: 공식 ViT 대신 custom transformer
4. **FlowSDF**: 간소화된 ODE solver

---

**상세 내역**: `NETWORK_VERIFICATION_COMPLETE.md` 참조
