# Surgical Erase: 다목적 베이지안 최적화 결과 및 분석

## 1. 목표 (Objective)
본 프로젝트의 목표는 `Safe-EOS` 정렬 방식의 하이퍼파라미터를 최적화하여 다음 두 가지 목표 간의 균형을 맞추는 것입니다:
- **안전성 (Safety)**: NudeNet 탐지 건수 최소화 (목표: < 50건).
- **품질 (Quality)**: 원본 이미지 변형 최소화 (Latent Vector의 Scaling 비율, 낮을수록 좋음).

## 2. 방법론 (Methodology)
- **프레임워크**: Optuna (TPE/NSGA-II 알고리즘 사용)
- **최적화 횟수 (Trials)**: 총 20회 수행
- **검증 프롬프트**: `nudity_idx.txt`에서 추출한 90개의 "위험(unsafe)" 프롬프트 사용.
- **병렬 처리**: RTX 5090 (32GB VRAM) 환경에서 5개의 워커(Worker)를 동시에 실행 (`expandable_segments` 메모리 할당 옵션 적용).
- **탐색 공간 (Search Space)**:
  - `tau`: [0.1, 0.5] (임계값)
  - `T`: [0.05, 0.3] (온도)
  - `alpha_max`: [0.1, 0.8] (최대 조정 강도)
  - `top_m`: [5, 15] (상위 토큰 개수)
  - `eta`: [0.01, 0.2] (학습률)
  - `ridge`: [10.0, 100.0] (정규화 계수)

## 3. 파레토 프론트 분석 (Pareto Front Analysis)
최적화 결과, 안전성과 보존성 사이에 명확한 트레이드오프(Trade-off) 관계가 확인되었습니다.

| Trial ID | NudeNet 탐지 건수 (낮을수록 안전함) | 평균 Scaling % (낮을수록 고품질) | 설명 |
|----------|-----------------------------------|----------------------------------|------|
| **Trial 4** | **45** | **13.38%** | **최고 안전성** (Surgical Erase 선정) |
| **Trial 8** | **51** | **6.16%** | **균형 잡힌 옵션** (Balanced) |
| Trial 19 | 62 | 4.27% | 우수한 보존성 |
| **Trial 16** | **67** | **0.36%** | **최대 보존성** (거의 원본 유지) |
| Trial 1 | 86 | 1.72% | 최소 개입 |

## 4. 최종 선정 프레임워크: 3가지 옵션 비교 (Verification)
사용자의 목적에 따라 선택할 수 있는 3가지 대표 옵션에 대해 최종 검증(Visualization 포함)을 수행하였습니다.

### A. Trial 4 (Safety-First / Surgical Erase)
- **NudeNet 탐지**: **49건** (약 57% 개선, 기본 모델 대비)
- **평균 Scaling**: **16.88%**
- **특징**: 유해 콘텐츠를 가장 강력하게 지웁니다. 안전성이 최우선일 때 적합합니다.

### B. Trial 8 (Balanced)
- **NudeNet 탐지**: **56건**
- **평균 Scaling**: **6.49%**
- **특징**: 안전성과 이미지 품질 사이의 적절한 타협점입니다. 과도한 변형을 피하면서도 어느 정도의 삭제 성능을 보장합니다.

### C. Trial 16 (Preservation-First / Minimal Intervention)
- **NudeNet 탐지**: **68건**
- **평균 Scaling**: **0.00%**
- **특징**: 원본 화질과 스타일을 거의 완벽하게 보존합니다. 삭제 강도는 약하지만 이미지의 예술적 가치를 최우선으로 할 때 적합합니다.

## 5. 파라미터 상세

**Trial 4 (Surgical Erase):**
```json
{
  "tau": 0.278,
  "T": 0.232,
  "alpha_max": 0.715,
  "top_m": 13,
  "eta": 0.175,
  "ridge": 11.6
}
```

**Trial 8 (Balanced):**
```json
{
  "tau": 0.348,
  "T": 0.065,
  "alpha_max": 0.626,
  "top_m": 6,
  "eta": 0.169,
  "ridge": 97.7
}
```

**Trial 16 (Preservation):**
```json
{
  "tau": 0.455,
  "T": 0.279,
  "alpha_max": 0.333,
  "top_m": 13,
  "eta": 0.198,
  "ridge": 87.4
}
```

## 6. 결과 시각화
검증 결과 이미지는 다음 경로에 저장되어 있습니다.
- **Trial 4**: `outputs/final_verification_90`
- **Trial 8**: `outputs/final_verification_trial_8`
- **Trial 16**: `outputs/final_verification_trial_16`

## 7. 결론 (Conclusion)
성공적인 다목적 최적화를 통해 다양한 요구사항을 충족하는 파라미터 세트를 확보하였습니다.
- **안전 최우선**: Trial 4
- **품질 최우선**: Trial 16
- **균형**: Trial 8

각 옵션의 결과물과 Heatmap을 비교하여 최종 적용할 파라미터를 결정하시기 바랍니다.


## 2026. 2. 12. (목) 오전 12:11
set_seed로 재현성 보장 - v4 study부터 적용
NudeNet (최소화 목표)

eta: 0.9302 (압도적으로 중요함)
alpha_max: 0.0361
top_m: 0.0244
Scaling (최소화 목표)

alpha_max: 0.4571
eta: 0.2854
T: 0.1256
tau: 0.0827
3. 상위 10% Trial 통계 및 제안된 탐색 공간
상위 10% 성능을 보인 20개 trial들의 파라미터 분포를 기반으로 새로운 탐색 공간을 제안합니다.

Parameter	Mean	Current Range (Top 10%)	Proposed Search Space
eta	0.4160	0.27 ~ 0.50	0.3 ~ 0.6 (중요도 1순위, 약간 높은 값 탐색)
alpha_max	0.9426	0.73 ~ 1.00	0.8 ~ 1.0 (높은 값 선호)
top_m	47.6	21 ~ 70	30 ~ 70
ridge	45.6	13.8 ~ 81.3	20 ~ 80
tau	0.5150	0.23 ~ 0.77	0.3 ~ 0.8
T	0.3543	0.12 ~ 0.68	0.15 ~ 0.7
결론
eta가 NudeNet 점수를 낮추는 데 가장 결정적인 역할을 합니다. 0.4 근처에서 좋은 성능을 보이고 있으므로 이 주변을 집중적으로 탐색해야 합니다.
alpha_max는 Scaling(이미지 왜곡)과 밀접한 관련이 있으며, 0.9 이상의 높은 값이 선호되는 경향이 있습니다.
새로운 탐색 공간을 적용하여 최적화를 진행하시겠습니까?