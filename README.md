# Surgical Erase

Stable Diffusion v1.x에서 텍스트 임베딩을 denoising step마다 수정해 nudity/sexual 개념을 약화시키는 실험용 저장소입니다. 학습 없이 `encoder_hidden_states`를 직접 편집하는 `Safe-EOS Anchor Alignment` 접근을 구현하고, subspace 구축, 이미지 생성, NudeNet 평가, Optuna 기반 탐색까지 한 저장소에 묶어두었습니다.

## 상태

이 프로젝트는 현재 중단된 상태이며 더 이상 유지보수되지 않습니다. 다만 코드와 실험 스크립트는 남아 있으므로, 같은 아이디어를 재현하거나 후속 실험의 출발점으로는 사용할 수 있습니다.

## 무엇을 하는가

이 저장소의 핵심 흐름은 아래와 같습니다.

1. `unsafe`와 `safe` 또는 `neutral` 프롬프트 쌍으로 nudity concept subspace를 구축합니다.
2. Stable Diffusion 추론 중 텍스트 임베딩을 step별로 수정해 unsafe 성분을 줄입니다.
3. 생성 결과를 NudeNet으로 평가합니다.
4. 필요하면 Optuna로 alignment 하이퍼파라미터를 탐색합니다.

## 기술 스택

- PyTorch
- diffusers
- transformers / CLIP text encoder
- pandas, numpy
- Optuna
- NudeNet
- onnxruntime

## 빠른 시작

이 저장소에는 `requirements.txt`나 환경 고정 파일이 없어서, 아래 명령은 현재 코드 import 기준으로 정리한 예시입니다.

```bash
conda create -n surgical-erase python=3.10
conda activate surgical-erase

conda install pytorch pandas numpy -c pytorch
pip install diffusers transformers accelerate optuna nudenet onnxruntime-gpu tqdm pillow
```

CPU 환경이라면 `onnxruntime-gpu` 대신 `onnxruntime`를 사용해야 합니다. Stable Diffusion 가중치는 Hugging Face에서 받아오므로, 모델 접근 권한과 로그인 상태가 필요할 수 있습니다.

## 기본 사용 흐름

### 1. Subspace 생성

`scripts/build_subspace.py`는 CLIP text encoder로 contrastive pair의 EOS 임베딩 차이를 모아 PCA 기반 subspace를 만듭니다.

```bash
python scripts/build_subspace.py \
  --json_path data/modifiers_v2.json \
  --num_pairs 200 \
  --k 5 \
  --output_path data/subspace.pt
```

기본 출력은 `U`, `lam`, 그리고 가능한 경우 `v_safe`를 포함한 `data/subspace.pt`입니다.

### 2. 단일 프롬프트 또는 배치 추론

미리 만든 subspace를 사용할 수도 있고, `--subspace_path`를 주지 않으면 추론 시점에 새로 계산합니다.

```bash
python scripts/run_inference.py \
  --prompts "a photo of a nude woman" \
  --subspace_path data/subspace.pt \
  --output_dir outputs/demo \
  --fp16 \
  --visualize
```

CSV 배치 입력도 가능합니다.

```bash
python scripts/run_inference.py \
  --csvfile data/prompts/unsafe_prompt315.csv \
  --num_prompts 20 \
  --subspace_path data/subspace.pt \
  --output_dir outputs/batch \
  --fp16
```

### 3. NudeNet 평가

```bash
python scripts/evaluate_nudenet.py --image_dir outputs/batch
```

실행 후 `outputs/batch_nudenet_result.log`와 `outputs/batch_nudenet_detect.json`이 생성됩니다.

### 4. 하이퍼파라미터 탐색

```bash
python scripts/optimize.py \
  --n_trials 50 \
  --num_prompts 315 \
  --study_name surgical_erase_multi_opt_v17
```

주의할 점:

- 현재 `bayesian_search.py`의 기본 Optuna storage는 SQLite가 아니라 PostgreSQL URL로 하드코딩되어 있습니다.
- 스크립트 내부에서 `data/prompts/unsafe_prompt4703.csv`, `data/prompts/nudity_idx.txt`, `data/modifiers_v3.json`을 사용합니다.

## 주요 스크립트

| 파일 | 역할 |
| --- | --- |
| `scripts/run_inference.py` | 전체 추론 진입점. subspace 로드 또는 생성, aligner 설정, Stable Diffusion 실행, 결과 저장 |
| `src/surgical_erase/pipelines/sa_diffusion.py` | diffusers pipeline을 감싸서 step마다 `encoder_hidden_states`를 수정하는 커스텀 파이프라인 |
| `src/surgical_erase/aligners/safe_eos_aligner.py` | subspace projection, token별 score 계산, gating, trust-region 제약, temporal/schedule 로직 담당 |
| `src/surgical_erase/subspace/builder.py` | CLIP text encoder로 prompt pair 차분을 수집하고 PCA 기반 nudity subspace를 생성 |
| `scripts/evaluate_nudenet.py` | 생성 이미지 폴더를 NudeNet으로 평가하고 로그 및 JSON 결과 저장 |
| `scripts/optimize.py` | alignment 파라미터를 Optuna로 탐색하는 실험 스크립트 |
| `src/surgical_erase/visualization/detection_viz.py` | heatmap, token trajectory, step analysis 시각화 저장 |
| `scripts/analyze_results.py`, `scripts/analyze_study.py` | Optuna 결과 후처리 및 분석 |

## 핵심 설정값

`scripts/run_inference.py` 기준으로 자주 만지는 옵션은 아래와 같습니다.

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--model_id` | `CompVis/stable-diffusion-v1-4` | 사용할 Stable Diffusion 체크포인트 |
| `--modifiers_json` | `data/modifiers_v2.json` | subspace 구축에 사용할 modifier 정의 |
| `--k` | `5` | subspace 차원 수 |
| `--ridge` | `50.0` | 공분산 안정화를 위한 ridge |
| `--tau` | `0.18` | intervention 활성화 임계값 |
| `--T` | `0.1` | sigmoid gate temperature |
| `--alpha_max` | `0.5` | 최대 개입 강도 |
| `--top_m` | `8` | 수정할 상위 토큰 개수 |
| `--eta` | `0.08` | trust-region 비율 |
| `--temporal_mode` | `instant` | `instant`, `momentum`, `fixed` 중 선택 |
| `--schedule_mode` | `constant` | `constant`, `increasing`, `decreasing`, `bell` 중 선택 |
| `--align_mode` | `steer` | `eradicate`, `steer`, `combined`, `eos_delta` 중 선택 |
| `--start_step` | `0` | intervention 시작 step |
| `--end_step` | `50` | intervention 종료 step |

## 데이터와 출력물

- `data/modifiers*.json`: unsafe/safe modifier, subject, template 정의
- `data/prompts/unsafe_prompt315.csv`, `data/prompts/unsafe_prompt4703.csv`: 실험용 프롬프트 CSV
- `data/prompts/nudity_idx.txt`: 탐색 실험에서 사용할 인덱스 목록
- `outputs/...`: 생성 이미지, heatmap, CSV 로그, NudeNet 평가 결과

## 주의사항

- 현재 README는 2026-03-15 기준으로 실제 코드 기본값과 파일 구성을 다시 맞춰 정리한 것입니다.
- `scripts/evaluate_nudenet.py`는 현재 `CUDAExecutionProvider`를 강제로 사용하도록 되어 있어 CPU-only 환경에서는 수정이 필요할 수 있습니다.
- 저장소 안에는 `*_backup.py`, 분석용 스크립트, 일회성 검증 파일이 함께 들어 있습니다. 모두 제품 코드라고 가정하면 안 됩니다.
- 별도 테스트 스위트나 재현 가능한 환경 lockfile은 포함되어 있지 않습니다.

## 권장 읽기 순서

처음 보는 경우에는 아래 순서가 가장 빠릅니다.

1. `scripts/run_inference.py`
2. `src/surgical_erase/aligners/safe_eos_aligner.py`
3. `src/surgical_erase/pipelines/sa_diffusion.py`
4. `src/surgical_erase/subspace/builder.py`
5. `scripts/evaluate_nudenet.py`

## Last Reviewed

2026-03-15
