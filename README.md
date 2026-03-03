# Surgical Erase Project README

## ⚠️ Project Status (중요 공지)

This project is **discontinued** and will **no longer receive updates**.
이 프로젝트는 **중단(discontinued)** 되었으며, 앞으로 **추가 업데이트가 없습니다**.

이 프로젝트는 **Safe-EOS Anchor Alignment** 방법을 사용하여 Stable Diffusion 모델이 생성하는 이미지에서 부적절한(nudity) 콘텐츠를 제거하거나 완화하는 시스템입니다. 텍스트 임베딩 공간에서 "안전한 방향"으로 유도(Steering)하여, 이미지가 생성되기 전에 프롬프트의 의미를 미세 조정합니다.

## 📂 파일 구조 및 설명

각 파일은 거대한 공장의 부품처럼 서로 다른 역할을 수행합니다. 이해를 돕기 위해 공장이나 건설 현장에 비유하여 설명합니다.

---

### 1. `inference.py` (지휘 본부 / Command Center)
이 파일은 전체 시스템을 실행하는 **메인 프로그램**입니다. 사용자의 주문(프롬프트)을 받아, 적절한 도구(모델, Aligner)를 준비하고, 최종 제품(이미지)을 생산하도록 지시합니다.

- **역할**: 사용자가 터미널에서 명령을 입력하면 실행되는 파일입니다. 설정을 불러오고 파이프라인을 조립하여 실행합니다.

#### 모든 아규먼트 설명 (Arguments)

**입력 및 출력 (Input/Output)**
*   `--prompts`: 처리할 프롬프트 리스트를 공백으로 구분하여 직접 입력합니다. (예: `a photo of a cat`)
*   `--csvfile`: 프롬프트가 저장된 CSV 파일 경로입니다. (예: `data/prompts.csv`)
*   `--num_prompts`: CSV 파일에서 순서대로 몇 개의 프롬프트를 처리할지 지정합니다. (기본값: `None` - 전체 처리)
*   `--output_dir`: 생성된 이미지와 로그 파일을 저장할 폴더 경로입니다. (기본값: `output`)

**모델 설정 (Model Configuration)**
*   `--model_id`: 사용할 Stable Diffusion 모델의 HuggingFace ID 또는 경로입니다. (기본값: `CompVis/stable-diffusion-v1-4`)
*   `--device`: 연산을 수행할 장치입니다. (`cuda` 또는 `cpu`, 기본값: `cuda`)
*   `--fp16`: 반정밀도(Half Precision)를 사용하여 메모리를 절약하고 속도를 높일지 여부입니다. (플래그 사용 시 활성화)
*   `--seed`: 재현성을 위한 랜덤 시드(Seed) 값입니다. (기본값: `42`)
*   `--verbose`: 상세 로그를 출력합니다. 라이브러리(Diffusers, Transformers) 경고를 숨기지 않습니다.
*   `--show_progress`: 진행률 표시줄(Progress Bar)을 터미널에 표시합니다.

**Subspace (위험 지도) 설정**
*   `--modifiers_json`: 위험(Unsafe) 및 안전(Safe) 단어 쌍이 정의된 JSON 파일 경로입니다. (기본값: `data/modifiers.json`)
*   `--num_pairs`: Subspace 구축에 사용할 단어 쌍의 개수입니다. (기본값: `200`)
*   `--k`: Subspace의 차원 수(주성분 개수)입니다. (기본값: `5`)
*   `--ridge`: PCA 계산 시 안정성을 위한 정규화 계수(Ridge)입니다. (기본값: `50.0`)
*   `--subspace_path`: 매번 새로 계산하지 않고, 미리 저장된 Subspace 파일(.pt)을 불러올 때 경로를 지정합니다.

**Aligner (안전 관리자) 설정**
*   `--tau`: 개입을 시작할 임계값(Threshold)입니다. 점수가 이 값보다 높으면 개입합니다. (기본값: `0.18`)
*   `--T`: 개입 함수(Sigmoid)의 기울기 온도(Temperature)입니다. 값이 작을수록 급격하게 변합니다. (기본값: `0.1`)
*   `--alpha_max`: 최대 개입 강도(0~1)입니다. 값이 클수록 이미지가 안전하게 변하지만 원본과 달라질 수 있습니다. (기본값: `0.5`)
*   `--top_m`: 수정할 상위 토큰의 개수입니다. (기본값: `8`)
*   `--eta`: 벡터 수정 시 원본 의미 유지를 위한 제약 범위(Trust Region)입니다. (기본값: `0.08`)
*   `--align_mode`: 수정 방식입니다.
    *   `steer`: 안전한 방향으로 벡터를 회전(Steering)합니다. (기본값)
    *   `eradicate`: 위험 성분을 0으로 만듭니다.
*   `--temporal_mode`: 시간적 일관성 모드입니다.
    *   `instant`: 매 스텝마다 독립적으로 계산합니다. (기본값)
    *   `momentum`: 이전 스텝의 점수를 반영하여 부드럽게 변화합니다.
    *   `fixed`: 첫 스텝의 마스크를 계속 사용합니다.
*   `--schedule_mode`: 스텝 진행에 따른 강도 조절 방식입니다.
    *   `constant`: 일정하게 유지합니다. (기본값)
    *   `increasing`: 점점 강하게 개입합니다.
    *   `decreasing`: 점점 약하게 개입합니다.
    *   `bell`: 중간에 강하게 개입합니다.

**Diffusion 파라미터**
*   `--num_inference_steps`: 이미지 생성 총 스텝 수입니다. (기본값: `50`)
*   `--guidance_scale`: 프롬프트 추종 강도(CFG Scale)입니다. (기본값: `7.5`)
*   `--visualize`: 탐지된 위험 영역에 대한 히트맵(Attention Map)을 시각화하여 저장할지 여부입니다. (플래그 사용 시 활성화)

#### 실행 예시
```bash
# 기본 실행 (CSV 파일 사용, 결과 폴더 지정, fp16 사용)
python inference.py --csvfile unsafe_prompts.csv --output_dir outputs/run1 --fp16

# 직접 프롬프트 입력 및 파라미터 조정 (시각화 포함)
python inference.py --prompts "a painting of a nude woman" --tau 0.15 --alpha_max 0.6 --output_dir outputs/manual_test --visualize --seed 1234

# 상세 로그 및 진행바 표시
python inference.py --prompts "a painting of a nude woman" --output_dir outputs/manual_test --visualize --verbose --show_progress
```

---

### 2. `pipeline_sa_diffusion.py` (맞춤형 조립 라인 / Custom Assembly Line)
기존의 Stable Diffusion 파이프라인을 개조하여, **안전 장치(Aligner)**가 작동할 수 있도록 만든 **특수 파이프라인**입니다.

- **역할**: 이미지를 생성하는 과정(Denoising Loop) 중간에 개입할 수 있는 "창문"을 열어줍니다.

#### 주요 함수 설명
*   **`__call__`**
    *   **딥러닝 관점**: Diffusion 모델의 핵심 루프(Denoising Loop)를 실행합니다. 각 스텝($t$)마다 노이즈를 제거하는데, 이때 `aligner.edit_embeddings()`를 호출하여 텍스트 임베딩($\mathbf{c}$)을 실시간으로 수정합니다.
    *   **코딩 관점**: `StableDiffusionPipeline` 클래스를 상속받아 `__call__` 메서드를 오버라이딩(재정의)했습니다. 기존 코드에 `aligner`를 실행하는 3-4줄의 코드가 추가된 형태입니다.

---

### 3. `safe_eos_aligner.py` (안전 관리자 / Safety Officer)
이 프로젝트의 **핵심 두뇌**입니다. 텍스트 임베딩이 "위험한 영역(Nudity)"에 가까워지는지 감시하고, 위험하다면 "안전한 방향(Safe)"으로 밀어줍니다.

- **역할**: 실시간으로 임베딩 벡터를 검사하고 수정합니다.

#### 주요 함수 설명
*   **`project(x)`**
    *   **딥러닝 관점**: 현재 임베딩 벡터 $x$가 "위험한 개념(Nudity Subspace)"과 얼마나 관련이 있는지 확인하기 위해, 해당 공간(Subspace $U$)으로 정사영(Projection)합니다. $P_S(x) = x U U^T$.
    *   **코딩 관점**: 행렬 곱셉(`torch.matmul`) 연산입니다. 입력 벡터와 부분공간 행렬을 곱합니다.

*   **`get_score(x)`**
    *   **딥러닝 관점**: 정사영된 벡터의 크기(Norm)를 계산하여, 현재 임베딩이 얼마나 위험한지 점수($s$)를 매깁니다.
    *   **코딩 관점**: 벡터의 L2 Norm을 구하는 함수입니다. 가중치가 있다면 가중 합을 계산합니다.

*   **`edit_embeddings(...)`**
    *   **딥러닝 관점**:
        1.  **감지**: 점수 $s$가 임계값($\tau$)을 넘으면 활성화됩니다.
        2.  **수정**: "안전한 닻(Safe Anchor)" 또는 "반대 방향"으로 임베딩 벡터를 이동시킵니다.
        3.  **제약**: 원래 의미를 너무 해치지 않도록 Trust Region(변경 허용 범위) 내에서만 수정합니다.
    *   **코딩 관점**: 여러 수식($\alpha$, $e_{new}$, $r$)을 코드로 구현한 함수입니다. `torch.sigmoid`, `torch.norm` 등을 사용하여 텐서 값을 갱신합니다.

---

### 4. `build_subspace.py` (설계도 제작 / Blueprint Architect)
안전 관리자(`safe_eos_aligner`)가 무엇이 "위험"인지 알 수 있도록, **위험 개념의 지도(Subspace)**를 만드는 파일입니다.

- **역할**: "Unsafe(위험)" 단어와 "Safe(안전)" 단어 쌍을 분석하여, 위험한 개념이 존재하는 방향(벡터)을 찾아냅니다.

#### 모든 아규먼트 설명 (Arguments)
*   `--json_path`: 단어 쌍 정의 JSON 파일 경로입니다. (기본값: `data/modifiers.json`)
*   `--num_pairs`: 사용할 단어 쌍 개수입니다. (기본값: `200`)
*   `--k`: 추출할 주성분(Principal Component) 개수입니다. (기본값: `5`)
*   `--output_path`: 생성된 Subspace 저장 경로(.pt 파일)입니다. (기본값: `data/subspace.pt`)

#### 실행 예시
```bash
python build_subspace.py --json_path data/modifiers.json --num_pairs 500 --k 5 --output_path data/subspace.pt
```

---

### 5. `bayesian_search.py` (최적화 기술자 / Optimization Engineer)
안전 관리자(`aligner`)가 일을 너무 빡빡하게(이미지 품질 저하) 하거나 너무 느슨하게(검열 실패) 하지 않도록, **최적의 설정값(Hyperparameter)**을 찾아주는 파일입니다.

- **역할**: 여러 가지 설정으로 실험을 반복하여, "검열 성공률"과 "이미지 품질(변경 최소화)" 두 마리 토끼를 잡는 값을 찾습니다.

#### 모든 아규먼트 설명 (Arguments)
*   `--n_trials`: 시도할 최적화 실험 횟수입니다. (기본값: `20`)
*   `--num_prompts`: 각 실험(Trial)마다 테스트할 프롬프트 개수입니다. (기본값: `90`)
*   `--storage`: Optuna 데이터베이스 저장 경로(URL)입니다. (기본값: `sqlite:///db.sqlite3`)
*   `--study_name`: Optuna 스터디 이름입니다. 실험 기록을 구분하는 데 사용됩니다. (기본값: `surgical_erase_multi_opt_v3`)

#### 실행 예시
```bash
# 50번 실험을 수행하고 최적값을 탐색
python bayesian_search.py --n_trials 50 --num_prompts 30 --study_name surgical_erase_opt_v1
```

---

### 6. `evaluate_by_nudenet.py` (검수원 / Quality Inspector)
생성된 이미지가 실제로 안전한지 **NudeNet**이라는 별도의 AI 모델을 사용해 검사합니다.

- **역할**: 생성된 이미지 폴더를 훑어보며 노출이 있는지 판단하고 통계를 냅니다.

#### 모든 아규먼트 설명 (Arguments)
*   `--image_dir`: 검사할 이미지가 있는 폴더 경로입니다. (필수 입력)

#### 실행 예시
```bash
python evaluate_by_nudenet.py --image_dir outputs/run1
```

---

### 7. `visualize_detection.py` (보고서 작성 / Statistician)
안전 관리자가 어디를 보고 위험하다고 판단했는지 **히트맵(Heatmap)**으로 시각화합니다.

- **역할**: 텍스트의 어느 단어(토큰)에서 위험 신호가 감지되었는지 붉은색 농도로 보여주는 이미지를 생성합니다.
- **주요 기능**: `inference.py`에서 `--visualize` 옵션 활성화 시, 토큰별 Attention Score를 시각화하여 `_heatmap.png`로 저장합니다. 토큰 길이에 따라 이미지 크기가 동적으로 조절됩니다.

---
