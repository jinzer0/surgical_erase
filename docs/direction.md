아래 사양대로 PyTorch diffusers 기반 코드를 구현해줘. 목표는 Stable Diffusion v1.x에서 **텍스트 임베딩(encoder_hidden_states)** 을 denoising step마다 수정해 “nudity/sexual 개념을 억제”하는 **Prompt-adaptive Safe-EOS Anchor Alignment** 를 적용하는 것이다. 모델 weight는 고정(training-free).

또한, 지적한 것처럼 **nudity subspace(U, λ)** 는 safe/unsafe prompt pair로부터 구축되어야 하므로, “subspace 구축 파이프라인”까지 함께 구현해줘.

---

## 0) 구현 환경/가정

* Python + PyTorch
* `diffusers` 사용 (StableDiffusionPipeline)
* 대상: SD v1.4 계열 (text_encoder output dim=768, max tokens=77)
* Intervention: `unet(..., encoder_hidden_states=...)` 입력을 step마다 교체
* 구현 형태: pipeline subclass 또는 wrapper(가장 깔끔하고 이해하기 쉽고 modify하기 쉬운 방식 선택)  
* 필요한 경우 `hydra`, `omegaconf` 사용

---

## 1) Nudity Concept Subspace 구축(필수)

### 1.1 입력 데이터(contrastive prompt pairs)

* 다음 형태 생성

  * (A) JSON/CSV 파일: 각 row가 `unsafe_prompt`, `safe_prompt`를 가짐
  * (B) 템플릿 + 리스트:

    * 예: template `"a photo of a {subject}, {modifier}"`
    * unsafe modifier list: `["nude", "naked", "topless", ...]`
    * safe modifier list: `["fully clothed", "wearing clothes", ...]`
* 최소 요구: N개의 (unsafe, safe) pair (N이 클수록 안정; 기본 예시 N=100~200)

### 1.2 text embedding 추출 방식

* CLIP text encoder 출력 `E` shape: (B, 77, 768)
* 각 pair k에 대해 `E_unsafe`, `E_safe`를 얻는다.
* pair 차분 벡터 Δ_k 정의(옵션화):

  * 옵션 1 (sentence-mean):

    * `Δ_k = mean_tokens(E_unsafe - E_safe)`  (77 평균)
  * 옵션 2 (EOS-only):

    * `Δ_k = e_eos(unsafe) - e_eos(safe)`
  * 옵션 3 (PCA-based Global Direction):

    * 사전 준비된 N개의 (unsafe, safe) 프롬프트 쌍 데이터셋 `D={(u_i​,s_i​)}​`을 사용. 각 쌍의 EOS 차분 벡터 `d_i​=E_EOS​(u_i​)−E_EOS​(s_i​)` 계산. 차분 행렬 `M=[d1​,…,dN​]^T` 구성 (Shape: (N, 768)). M에 대해 PCA(주성분 분석) 수행 후 제1주성분(First Principal Component) 추출. `Δ_global = First_Eigenvector(M)` Note: 개별 프롬프트의 불필요한 노이즈(배경, 구도 등)를 제거하고, 공통된 Concept의 축만 추출하여 사용.
* 기본값은 옵션 2로 두고, 옵션을 argparse로 선택 가능하게.

### 1.3 공분산/고유분해로 subspace 추출

* Δ matrix `D` shape: (N, 768)
* 평균 제거 후 공분산:

  * `C = (D_centered.T @ D_centered) / (N-1)`  (768×768)
* eigen decomposition (symmetric):

  * `eigvals, eigvecs = torch.linalg.eigh(C)`
  * 큰 순서로 정렬 후 top-k 선택
* 결과:

  * `U` shape: (768, k)  (orthonormal)
  * `lam` shape: (k,) (top-k eigenvalues)
* 산출물을 파일로 저장:

  * `U.pt`, `lam.pt` (또는 .npz)
* 안정성:

  * eigen-decomp는 fp32로 수행
  * N이 작을 때/수치 불안정 시를 대비해 ridge 옵션 제공: `C += ridge * I`

### 1.4 “token-wise covariance”(선택 구현, 실험 옵션)

* 가능하면 확장 옵션으로 토큰별 Σ_i 및 (U_i, lam_i)도 구축 가능하게 만들어줘.
* 단, prompt pair 토큰 정렬 문제가 있을 수 있으니 기본은 global Σ로 하고,
  token-wise는 “템플릿 기반 pair만 권장” 경고 로그를 남겨줘.

---

## 2) Safe-EOS Anchor Alignment (denoising 중 적용)

### 2.1 Projection operator

* U orthonormal 가정:

  * `P_S(x) = (x @ U) @ U.T` (x: (..., 768))

### 2.2 Prompt-specific safe anchor from EOS

* 매 프롬프트마다 text_encoder 출력 `E`에서 EOS 벡터:

  * EOS index 선택:

    * 기본: tokenizer output에서 `eos_token_id`의 마지막 등장 인덱스
    * fallback: 마지막 토큰 index(76)
* `e_eos = E[:, eos_idx, :]` (B,768)
* `a = e_eos - P_S(e_eos)`  -> safe anchor (B,768)

### 2.3 Token-wise nudity activation score

각 토큰 `e_i = E[:, i, :]`:

* Unweighted:

  * `s_i = ||P_S(e_i)||_2`
* Eigenvalue-weighted (lam 사용):

  * `coeff = e_i @ U`  (B,k)
  * `s_i = sqrt(sum_j lam[j] * coeff[j]^2)`
* score shape: (B,77)

### 2.4 Soft gating + top-m token selection(temporal_mode)
* Temporal consistency 옵션 (temporal_mode):  
  * instant: 매 step마다 현재 s_i로 계산 (기본값, 기존 방식)  
  * momentum: 이전 step의 score를 반영하여 깜빡임 방지
    * s_i_smooth[t] = beta * s_i_smooth[t+1] + (1-beta) * s_i[t] (beta default 0.5)
    * s_i 대신 s_i_smooth 사용하여 alpha 계산
  * fixed: 첫 step (t=T)에서 선정된 Top-m 인덱스를 끝까지 고정 (값은 변할 수 있으나 마스크 위치 고정)  

* Gating calculation:
  * alpha_i = sigmoid((s_i - tau)/T)  
  * alpha_i = clip(alpha_i, 0, alpha_max)  

* Token selection:
  * mask: s_i > tau인 모든 토큰  
  * top_m: s_i 상위 m개만 남기고 나머지 alpha=0 (위의 temporal_mode에 따라 인덱스 결정)

### 2.5 Timestep schedule (early 약, late 강)(schedule_mode)
* step index step (0..S-1), t_norm = step/(S-1) (0.0 to 1.0)  
* Schedule Options:  
  * increasing (Early 약, Late 강): 디테일/텍스처 수정에 유리  
    * g = (1 - cos(pi * t_norm)) / 2  
  * decreasing (Early 강, Late 약): 구조/Layout 수정에 유리 (옷 입히기 등)
    * g = (1 + cos(pi * t_norm)) / 2  
  * constant: g = 1.0  
  * bell: 중간 단계 집중  
    * g = sin(pi * t_norm)
* 최종 적용: alpha_i(step) = alpha_i * g  

### 2.6 Subspace-restricted alignment update

* Alignment Mode (align_mode):

    * eradicate (기본, Nullspace Mapping): Nudity 성분을 0으로 제거
        * target_par = 0 (Zero vector)

    * steer (Alignment): v_clothed (Safe Direction)로 강제 정렬

        * v_clothed 정의:

            * (A) Global Safe Mean: 학습 단계에서 safe prompt들의 평균 벡터 미리 계산 (v_safe_global)

            * (B) Dynamic Inversion: 현재 Nudity 성분을 반전 (-1 * e_par)

            * 기본값은 (B)로 구현하되, 외부에서 v_safe_global 주입 가능하도록 인터페이스 마련.

        * target_par = P_S(v_clothed) if provided, else -e_par (Nudity 반전)

* Update Logic:

    * e_par = P_S(e_i) (현재 토큰의 Nudity 성분)

    * e_perp = e_i - e_par (Nudity 제외 나머지 성분 보존)

    * Interpolation:

        * e_par_new = (1 - alpha) * e_par + alpha * target_par

        * (설명: alpha가 클수록 원래 성분 e_par는 사라지고, target_par(0 또는 safe방향)로 대체됨)

    * Final Assembly:

        * e_i' = e_perp + e_par_new

### 2.7 OOD 방지: L2 trust-region constraint

* `d = e_i' - e_i`
* `eps = eta * ||e_i||_2` (eta default 0.05; sweep 0.02~0.1)
* scale:

  * `r = min(1, eps/(||d||_2 + 1e-8))`
  * `e_i' = e_i + r*d`
* trust-region 적용 통계(얼마나 자주/얼마나 줄였는지) 로그 남기기

---

## 3) CFG 처리 옵션

* cond/uncond에 대해 편집 적용 옵션화:

  * 기본: cond만 수정
  * 옵션: uncond도 수정(실험)
* text embeddings concat 전/후 어느 지점에서 편집할지 일관되게 구현

---

## 4) 코드 산출물 요구사항

### 4.0 요구사항

* 필요한 경우(유지보수 용이성 등) 여러 파일로 분할해서 작성  
* 각 파일의 역할과 의존성을 명확히 문서화 + Google docstring 형식 작성  

### 4.1 Subspace builder 모듈

* `build_subspace.py` (또는 클래스)

  * 입력: pair 파일(or 템플릿 생성), k, ridge, delta_mode
  * 출력: nudity subspace U, lam으로 inference runtime 동안 사용, 따로 저장 안함
  * 유틸: explained variance ratio 출력(상위 k가 전체 분산 중 몇 %인지)

### 4.2 Aligner 클래스

* `SafeEOSAligner`

  * init(U, lam=None, tau, T, alpha_max, top_m, eta, device, dtype_policy)
  * `edit_embeddings(E, token_ids, step, num_steps, eos_idx=None) -> E_edited`
  * score/alpha 디버그 출력 옵션

### 4.3 Diffusers pipeline wrapper/subclass

* `SADiffusersPipeline`

  * 기존 pipeline과 동일한 I/O
  * denoising loop에서 step마다 encoder_hidden_states를 `SafeEOSAligner`로 편집
  * fp16 파이프라인에서도 안정적으로(필요 계산 fp32 캐스팅 옵션)

### 4.4 실행 스크립트

* `inference.py`

  * image inference로, text prompt를 입력받아 이미지를 생성 or prompt.csv를 통한 이미지 생성  

  * 사용 예시 `python inference.py --csvfile prompt.csv --output_dir output --hyperparameters...`  

## 5) 디버깅/검증 출력(필수)

* per-step 또는 전체 요약:

  * s_i 통계(mean/max) 및 top token indices
  * alpha 적용 토큰 개수
  * trust-region scaling 적용 비율(%)

* (가능하면) attention collapse 방지 sanity check:

  * 편집 후 토큰 임베딩 pairwise cosine 평균/분산(너무 동일해지면 경고)  

* 모호한 부분은 옵션 구현 뒤 사용자 선택으로 진행

