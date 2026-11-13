# 공식 기반 재구축 및 최적화 재조사

## 범위와 상태

v0.2는 기존 신경망 포팅을 더 고치는 대신, 고정된 공식 Pixal3D의 클래스와
함수를 직접 호출하는 얇은 CUDA orchestration layer다. 공식 checkout은 수정하지
않고 외부 디렉터리에 설치한다. 기존 MLX 저장소도 수정하지 않는다.

CPU 계약 테스트만 완료한 단계다. **native 이미지 빌드, L4/T4 생성, 원본 CLI와의
수치·품질 동등성, 성능 개선은 검증하지 않았다.** 과거 7 GiB T4 측정은 legacy
런타임의 숫자이며 새 런타임의 기대값으로 재사용하지 않는다.

## 새 기본 경로

1. `pipeline.Pipeline`: 입력·카메라 정규화, source/model identity, 외부 worker 실행,
   완료 결과 검증. 기본값은 1024 geometry, 4096 texture, 1M face target이다.
2. `upstream_worker`: official `init_pipeline`, `preprocess_image`, MoGe camera helper,
   `run`/`load_views`/`run_mv`, `o_voxel.postprocess.to_glb`를 직접 호출한다.
3. 원본 texture NAF **1024**, BF16 flow / FP16 decoder torso, 모든 원본 sampler
   설정, CuMesh hole filling, remesh/UV/PBR 경로를 유지한다.
4. 모델·backend를 바꾸지 않는 dependency injection만 사용한다: pinned DINO/
   BiRefNet 경로, local NAF repository/검증된 원본 가중치, local pipeline config.
   원본 소스나 모델 forward는 패치하지 않는다.
5. worker는 별도 프로세스에서 실행한다. CUDA 전역 backend 설정, upstream의
   `src`/`inference` 모듈 이름, monkey-patch 수명이 서버 프로세스에 누출되지 않는다.

기본 low-VRAM + SDPA는 **공식이 이미 제공하는 선택지**다. 새 알고리즘 최적화가
아니다. 원본 CLI의 standard/1536 기본과는 다르므로 설정은 보고서에 명시한다.
`--standard`와 1536 선택도 가능하다. 공식 1536 cascade의 token 기반 resolution
조정은 upstream 동작 그대로이며 요청/실제 grid와 fallback 여부를 보고한다.
`max_num_tokens`는 최종 voxel hard limit이나 OOM 방지 보장이 아니다.

`--no-rembg`는 원본 matting/crop을 우회하고 주어진 framing에서 alpha를
검정 배경에 합성한다. 원본 자동 전처리와 동일하다고 주장하지 않는다. mapped
입력은 crop하지 않으며 원본 `load_views`로 변환한다. MV별 가중치를 사용한다.

## T4는 별도 호환화 프로젝트

원본 flow는 BF16을 사용한다. T4는 이를 native BF16 Tensor Core 경로로 실행할
수 있는 GPU가 아니다. 따라서 SM75에서 FP16/FP32로 자동 교체하면 원본 그대로의
baseline이 아니게 된다. 새 기본 worker는 SM80 미만을 **명시적으로 거절**한다.

초기 Modal 선언은 L4/SM89로 변경했지만 실제 배포하지 않았다. GPU를 빌리거나
업그레이드하는 자동 동작은 없다. Colab 무료 T4는 새 기준선 실행 대상이 아니다.
NATTEN/FlexGEMM/CuMesh/o_voxel을 SM75에서 실행 가능한지, 어떤 precision으로
overflow 없이 실행되는지는 각각 별도 검증해야 한다.
[NVIDIA Ampere의 추가 Tensor Core 형식](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#improved-tensor-core-operations),
[FlashAttention GPU/precision 지원](https://github.com/Dao-AILab/flash-attention#nvidia-cuda-support).

CUDA 12.4 devel + Torch 2.6 + source build를 선언했다. 원본 Spaces의 cp310/H-series
wheel을 Python 3.12/L4에 재사용하지 않는다. native dependency source와 NATTEN
0.21.0을 고정했지만, 이 조합의 빌드 성공을 아직 확인하지 않았다.
[공식 설치](https://github.com/TencentARC/Pixal3D/blob/f7cf38429b0bd264f1995f0f8743a88b1c728b94/README.md#installation),
[NATTEN 빌드 대상 지정](https://natten.org/install/).

## 최적화 후보 재분류

아래는 **조사 결과 및 실험 우선순위**다. 새로운 최적화는 아직 기본 경로에
적용하지 않았다. legacy에서 좋았던 chunk 수치도 native backend에는 이식하지 않는다.

| 우선순위 | 후보 | 근거 / 목표 | 통과 조건 |
| --- | --- | --- | --- |
| P0 | 원본 baseline/fixture 확보 | 네트워크·texture NAF·remesh를 바꾸지 않은 기준선 필요 | 동일 입력/camera/seed의 원본 CLI와 stage/latent/raw mesh/PBR 비교 |
| P1 | DINO 공유 + 512/1024 feature 재사용 | 공식 extractor 4개가 각각 backbone을 소유하고 같은 크기 특징을 재계산 | 전처리/weight/dtype 동일, cached/uncached 특징 수치 비교; end-to-end 시간 포함 |
| P1 | active 좌표 직접 projection | 원본은 R³ 전체 투영 후 sparse 좌표만 선택 | 1024 grid·카메라·border/interpolation 일치, dense/sparse 전체 pretrained 비교 |
| P1 | 원본 NAF의 selected-query/타일 평가 | texture 1024의 전체 1024-channel 특징 맵이 큼 | **target 1024 유지**, 원본 NATTEN의 경계·dilation·4-corner 비교; native kernel 우선 |
| P2 | 원본 BF16 flow의 static conditioning cache | per-block projection/global K/V는 동일 conditioning에서 재사용 가능 | cache 0 기준과 비교, timestep·latent 의존값 제외, cache 준비 시간 포함 |
| P2 | native sparse backend/algorithm autotuning | FlexGEMM은 여러 implicit/explicit/split-K 구현 제공 | decoder 실제 좌표·채널·dtype별 profiler; 수치 오차/peak/전체 decode 함께 측정 |
| P2 | low-VRAM vs 제한적 model residency | 원본에서 이미 지원; host/device 전송 비용과 GPU 상주 집합 tradeoff | 동일 장치에서 교차 순서 반복; OOM 실패를 결과에서 제외하지 않음 |
| P3 | compile/CUDA graphs/동기화 감소 | shape가 정적인 flow loop에만 검토 가치 | compile/warmup 포함, dynamic sparse 단계 분리, correctness/메모리 확인 |
| 별도 | T4 precision/backend 호환화 | BF16 기준선과는 다른 구현 | 연산별 range/overflow, native kernel SM75 지원, end-to-end 품질 검증 |

FlexGEMM이 이미 native sparse convolution을 제공하므로 gather/GEMM 구현을 다시
기본으로 만드는 것은 목표가 아니다. 원본 CFG sampler도 `guidance_strength==1`
일 때 negative forward를 생략한다. 이미 있는 최적화를 새 성과로 세지 않는다.
원본 sparse 표현, low-VRAM, 선택 가능한 attention backend도 그대로 활용한다.
[원본 CFG 분기](https://github.com/TencentARC/Pixal3D/blob/f7cf38429b0bd264f1995f0f8743a88b1c728b94/pixal3d/pipelines/samplers/classifier_free_guidance_mixin.py),
[FlexGEMM 알고리즘](https://github.com/JeffreyXiang/FlexGEMM/tree/6dd94a859c26ee8246888502eada3dd8ad85532e).

품질 옵션인 geometry/texture/face 수 축소와 remesh 제거는 backend 최적화와
분리한다. 원본 4096 texture를 1024로 줄인 시간을 “동일 품질 가속”으로 보고하지 않는다.

## 측정 도구와 다음 acceptance

`scripts/benchmark_upstream.py`는 기본 dry-run이다. `--execute`일 때만 로컬 GPU에서
동일 입력으로 low-VRAM/standard SDPA를 교차 순서로 각 3회 실행한다. 별도
worker/asset 단위라 kernel/autotune cache가 남을 수 있으며 warm-forward 벤치가 아니다.
한 번 실패하면 즉시 중단하고 결과를 보존한다. Modal 호출은 없다.

report는 load / preprocess-camera / denoise-decode / CUDA remesh-PBR의 시간과
allocated/reserved peak, 각 구간 끝 host RSS를 기록한다. RSS는 peak 샘플이 아니고,
PyTorch 수치는 외부 allocator/context의 모든 메모리를 포함하지 않는다.
세부 SS/LR/HR/texture profiler와 device-wide peak 측정은 다음 acceptance에 추가해야 한다.

`--save-reference`는 native shape/texture latent와 raw mesh/voxel tensor를 CPU에
저장한다. 작은 toy contract가 아닌 pretrained fixture로 위 후보들을 한 항목씩
검증하는 데 사용한다. original CLI와 비교할 때 camera/초기 noise/전처리/후처리
설정을 고정하고, 최종 GLB hash만으로 품질·수치 동등성을 판정하지 않는다.

다음 실제 검증 순서는 native 이미지 빌드 → CPU weight prefetch → 공식 CLI 및
wrapper의 동일 1024 단일 이미지 생성 → raw geometry/PBR 확인 → 반복 baseline이다.
이 단계들은 아직 실행하지 않았으며 별도 GPU 사용 결정이 필요하다.

## 단일 runtime과 과거 이력

이전 신경망 연산 모듈, CPU mesh/PBR exporter, 별도 runtime/CLI, T4 전용 노트북,
연산 벤치마크와 해당 구현의 테스트는 현재 트리에서 삭제했다. 별도 backend 선택이나
호환 runtime은 없다. 이전 코드는 Git 이력에만 남으며 과거 validation JSON은
역사적 증거로 보존한다. GPU/CPU phase 분할을 원본에 억지로 맞추지 않기 위해
Modal은 GPU에서 export까지 완료한 뒤 GLB를 제공한다.
