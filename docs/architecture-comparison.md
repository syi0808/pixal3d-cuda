# 공식 Pixal3D / MLX / CUDA 아키텍처 비교

> v0.2 재구축 이전의 고정된 CUDA revision을 분석한 문서다.
> 현재 기본 경로는 [공식 runtime 기반](upstream-runtime.md)으로 변경되었다.

2026-11-13 코드 감사. 이번 작업은 GPU를 실행하거나 런타임을 변경하지 않았다.
공식 런타임의 실제 peak VRAM/속도는 아직 측정하지 않았으므로 아래 비교를
공식 대비 성능·품질 동등성 검증으로 해석하면 안 된다.

## 결론

**낮은 VRAM은 모델을 작게 바꾼 결과가 아니라, 모델 상주 수·특징 materialization·
중간 결과 수명·후처리 장치를 제한한 결과다.** MLX 포팅 과정에서 선택한 실행
정책이 CUDA에도 이어졌지만, 공식 구현에도 이미 모델별 CPU offload 모드가 있다.
따라서 “MLX에서 왔기 때문에 신경망 자체가 저메모리 구조다”라는 설명은 부정확하다.

또한 전부 의미 보존 최적화인 것은 아니다. **단일 이미지의 텍스처 NAF 설정,
기본 geometry/texture/face 수, 배경 제거와 메시 후처리는 공식 실행과 다르다.**
동일 체크포인트를 사용한다는 사실만으로 공식과 같은 결과라고 볼 수 없다.

## 비교 기준과 재현 범위

| 대상 | 고정한 revision |
| --- | --- |
| 공식 코드 `TencentARC/Pixal3D` | `f7cf38429b0bd264f1995f0f8743a88b1c728b94` |
| CUDA 감사 대상 | `924b56d0b6fbe4b7fec4c7e3263ec385db903b70` |
| MLX 읽기 전용 참조 | `56b4ac8ff9ac697f35593eb26962cea08ac46df6` |
| CUDA 단일 이미지 모델 pin | HF `0b31f9160aa400719af409098bff7936a932f726` |
| 조회 시점 HF `main` / CUDA MV pin | `b0cb2e1b794cab9aa0ac38a95d794a4d9337437f` |

공식 모델 카드의 `main`은 TRELLIS.2 기반 개선판이고 `paper`는 Direct3D-S2 기반
논문판이다. 이번 비교 대상은 **우리 체크포인트에 대응하는 TRELLIS.2 개선판**이다.
[공식 모델 설명](https://huggingface.co/TencentARC/Pixal3D/blob/b0cb2e1b794cab9aa0ac38a95d794a4d9337437f/README.md).

HF API의 두 revision을 비교해 단일 이미지용 7개 체크포인트의 LFS SHA-256,
7개 모델 JSON의 Git blob ID, `pipeline.json` blob ID가 모두 같음을 확인했다.
따라서 이번 차이를 “단일 이미지 모델 pin이 오래되어 다른 가중치라서”라고
설명할 근거는 없다. DINO는 공식 코드의 `camenduru` 배포 대신 pinned `facebook`
배포를 사용하므로 두 배포의 파일 동등성까지 검증했다는 뜻은 아니다.

숫자의 원본은 [감사 요약 JSON](evidence/architecture-audit-2026-11-13.json),
실제 CUDA 측정은 [기존 T4 보고서](evidence/modal-t4-chunks-2026-11-13.report.json)다.

## 1. 신경망의 큰 구조는 유지되어 있다

```text
이미지 + 카메라
  → DINOv3 / NAF 투영 conditioning
  → SS DiT → occupancy decoder → 32-grid sparse structure
  → LR shape DiT → shape decoder로 HR 좌표 예측
  → HR shape DiT → shape latent
  → shape-conditioned texture DiT → texture latent
  → shape decoder + shape subdivision을 공유하는 texture decoder
  → flexible dual-grid mesh + PBR GLB
```

| 항목 | 공식 TRELLIS.2 경로 | 현재 CUDA |
| --- | --- | --- |
| DiT 구성 | SS / LR shape / HR shape / texture의 4개 flow | 동일한 4개 flow |
| 주요 크기 | hidden 1536, 30 blocks, 12 heads, MLP 8192 | 동일 |
| Conditioning | DINO global 5 tokens + 좌표별 projection; shape/texture projection 2048 channels | 같은 구성 |
| Shape/texture decoder | channels `[1024,512,256,128,64]`, blocks `[4,16,8,4,0]`, latent 32 | 같은 구성 |
| Shape / PBR 출력 | shape 7 channels; base color 3 + metallic + roughness + alpha | 같은 채널 구성 |
| Flow schedules | stage당 12 steps, CFG/시간 재매핑 | pinned pipeline 설정 사용 |
| 공간 표현 | sparse structured latent / sparse decoder | 역시 sparse; MLX만의 기법이 아님 |
| 축소 모델 / quantization | 해당 없음 | 적용하지 않음 |

근거: [공식 flow](https://github.com/TencentARC/Pixal3D/blob/f7cf38429b0bd264f1995f0f8743a88b1c728b94/pixal3d/models/structured_latent_flow.py#L15),
[공식 모델 설정](https://huggingface.co/TencentARC/Pixal3D/blob/0b31f9160aa400719af409098bff7936a932f726/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.json),
CUDA `pixalcuda/flow.py:116`, `pixalcuda/decoder.py:105`, `pixalcuda/weights.py:99`.
키·shape를 엄격히 검증해 원본 가중치를 로드한다. 이것은 가중치 호환성의
근거이지 각 backend의 전체 수치 동등성 증명은 아니다.

512 경로도 HR/texture 단계를 삭제하지 않는다. 1024-named checkpoint를
32³ refinement grid에 실행하고 최종 512 geometry로 decode한다.
반면 공식 `Pixal3DImageTo3DPipeline.run`의 선택지는 1024/1536 cascade다.
따라서 **우리 512와 공식 기본 1536을 직접 비교하면 계산량과 품질 조건부터 다르다.**

## 2. VRAM을 낮추는 실제 실행 구조

| 항목 | 공식 standard | 공식 `--low_vram` | 현재 CUDA |
| --- | --- | --- | --- |
| 생성 모델 | pipeline 모델 GPU 상주 | 현재 flow/decoder만 GPU로 이동 후 CPU 복귀 | 현재 모델만 meta에서 materialize, 실행 후 폐기 |
| DINO | stage별 extractor 4개가 각각 backbone 소유, 모두 GPU로 이동 | extractor를 차례로 GPU로 이동 | DINO 1개로 512/1024 특징 추출 후 폐기; 특징은 CPU 보관 |
| 3D 투영 특징 | 전체 R³ grid를 만든 뒤 active 좌표 선택 | 동일한 dense projection | active 좌표에만 직접 projection |
| NAF | 전체 upsampled D-channel 특징 맵 생성 후 projection | 같은 전체 특징 맵 생성 | guide Q/K 맵은 유지하되, 필요한 bilinear 4-corner만 chunk별 평가 |
| Flow 임시 버퍼 | backend 연산 단위 | 대체로 동일 | attention query / FFN / sparse gather를 chunk로 제한 |
| 중간 결과 | GPU tensor를 단계 간 유지/정리 | 모델은 offload하지만 GPU 중간 tensor는 남음 | stage checkpoint를 CPU로 내리고 디스크 저장 |
| 메시/PBR 후처리 | CuMesh hole fill + `o_voxel` export | 동일 후처리 경로 | CPU NumPy/SciPy/trimesh/xatlas 경로; Modal은 별도 CPU worker |

공식 [초기화](https://github.com/TencentARC/Pixal3D/blob/f7cf38429b0bd264f1995f0f8743a88b1c728b94/inference.py#L74)와
[단계별 offload/conditioning](https://github.com/TencentARC/Pixal3D/blob/f7cf38429b0bd264f1995f0f8743a88b1c728b94/pixal3d/pipelines/pixal3d_image_to_3d.py#L238)를
구분해야 한다. 공식 standard의 큰 상주 집합은 low-VRAM 모드와 다르며,
공식 low-VRAM도 이미 매우 유사한 모델 순차 실행을 한다.

MLX의 `generate_pixal3d.py`는 stage별 load/cleanup, 큰 flow의 `retain=False`,
bounded feature/conditioning cache와 sparse NAF query를 사용한다.
CUDA는 이 방향을 계승하고 CPU checkpoint/독립 CPU export를 더 명시적으로 나눴다.
다만 MLX는 unified memory를 사용하므로 MLX active/cache 값과 CUDA 전용 VRAM은
같은 지표가 아니다. CPU 이동·해제가 GPU 수요를 줄여도 host RAM/IO 비용은 남는다.

### 전체 특징 대신 필요한 좌표만 계산하는 차이

기존 1024 shoe의 active HR 좌표는 10,305개였다.

| tensor 예시, FP32 | 이론상 tensor payload |
| --- | --- |
| 공식 dense projection `64³ × 2048` | 2.000 GiB |
| 현재 active projection `10,305 × 2048` | 80.508 MiB |
| 공식 texture NAF 전체 특징 `1024² × 1024` | 4.000 GiB |
| 공식 shape NAF 전체 특징 `512² × 1024` | 1.000 GiB |

이 값은 **해당 tensor 한 개의 크기**이며 실제 peak 차이나 동시에 살아 있는
모든 buffer의 합을 측정한 값이 아니다. 공식 projection은 low/high 결과를 concat하므로
중간 tensor도 있고, 실제 dtype·수명·backend workspace가 peak를 결정한다.
CUDA도 NAF guide image encoder와 target 크기의 256-channel Q 맵은 생성한다.
“NAF 전체를 10,305 query만큼만 계산한다”거나 위 숫자를 단순 합산해 절감량으로
표현하면 안 된다. 동일 target의 sparse 평가가 전체 맵 평가와 같은 의미를
목표로 하지만 pretrained NAF 원본과의 직접 수치 대조는 아직 없다.

## 3. 실제 1024 T4 측정으로 보는 메모리 예산

기존 측정은 1024 geometry / 1024 texture / 100k-face 목표 / cache 1024 MiB,
HR token 10,305 / 최종 sparse voxel 2,771,719 조건이다. 이번에 재실행한 값이 아니다.

| 단계 | PyTorch peak allocated |
| --- | --- |
| DINO features | 1.355 GiB |
| Sparse structure | 6.093 GiB |
| HR conditioning | 2.706 GiB |
| Shape HR flow | **7.010 GiB** |
| Shape decode | 5.878 GiB |
| Texture flow | 7.004 GiB |
| Texture decode | 5.878 GiB |

HR flow 한 모델의 실제 FP32 parameter는 **5.166 GiB**다. 해당 stage의
1 GiB cache budget에서는 30개 중 16개 block의 projection을 캐시하여
**966.094 MiB**를 사용했다. 모델과 cache에 activation/workspace를 더하면
관측된 약 7 GiB peak가 설명된다. 최대 reserved는 7.139 GiB였다.
이 worker에서 CUDA가 보고한 전체 용량은 14.571 GiB였다.

동일 runtime dtype으로 4개 flow를 모두 상주시킨다고 가정하면 **parameter만
20.487 GiB**다. 중복 checkpoint load를 제거하고 DINO/NAF/decoder까지 합친
고유 parameter의 가상 합은 **25.289 GiB**다. MoGe/배경 제거/activation은 제외했다.
이는 공식 런타임의 메모리 수치가 아니라 **우리 모델을 현재 dtype으로 전부
상주시킬 수 없는 이유를 설명하는 가상 계산**이다.

반대로 공식은 flow block을 BF16, decoder torso를 FP16으로 변환한다.
CUDA는 T4에서 확인한 FP16 overflow를 피해 DINO/flow/NAF/sparse decoder를
FP32로 실행한다. 따라서 **모델 한 개의 parameter당 메모리는 오히려 CUDA 쪽이 크다.**
예를 들어 대응 HR flow의 block만 BF16, 나머지 FP32라고 계산하면 약 2.615 GiB다.
checkpoint 이름의 `_bf16`은 현재 CUDA 실행 dtype을 뜻하지 않는다.

host RAM도 공짜는 아니다. 같은 측정의 process RSS 최고 샘플은 texture flow에서
10.913 GiB였다. PyTorch allocated/reserved는 CUDA context·일부 외부 라이브러리
할당을 모두 포함하는 device-wide 사용량이 아니고, 공식 peak는 아직 미측정이다.
따라서 “공식은 몇 GB인데 우리는 정확히 몇 배 절약했다”는 수치는 제시하지 않는다.

## 4. 원본과 달라진 품질 조건: 우선 확인할 사항

### P0: 단일 이미지 texture conditioning은 원본과 다르다

- 공식 single-image 설정은 shape HR NAF target **512**, texture NAF target **1024**다.
- MLX `generate_pixal3d.py:775`는 MV일 때만 texture NAF 1024를 따로 만들고,
  단일 이미지에서는 shape HR conditioning을 그대로 texture에 재사용한다.
- CUDA `pixalcuda/pipeline.py:193`도 같은 분기로, 단일 이미지 texture에 NAF 512를 쓴다.
- `Pipeline.condition`의 단일 이미지 분기는 `naf_target` 인자를
  `build_conditioning`에 전달하지 않는다(`pixalcuda/pipeline.py:263`).
  향후 분기를 추가할 때 인자 전달까지 수정해야 실제 target 변경이 반영된다.

즉 이 부분은 **MLX에서 유래한 실제 의미 차이**다. 공식 texture 모델이 받는
conditioning과 다르므로 색/재질 디테일에 영향을 줄 수 있다. 현재 메시 균열의
원인이라고 단정할 수는 없으며, geometry 문제와 분리해 대조해야 한다.
복원 시 CUDA의 sparse NAF 평가를 유지할 수 있으므로 4 GiB 전체 특징 맵을
일부러 생성할 필요는 없다. 단, target 증가에 따른 guide 메모리/계산량은 재측정해야 한다.

### P0: export 경로가 동등하지 않다

공식 CLI는 `decimation_target=1,000,000`, `texture_size=4096`, `remesh=True`로
`o_voxel.postprocess.to_glb`를 호출한다. 그 전에 CuMesh로 hole filling도 수행한다.
우리 기본값은 100,000 faces / texture 1024이며 CPU cleanup/decimation/UV/baking을
사용한다. 현재 exporter는 texture 4096을 지원하지 않고 동일한 remesher도 아니다.
메시·표면 품질 비교에서는 **raw decoded geometry, 후처리 후 geometry, PBR bake**를
따로 봐야 한다. CPU로 옮겼다는 것과 같은 알고리즘을 포팅했다는 것은 다르다.
근거: [공식 export 설정](https://github.com/TencentARC/Pixal3D/blob/f7cf38429b0bd264f1995f0f8743a88b1c728b94/inference.py#L259),
[공식 hole fill](https://github.com/TencentARC/Pixal3D/blob/f7cf38429b0bd264f1995f0f8743a88b1c728b94/pixal3d/representations/mesh/base.py#L35).

전처리도 공식 pipeline의 BiRefNet과 현재 CPU `u2net`이 다르다.
공정한 대조에는 한 번 만든 동일 RGBA/crop과 고정 카메라를 양쪽에 입력해야 한다.
동일 seed만으로 backend 간 noise가 같아지지 않으므로 같은 initial noise/좌표/
conditioning tensor를 주입해 비교해야 수치 차이의 시작점을 찾을 수 있다.

## 5. VRAM 여유가 곧 속도 여유는 아니다

원본은 기본 sparse conv가 FlexGEMM이고 attention backend 선택을 제공한다.
CUDA는 확장 모듈 없이 PyTorch searchsorted neighbor map + offset별 gather/GEMM/
scatter와 SDPA를 사용한다. **이것은 MLX 친화적인 연산 분해의 영향이 남은 부분**이다.
native kernel을 쓰면 launch/메모리 이동/중간 buffer가 줄 수 있지만, 더 많은
VRAM을 쓴다거나 반드시 더 빠르다고 단정할 수는 없다. SM75/FP32 지원과 실제
입력 크기에서의 수치·성능을 먼저 확인해야 한다.

현재 T4 SDPA는 측정상 이미 FP32 SM75 CUTLASS memory-efficient kernel이다.
math attention으로 바꿔 VRAM을 더 쓰게 하면 이전 probe에서 더 느렸다.
query chunking은 모든 key를 유지하므로 window/local attention으로 모델을 바꾼 것은 아니다.
FlashAttention-2 본체의 일반 지원 GPU와 별도 Turing 구현은 구분해야 하며,
원본 BF16 경로를 T4의 native BF16 Tensor Core 경로로 단순 복제할 수는 없다.
[FlashAttention 공식 지원 범위](https://github.com/Dao-AILab/flash-attention#nvidia-cuda-support),
[NVIDIA의 Ampere Tensor Core 추가 기능](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#improved-tensor-core-operations).

실제 1024 보고서에서 GPU stage 합은 668.173 s, 모델 callback 합은 615.487 s,
모델 load 합은 27.929 s다. **로드를 전부 제거해도 이 관측치 기준 직접 제거 가능한
load 시간은 GPU stage 합의 약 4.2%**에 불과하다. 이는 이상화한 load 항목의 상한이며
cache 변경의 전체 성능 상한이나 새 벤치마크 결과는 아니다. 무조건 모든 모델을
GPU에 붙잡아 두는 것보다 긴 flow callback 내부를 최적화할 근거가 더 강하다.

## 6. 권장 진행 순서

1. **공식 의미 복원/대조**: 단일 이미지 texture NAF 1024를 별도 생성하고 인자 전달을
   검증한다. 기존 512 공유 경로와 구분해 품질·메모리·시간을 기록한다.
2. **메시 차이 분리**: 동일 raw decoder output으로 현재 CPU 후처리와 공식
   hole fill/remesh/bake를 비교한다. export face/texture 설정 차이도 명시한다.
3. **동일 입력·동일 1024 비교**: 공식 standard/low-VRAM과 CUDA를 별도 프로세스에서
   실행하고, stage별 allocated/reserved/device peak/host RSS/시간을 같은 기준으로 측정한다.
   T4에서 원본 BF16 경로가 불가능하면 호환성 변경을 명시하거나 native BF16 GPU의
   reference 출력을 먼저 확보한다. “원본 그대로 T4 비교”라고 부르면 안 된다.
4. **CUDA-native backend**: 검증된 conditioning/decoder를 기준으로 FlexGEMM 등 native
   sparse 연산 또는 fused 대체 구현을 선택 가능하게 검토한다. gather 경로는 reference로 남긴다.
5. **선택적 precision/상주 정책**: overflow 없는 연산만 FP16 + FP32 accumulation을
   검증한다. decoder 재사용·전체 projection cache·제한된 hot model cache는 남은
   VRAM을 사용할 후보지만, 모델 전체 상주는 현재 FP32로 T4에 들어가지 않는다.

이 순서는 제안이며 이번 감사에서 구현/배포한 변경이 아니다.
공식 대비 품질 차이를 먼저 고정해야 “원본보다 빠르다”와 “다른 계산을 덜 했다”를
구별할 수 있다. 기존 MLX 저장소는 변경하지 않았다.
