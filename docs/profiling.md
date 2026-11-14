# Native 수치 차이 분리 — L4 prefix 진단

이 문서는 초기 prefix 결과다. 이후 [전체 fixed-kernel profiling](full-profiling.md)에서
HR/texture와 raw mesh/재질 voxel의 149개 체크포인트까지 exact를 확인하고,
모델 초기화·xatlas·inpaint 비용을 분리했다. Prefix에서 미검증이던 항목을 구분해 읽는다.

2026-11-14 KST에 실제 Modal L4 한 장에서 pretrained pipeline을 한 번 로드하고,
같은 준비된 신발/카메라/seed 42로 7개 prefix를 순차 실행했다. **생산 inference나
GLB benchmark가 아니라 최초 divergence를 찾는 진단**이다. 별도 dev environment는
없고, `main`의 일시적 진단 App만 사용한 뒤 종료했다. Production App은 변경하지 않았다.

## 결과

SS의 conditioning/noise/12-step sampling, SS 좌표, LR conditioning/noise/12-step
sampling, shape decoder의 512→512 sparse convolution 입출력, raw upsample 좌표와
1024 경로의 quantized HR 좌표까지 **73개 체크포인트**를 비교했다. CPU/CUDA RNG도
캡처했다. 부동소수 tolerance는 `atol=1e-6`, `rtol=1e-5`, 좌표는 exact다.

| 프로필 | SPLITK | 기준과의 결과 |
| --- | --- | --- |
| Reference 첫 실행 | 모두 1 | private 기준 fixture |
| Reference 반복 | 모두 1 | 73개 체크포인트 exact |
| DINO 공유/특징 cache만 | 모두 1 | 73개 체크포인트 exact |
| Active projection만 | 모두 1 | 73개 체크포인트 exact |
| Conditioning cache만 | 모두 1 | 73개 체크포인트 exact |
| 최적화 전체 | 모두 1 | 73개 체크포인트 exact |
| Reference, 관측된 key 하나만 변경 | 해당 key만 2 | decoder convolution부터 차이 |

마지막 대조군은 hook을 모두 끄고 FlexGEMM의
`(2^13, 512, 512, 27)` 선택만 1→2로 바꿨다. 최초 차이는
**`shape_slat_decoder.blocks.0.4.conv2.output`**이다. 해당 convolution 입력은 exact였고
출력 좌표도 같았지만, FP16 feature의 최대 절대 오차는 **0.0078125**, relative L2는
**0.0003570**이었다. 이후 raw upsample 좌표 수가 **776,186→776,190**, quantized HR
좌표 수가 **10,578→10,579**로 달라졌다. 모든 캡처 tensor는 finite였다.

이는 **native 알고리즘 선택 차이만으로 좌표 구조까지 달라질 수 있음**을 재현한다.
단순한 seed 누락이나 이 구간의 hook 오류로만 볼 근거는 없었다. Reference 반복과
optimized의 inner Triton autotune cache도 같았다. 생산 환경에 SPLITK=1을 강제하거나
정확한 원본 연산 대신 다른 구현을 넣은 것은 아니다.

공식 decoder는 subdivision prediction을 `>0`으로 이산화한다. 작은 feature 오차가
이후 다른 subdivision/좌표 선택으로 이어질 수 있는 구조다.
[Pinned decoder](https://github.com/TencentARC/Pixal3D/blob/f7cf38429b0bd264f1995f0f8743a88b1c728b94/pixal3d/models/sc_vaes/sparse_unet_vae.py),
[pinned SPLITK 구현](https://github.com/JeffreyXiang/FlexGEMM/blob/6dd94a859c26ee8246888502eada3dd8ad85532e/flex_gemm/kernels/triton/spconv/sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk.py).
후자는 SPLITK=1의 단일 누적과 >1의 FP32 partial sum/reduction 경로가 다르다.

**한계:** 이전 full-asset pair의 유일한 원인을 확정한 것은 아니다. 그때는 여러 key가
mixed SPLITK였고 inner Triton config가 보존되지 않았으므로 이 진단은 literal replay가
아니다. 과거 HR 좌표 수 10,579/10,577과 이번 10,578/10,579도 다르다. HR/texture flow,
1024 NAF texture 조건, 최종 decoder, remesh/UV/PBR는 이번 실행에서 비교하지 않았다.
따라서 앞선 full tensor parity 실패를 통과로 바꾸거나 GLB 동등성을 주장하지 않는다.

## 커널 프로파일

Reference 반복과 optimized의 **LR sampling step index 1 한 번씩**을 PyTorch profiler로
캡처했다. `aten` operator의 device attribution과 실제 kernel event는 같은 실행을
중복 표현하므로 합산하지 않았다.

| 항목 | Reference | Optimized |
| --- | ---: | ---: |
| `aten::addmm` 호출 | 490 | 401 |
| `aten::addmm` self device | 257.14 ms | 241.95 ms |
| Flash attention self device | 59.88 ms | 58.83 ms |
| `aten::copy_` 호출 | 1,162 | 1,251 |
| `aten::clone` 호출 | 0 | 89 |
| `aten::equal` 호출 | 0 | 7 |
| 실제 CUDA kernel event 수 | 4,609 | 4,534 |
| CUDA kernel duration 합 (wall time 아님) | 465.85 ms | 450.62 ms |

Static cache가 linear 89회를 없앴지만 clone 89회와 입력 비교를 추가했고, 남은
GEMM/attention 비용이 크다. 이것이 전체 160초 신경망 구간의 가속을 곧바로 보장하지
않는 이유다. 각 1회이며 profiler/동기화/CPU snapshot/디스크 비교 오버헤드가 있다.
case 시간이나 kernel duration 합을 생산 latency/일반 가속률로 사용하지 않는다.
`_local_scalar_dense`/`equal`의 CPU duration에는 GPU 대기도 있어 CPU-bound 증거가 아니다.

로드 포함 계측 stage 합은 **360.41초**, timeout은 1,200초였다. 별도의 GPU 재시도,
전체 GLB 생성 또는 하드웨어 변경은 없었다. 종료 뒤 active container 0을 확인했다.
실행·trace 비교·resource 범위는
[sanitized evidence](evidence/modal-l4-prefix-2026-11-14.profiling.json)에 기록한다.

## 실행 도구

```bash
python scripts/profile_modal_native.py \
  --source-job COMPLETED_NATIVE_JOB_ID --output-dir outputs/prefix-probe
```

기본은 dry-run이며 Modal import/접속, 파일 저장이나 GPU 요청이 없다. `--execute`를
붙일 때만 캐시된 이미지/가중치로 실행한다. 완료된 unmatted SV job의 prepared image,
입력 SHA와 실제 camera를 재사용하며 source/model lock을 확인한다. 라이선스 opt-in이나
RMBG 다운로드는 없다. 입력/camera/캐시 파일은 공개하지 않는다.

- 운영 GPU가 실행/대기 중이면 추가 GPU 요청 전에 거절한다. 이 확인은 전역 분산 lock이
  아니므로 진단 중 별도의 생성 요청을 동시에 제출하지 않는다.
- 일시적 App은 min=0/max=1, L4, CPU 2개, host RAM 16–64 GiB, retries=0, timeout 20분이다.
  Production의 기본 옵션과 배포는 바꾸지 않는다.
- GPU 할당 대기를 포함한 client deadline은 2시간이다. timeout/중단 시 저장된 call ID를
  취소하고 container를 종료한다. 기존 call의 종료를 확인하기 전에 재제출하지 않는다.
- `reference_first`만 private CPU tensor fixture를 저장한다. 각 prefix의 payload 상한은
  512 MiB이며 초과 시 실패한다. 비교 대상 case는 tensor를 계속 축적하지 않는다.
- 나머지 trace는 digest/shape/dtype/finite와 strict 비교다. Event 누락/순서·tree·dtype·shape
  차이를 숨기거나 정렬/느슨한 tolerance로 통과시키지 않는다.
- DINO 공유는 이미 로드된 같은 checkpoint 모델의 alias로 격리하고 case마다 복원한다.
  모델 초기화 시간/CPU 메모리 절감 비교가 아니다. 모든 hook/SPLITK override도 복원한다.
- `trace.json`, 두 `kernel-trace.json`, 각 case의 inner autotune cache와 선택된 SPLITK map을
  private jobs volume 아래 보존한다. `.pt`/trace는 입력 파생 정보일 수 있으므로 외부
  trace 서비스에 올리거나 Git에 추가하지 않는다.

## 다음 순서

1. 전체 경로 비교에는 **top-level SPLITK map + inner Triton cache + 장치/software pins**를
   묶어 고정한다. 이번 prefix만으로 전역 결정성이 보장되는 것은 아니다.
2. 같은 HR/texture 입력을 replay해 나머지 hook 동등성을 검증한다. 성능 실험과 깊은
   tensor 캡처는 분리하고 교차 반복한다.
3. 앞선 full-asset 기준 큰 비용인 UV/export와 모델 로드를 먼저 세분화한다. 신경망은
   GEMM/FFN, clone/동기화 비용을 구분한 뒤 cache 예산/compile을 시험한다.

가속 후보나 cache 예산을 production에 새로 적용한 결과는 아니다. 별도 MLX 저장소는
수정하지 않았다. [전체 측정과 남은 제약](performance.md)을 함께 참조한다.
