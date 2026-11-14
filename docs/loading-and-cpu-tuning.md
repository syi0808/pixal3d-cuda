# Coverage-guarded 로딩·CPU 최적화

별도 MLX 저장소나 고정된 공식 checkout을 수정하지 않는다. 네트워크 연산과
sampler/geometry/NAF/texture 품질 옵션은 바꾸지 않는다.

## 초기화 생략

`skip_weight_init=true`는 checkpoint로 덮어쓸 random initializer만 생략한다.
공식 `models.from_pretrained`와 원래 `load_state_dict`를 그대로 호출하며 dtype이나
`assign`/parameter layout을 바꾸지 않는다. DINO/NAF/MoGe loader는 이 hook의 대상이 아니다.

1. Local safetensors header의 key/shape를 먼저 읽는다.
2. Pinned allowlist의 model constructor에서만 `nn.init` random fill을 일시 생략한다.
   Constant/zero/one 초기화와 buffer 계산은 유지한다.
3. 생성된 **모든 named parameter와 buffer**가 checkpoint에 같은 shape로 존재하는지
   확인한다. 적용 후에도 missing key가 없어야 한다. 불완전 coverage나 알 수 없는
   model type이면 미초기화 모델을 쓰지 않고 실패한다. 자동 full-init fallback은 없다.
4. 성공/오류에서 모든 임시 hook을 복원한다. Coverage/count는 `report.initialization`에 남긴다.

Constructor에서 parameter 값을 읽어 별도 상태를 만드는 임의의 모델을 지원하는
일반-purpose skip-init이 아니다. 검토한 pinned 모델과 그 checkpoint가 대상이다.
Sampler의 명시적 seed 이전 RNG 소비량은 달라질 수 있으나 sampler noise/결과 동등성은
고정 kernel의 전체 checkpoint 비교로 검증한다. 생성자의 random draw를 다시 실행해
RNG 상태만 맞추는 식으로 비용을 되살리지 않는다.

Python `Pipeline(skip_weight_init=False)`, CLI `--full-init`로 원래 초기화를 선택한다.
CLI/HTTP `reference=true`는 no-init과 병렬 inpaint도 끈다. Source/옵션 fingerprint가
바뀌므로 이전 job을 새 결과인 것처럼 재사용하지 않는다.

## 동일 연산의 병렬 inpaint

`inpaint_workers=1`은 원본 exporter다. 더 큰 값에서는 고정된 `to_glb` source의
정확히 4개 연속 OpenCV inpaint assignment만 AST로 확인해 병렬 batch로 치환한다.
Checkout 파일은 수정하지 않으며 나머지 native remesh/UV/raster/attribute/material
연산은 유지한다. Block/call 수가 달라지면 다른 exporter를 추측해 실행하지 않고 실패한다.

Base color radius 3, metallic/roughness/alpha radius 1, 원본 TELEA, 동일 mask/dtype와
channel order를 유지한다. OpenCV global thread 설정은 바꾸지 않는다. 독립된 4개
호출만 최대 `min(workers,4)` thread로 실행하며 출력이 모두 준비된 뒤 원래 exporter를
계속한다. 알고리즘/해상도 축소, constant-channel 근사 또는 mask 변경은 없다.
실제 4K 입력에서 sequential 대비 모든 채널의 byte/SHA 일치를 확인한다.

Python `Pipeline(inpaint_workers=...)`, CLI `--inpaint-workers` 및 operator TOML로
조정한다. HTTP 사용자는 임의로 CPU나 thread 예산을 올릴 수 없다.

## CPU 2/4/8 실험

```bash
python scripts/optimize_modal_native.py \
  --source-job COMPLETED_NATIVE_JOB_ID --output-dir outputs/native-tuning
```

기본은 dry-run이다. `--execute`에서만 main의 일시적 진단 App으로 순차 실행한다.
CPU=2/4/8마다 실제 L4 한 장, host RAM request/limit=16/64 GiB, timeout 35분,
retry=0이다. 이전 GPU container가 scale-to-zero된 뒤 다음 tier를 요청한다.
운영 GPU가 바쁘면 추가 할당을 거절한다. 개발 environment나 GPU fallback은 없다.

- CPU2에서 같은 DINO/reuse 옵션을 유지한 full-init/no-init을 새 모델 각각으로 읽고,
  고정 SPLITK/inner tuning cache/TF32/cuDNN 설정에서 전체 149 checkpoint를 비교한다.
- Native export에서 **GPU clustering 이후 xatlas에 전달된 chart 목록**과 4개 4K
  inpaint 입력을 비공개 fixture로 캡처한다. 다른 CPU tier에서도 동일 파일 hash를 확인한다.
- 각 tier에서 chart 생성/packing 2회, inpaint worker 1/2/4를 정방향·역방향으로 2회씩
  측정한다. UV finite/range 및 원래 oriented face multiset 보존을 확인한다. 이는 UV 좌표
  bit-equality가 아니라 seam split을 허용하는 topology 검사다. Neural/픽셀 비교에는
  정렬이나 tolerance 완화를 사용하지 않는다.
- 픽셀은 tier 내부와 tier 사이에서도 exact/SHA가 같아야 한다.

CPU quota 이외에 host/실행 순서를 완전히 교차 통제한 fleet benchmark는 아니다.
측정은 pilot이며 자동 resource 변경은 하지 않는다. `gpu_cpu_cores`는 GPU worker에만
적용하며 API/prepare/prefetch의 `cpu_cores`와 구분한다. 둘 다 min=0을 유지한다.

```bash
python scripts/summarize_native_tuning.py outputs/native-tuning
```

비용은 [Modal 공개 요금](https://modal.com/pricing)의 L4/요청 CPU와 16/32/64 GiB
메모리를 사용한 **정규화 추정치**다. 실제 청구서가 아니다. 실제 과금은 request/usage 중
큰 값, startup/idle/지역 등에 영향을 받는다. 공통 시간에는 진단 snapshot/capture
오버헤드가 있으므로 최종 선정 설정은 실제 HTTP 생성으로 다시 확인한다.

`.env`, 이미지/가중치, 약 3 GiB reference tensors, chart/inpaint fixtures, Kineto trace와
GLB는 Git에 넣지 않는다. 실제 실행값·기본값 선정은 검증 기록에서 별도로 구분한다.

## 2026-11-14 L4 실측·적용 결과

### Coverage와 초기화 생략

Pinned native 모델 7개의 **3,450개 parameter tensor / 6,521,594,742개 parameter**가
checkpoint에 모두 포함됐다. Missing/unexpected key는 모두 0이었다. 적용된 코드에서는
buffer도 전부 있어야 하며 실제 main 실행의 buffer 1개도 완전히 덮어썼다.
초기 실험에 있던 미사용 derived-buffer 예외는 rollout 전에 제거했다.

같은 DINO/reuse 옵션, 준비된 신발, camera/seed 42, 원본 schedule/precision/1024 경로로
full-init과 no-init을 비교했다. 같은 calibration manifest를 유지하고 새로운 tuning을
거절한 상태에서 **149개 checkpoint 전부 exact**였다. HR/texture의 conditioning/noise/
step, latents, 최종 raw vertices/faces/재질 voxel을 포함하며 모든 tensor가 finite였다.

| 구간 | Full init | Covered no-init |
| --- | ---: | ---: |
| 모델 로드 | 130.73 s | **12.87 s** |
| Neural 진단 | 154.06 s | 125.09 s |

두 번째 load는 파일 cache가 warm하므로 90.2% 차이 전부를 initializer 효과로
귀속하지 않는다. Neural 시간 차이도 첫 kernel calibration/warmup 때문일 수 있으며
no-init의 추론 가속으로 세지 않는다. 별도 새 main worker의 실제 load는 **20.62초**였다.
Checkpoint 값·dtype/layout·네트워크 연산·샘플러는 바꾸지 않았다.

### 같은 xatlas/inpaint 입력의 CPU 비교

세 tier에서 같은 2,899개 post-clustering chart와 4K inpaint 입력을 재생했다.
Fixture 파일 hash가 모두 같고 각 CPU tier의 chart 생성/packing을 2회 측정했다.
모든 chart의 oriented topology/UV finite/range 검사가 통과했다. UV 좌표나 최종 GLB
bit-equality를 주장하는 결과는 아니다.

| Modal GPU worker CPU 요청 | Chart 생성+packing 중앙값 | 직렬 inpaint | 병렬 2 | 병렬 4 |
| --- | ---: | ---: | ---: | ---: |
| 2 cores | 197.90 s | 36.85 s | **18.56 s** | 23.80 s |
| 4 cores | 99.16 s | 35.81 s | 19.36 s | **19.24 s** |
| 8 cores | **50.93 s** | 37.95 s | 20.15 s | **18.79 s** |

병렬 4가 항상 낫지는 않았다. 각 tier에서 worker 1/2/4를 정·역순으로 2회 실행한
총 18개 결과의 **4개 channel SHA가 tier 사이에서도 모두 같았다**. 실제 L4 이미지에서
import된 OpenCV는 **5.0.0**이었다. Dev/CI의 4.12.0에서도 작은 mask/shape 계약을
검사하지만, native 이미지의 모든 Python 전이 의존성이 hermetic하다는 주장은 아니다.

### 비용 해석과 선정

가중치 로드/신경망/나머지 export에 공통 166.15초를 둔 초기 진단 모델과, 요청 RAM을
32 GiB로 정규화한 공개 list-price 계산은 다음과 같다. **청구서나 실제 asset 비용이 아니다.**

| CPU / 적정 inpaint worker | 모델 asset 시간 | 정규화 추정 USD |
| --- | ---: | ---: |
| 2 / 2 | 382.61 s | 0.1221 |
| 4 / 4 | 284.55 s | 0.0983 |
| 8 / 4 | 235.87 s | 0.0938 |

실제 HTTP 검증은 cold GPU body가 329.19초였고 초기 모델보다 길었다. 이를 반영한
공통 시간 236.64초 sensitivity에서는 RAM 16/24 GiB 가정에서 4코어가 약간 저렴하고,
32/64 GiB에서는 8코어가 약간 저렴했다. 32 GiB 가정의 4/8코어 추정치는
0.1226/0.1219 USD로 **거의 손익분기점**이다. 실제 사용 메모리, cold-start, host 차이와
UV 입력에 따라 순위가 바뀔 수 있어 보편적인 최소비용 설정으로 확정하지 않는다.
또한 GPU body 이전 startup, region, idle 등은 이 모델에서 빠져 있다.

**기본값은 낮은 latency를 우선해 `gpu_cpu_cores=8`, `inpaint_workers=4`,
`skip_weight_init=true`로 선정했다.** API/prepare/prefetch CPU는 2로 유지했다.
비용만 우선하거나 신경망 구간이 길면 `gpu_cpu_cores=4`가 합리적인 대안이다.
선택/요금/가정은 `scripts/summarize_native_tuning.py`로 재계산한다. 정규화 모델에 실제
API의 cold GPU non-export 시간을 반영한 sensitivity도 함께 출력한다.

### Main rollout

캐시된 native layer로 **40.999초**에 재배포했다. 실제 인증 HTTP POST로 새 asset을
제출했고, 상태 대기는 operator Dict로 관측해 HTTP API의 반복 cold-start를 피했다.
Health는 인증 200/비인증 401이며 no-init/8코어/worker4와 RMBG off를 확인했다.

| 실제 API stage | Seconds |
| --- | ---: |
| Load | 20.62 |
| 전처리·MoGe camera | 9.33 |
| Denoise/decode | 159.28 |
| Native remesh/UV/4K PBR/GLB | 120.74 |
| Stage 합 | **309.97** |

Stage 밖 import/준비, queue/통신/다운로드는 합계에서 제외한다. GPU running→complete는
329.19초, allocation queue는 약 69.77초였다. 결과는 **38,186,064 bytes**, embedded
4096² base-color/metallic-roughness를 독립적으로 재검증하고 실제 GLB를 렌더했다.
새 worker의 엄격한 coverage와 no-init/worker4도 report에서 확인했다. Camera는
기존 입력의 실제 camera와 같았다. OOM, 재시도, 해상도/장치 fallback은 없었다.
과거 530초 asset과의 단순 차이를 동일-output 가속률로 발표하지 않는다.

이 실험은 단일 prepared SV shoe의 pilot이다. Chart CPU 비교는 host/순서를 완전히
교차 통제하지 않았고, 최종 GLB byte parity/MV/1536/T4 품질은 별도다.
CPU 계약 테스트는 **239 passed**이며 GPU 수치 검증을 대신하지 않는다.
진단 L4 3회와 최종 API L4 1회를 순차 실행하고 scale-to-zero를 확인했다.
Main 환경만 사용하며 max GPU=1/min=0을 유지했고 MLX는 수정하지 않았다.

[Sanitized 측정·배포 증거](evidence/modal-l4-loading-cpu-2026-11-14.optimizations.json).
