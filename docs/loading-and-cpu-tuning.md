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
