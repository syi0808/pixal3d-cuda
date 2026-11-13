# 공식 runtime 최적화 현황

v0.2부터 기본/대체 backend 구분 없이 공식 Pixal3D 경로 하나만 제공한다.
이전 MLX-derived 연산 구현과 전용 벤치마크는 삭제했다. 과거 측정은
[이전 revision](https://github.com/syi0808/pixal3d-cuda/blob/a6ce8a3c4a68e377569b7965035a8f00d74ff9b0/docs/performance.md)
및 [보존된 증거](validation.md)에 있으며 새 runtime 수치가 아니다.

## Modal L4 실제 측정 — 2026-11-14 KST

CUDA revision `02f84c942844558d03daa0205f2295d27f156aa1`의 같은 공식 pipeline에서
optimized → reference 순서로 **각 1회** 실행했다. Reference는 hook만 끈 것으로,
원본 CLI를 별도로 실행한 대조군이 아니다. MLX 저장소는 수정하지 않았다.

- 입력: 준비된 512×512 RGBA 신발 한 장, seed 42, 같은 자동 MoGe camera.
- 품질: 실제 geometry 1024, texture 4096, 1M face target, 원본 12-step schedules.
- 장치: NVIDIA L4/SM89, Torch 2.6.0+cu124, 원본 BF16 flow/FP16 decoder,
  SDPA/FlexGEMM, low-VRAM. 보고된 CUDA 용량은 22.06 GiB다.
- 미리 받은 가중치를 사용하지만 매 asset마다 새 프로세스에서 모델을 읽는다.
  hot-model 추론이 아니며 import/JIT/autotune/OS cache 변동을 통제한 반복 실험도 아니다.
- 자동 matting/crop은 끄고 RGBA alpha를 검정 배경에 합성했다. 두 실행 모두
  RMBG를 로드하지 않았고 OOM, 해상도/precision/GPU fallback은 없었다.

| 구간 | Reference (s) | Optimized (s) |
| --- | ---: | ---: |
| Pipeline 로드 | 134.97 | 117.23 |
| 전처리·카메라 | 10.70 | 8.89 |
| Denoise + decode | 160.01 | 159.90 |
| Remesh + UV + PBR + GLB | 300.61 | 244.41 |
| 위 단계 합계 | **606.28** | **530.44** |
| 진단용 reference 저장 (합계에서 제외) | 0.46 | 0.44 |

**합계 12.51% 감소는 관측값이지 동등 품질 가속의 증명이 아니다.** 신경망 구간은
0.10초(0.063%) 차이로 사실상 같다. 약 75.84초 차이 중 56.20초가 출력 형상이 다른
export에서 발생했다. 이 실험은 cache 예산만 바꾼 비교나 개별 hook의 ablation이 아니다.

Stage 시간은 CUDA synchronize로 경계를 측정한다. 모델 파일 로드는 포함하지만
queue/container startup, stage 밖 import·하드웨어 검사, HTTP/다운로드는 제외한다.
두 실행에 진단용 `save_reference=true`가 있었으므로 저장 시간을 빼도 production
`false`와 메모리/후속 cache 상태까지 같은 것은 아니다. GPU 할당 대기는 첫 실행 약
84분, 둘째 실행 26분 58초였으며 모델 추론 시간에 더하지 않는다. 둘째 실행의 서버
`running_gpu` → `complete`는 626.80초다. 사용자 선택대로 L4만 기다렸고 업그레이드는 없었다.

| Peak / 산출물 | Reference | Optimized |
| --- | ---: | ---: |
| PyTorch allocated | 14.22 GiB | 13.29 GiB |
| PyTorch reserved | 14.84 GiB | 14.30 GiB |
| Device used, sampled | 15.12 GiB | 14.59 GiB |
| Process RSS, sampled | 26.75 GiB | 25.84 GiB |
| GLB bytes | 38,137,548 | 37,274,032 |
| Exported triangles | 986,734 | 971,472 |

Device used/RSS는 100 ms 샘플로 짧은 peak를 놓칠 수 있다. PyTorch allocator peak는
native/context의 모든 할당을 포함하지 않는다. 64 GiB host RAM limit은 VRAM이 아니다.
Face target은 strict cap이 아니다. 두 실제 GLB를 다운로드해 embedded PBR를 검증하고,
Blender에서 원본 normals/UV를 유지한 동일 카메라로 렌더했다. 둘 다 4096² base-color와
metallic/roughness가 연결됐다. 이는 생산 품질·watertight·다른 입력의 성공 보장은 아니다.

### 수치 동등성은 실패

Source/model pins, 입력 SHA, seed/precision/설정, pipeline JSON 및 계산된 camera는
일치했다. 하지만 strict 비교(`atol=1e-6`, `rtol=1e-5`, 좌표/faces exact)에서
shape/texture 좌표는 **10,579 대 10,577**, raw mesh tensor 크기도 달라 통과하지 못했다.
좌표 교집합은 10,577개지만 이것으로 latent/mesh/PBR 동등성을 대신 판정하지 않는다.
모든 저장 tensor는 finite였다. 허용 오차를 늘리거나 mesh를 정렬해 통과 처리하지 않았다.

두 로그에서 FlexGEMM의 `(2^13, 512, 512, 27)` SPLITK 자동 선택이 1 대 2로 달랐다.
이는 통제되지 않은 수치/성능 변동 요인이다. 현재 자료만으로 hook 오류인지,
native kernel 비결정성인지 또는 둘 다인지 원인을 확정할 수 없다.
따라서 엄격한 동등성과 일반화된 가속 주장은 보류한다.

### 관측된 재사용

Optimized에서 DINO backbone 1회 로드/3회 공유, 특징 cache 2 hit/2 miss,
peak 35.04 MiB/예산 128 MiB, eviction 0이었다. Projection은 32³ → 2,500,
64³ → 10,577 좌표만 계산했지만 NAF target 1024를 낮추지는 않았다.
Static cache는 네 flow 모두 hit가 있었고 각각 약 239–253 MiB를 사용했다.
sampler/worker 종료 시 보관한 cache tensor는 0이다. Reference의 load counter 0은
공유 loader를 계측하지 않았다는 뜻이지 DINO를 로드하지 않았다는 뜻이 아니다.

별도 health HTTP 측정은 관측 cold median 7.32초(n=2), 즉시 warm median 0.34초(n=6)였다.
네트워크/연결 비용을 포함한 health 응답이며 모델 추론/처리량이 아니다.

[비공개 경로를 제거한 측정·검증 기록](evidence/modal-l4-native-2026-11-14.performance.json).
진단 뒤 `save_reference=false`로 main을 재배포했다. min=0/max GPU=1을 유지하며
추가 생성·자동 재시도는 제출하지 않았다.

## 다음 우선순위

1. **동등성부터:** reference/reference 반복과 고정 kernel 선택의 frozen-input 연산별 비교로
   최초 divergence를 찾는다. 그 후 DINO/projection/static cache를 각각 ablation한다.
2. **Export/UV:** optimized 단계 합계의 46%다. 로그의 parameterization 구간이 약
   190초 대 242초였으므로 먼저 세분화한다. CPU-bound로 단정하거나 remesh/UV를 없애지 않는다.
3. **모델 로드:** optimized의 22%(117초)다. scale-to-zero를 유지하는 worker 내 모델 재사용을
   조사하되 job isolation, hook 복원, 메모리와 idle 비용을 함께 검증해야 한다.
4. **신경망 세분화:** 160초 안의 SS/LR/HR/texture, cache clone/동기화와 native NAF 비용을
   측정한다. DINO cache는 eviction이 없어 예산 증설만으로 개선될 근거가 없었다.

후보의 위험/통과 조건은 [최적화 재조사](upstream-runtime.md),
옵션/예산/검증 범위는 [최적화 구현](optimizations.md)에 기록한다.
추가 유료 실험, 예산 변경, model residency 변경은 이번 측정에 포함하지 않았다.

## 실행 계획

```bash
python scripts/benchmark_upstream.py --image object.png
python scripts/benchmark_upstream.py --image object.png --execute --output-dir outputs/native-bench
```

첫 명령은 dry-run이며 모델 다운로드나 CUDA 초기화/Modal 호출이 없다.
둘째 명령만 GPU를 실행한다. 교차 순서 반복, 동일 옵션, 모델 로딩/후처리를 포함한
asset wall time을 기록하고 실패/OOM에서는 중단한다. 위 Modal pilot과 달리
교차 반복 실험용이며, 기존 결과 폴더나 완료 여부가 불분명한 remote job을 재제출하지 않는다.
