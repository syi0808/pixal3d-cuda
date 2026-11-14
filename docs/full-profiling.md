# Fixed-kernel 전체 경로 검증·프로파일

2026-11-14 KST, 실제 Modal L4 한 장에서 [prefix 진단](profiling.md)을 확장했다.
완료된 SV job의 준비된 512² RGBA 신발과 실제 MoGe camera를 재사용했다. Seed 42,
geometry 1024, texture NAF 1024, 원본 BF16 flow/FP16 decoder와 각 12-step schedule을
유지했다. Matting/crop, 모델 다운로드, 하드웨어/품질 fallback은 없다.

## HR/texture 포함 149개 체크포인트 exact

| 프로필 | 비교 범위 | 결과 |
| --- | --- | --- |
| Reference 첫 실행 | 전체 native 신경망, 커널 calibration | private 기준 fixture |
| Reference 반복 | 같은 모델/입력/설정 | **149개 checkpoint exact** |
| Optimized | DINO 공유/cache + active projection + static conditioning cache | **149개 checkpoint exact** |

Conditioning, CPU/CUDA RNG, noise, SS/LR/HR/texture의 각 step, shape/texture latent,
선택된 decoder convolution, 최종 vertices/faces/재질 attrs/coords를 포함한다. 모든 캡처
tensor는 finite였다. Float `atol=1e-6`, `rtol=1e-5`, integer exact 조건을 바꾸지 않았으며,
실제로는 tolerance 이내 수준을 넘어 캡처 값이 bit-exact였다. 최종 latent 좌표 10,637개,
raw mesh는 vertices 3,300,965 / faces 7,495,674, 재질 voxel은 3,294,505개였다.

고정/확인한 항목:
- Masked submanifold forward의 top-level SPLITK=1.
- 첫 실행에서 얻은 inner Triton/native cache를 보존하고, 비교 실행에서 새로운
  `_benchmark`/`_bench` 요청을 거절했다. 세 실행의 실제 cache 내용도 같았다.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`, cuDNN benchmark off / deterministic on,
  PyTorch·cuDNN·FlexGEMM TF32 off. 실행 전후 설정이 같았다.

이 제어는 **진단 프로세스에만** 적용했다. Production 기본값/배포는 바꾸지 않았다.
이전 실험과 TF32/cuDNN 설정이 다르므로 이전 GLB의 literal replay가 아니며,
서로 다른 cold container/GPU/입력이나 MV/1536/T4의 전역 결정성을 증명하지 않는다.
`torch.use_deterministic_algorithms`는 켜지 않았고 custom native 연산의 보장도 아니다.
원본 standalone CLI를 별도 실행한 비교도 아니다.

## 모델 로드: 다운로드보다 생성자 초기화

| 구간 | Reference 로드 | Optimized 로드 |
| --- | ---: | ---: |
| 계측 전체 | 126.35 s | 122.33 s |
| Neural model 생성자 합 | 109.01 s | 101.22 s |
| `nn.init` fill 합 (생성자 등에 포함) | **102.69 s** | **95.93 s** |
| Safetensors mapping/load 호출 | 8.54 s | 12.68 s |
| `load_state_dict` 적용 | 5.60 s | 6.22 s |
| DINO loader 합 | 1.19 s / 4회 | 1.07 s / 1회 |
| NAF loader 합 | 1.03 s | 0.07 s |

**중첩된 시간은 더하면 안 된다.** `normal_`과 `kaiming_uniform_`가 대부분의 초기화
시간을 차지했다. 체크포인트로 덮어쓰기 전 생성자에서 가중치를 채우는 비용이 로드의
약 78–81%다. 실제 weight 읽기/mapping 시간에는 메모리·page-fault 비용도 포함될 수
있으며 순수 디스크 I/O benchmark가 아니다.

Reference를 먼저 읽고 optimized는 신경망/두 export 뒤에 읽었으므로 OS cache/CPU
클럭/메모리 상태와 순서를 통제한 loader speedup 실험은 아니다. DINO 1회 로드와
3회 공유는 확인했지만 전체 4초 차이를 DINO 공유 효과로 해석하지 않는다.

NAF는 이번 캡처에서 HR/texture의 동일 image/grid 크기가 한 라벨로 합산됐다.
공개 수치는 NAF 전체 합만 사용한다. 진단 후 extractor identity로 라벨을 분리하는
metadata-only 수정과 회귀 테스트를 추가했다. 그 수정 때문에 GPU를 재실행하지 않았다.

## 같은 raw mesh의 UV/export 2회

첫 neural 실행의 vertices/faces/attrs/coords를 CPU fixture로 고정했다. 매 export에서
재로딩해 네 tensor의 SHA-256이 같은지 확인했다. 원본 remesh/1M face target/4K PBR와
WebP GLB export를 모두 유지했다.

| 구간 | 1회차 | 2회차 |
| --- | ---: | ---: |
| 전체 export stage (fixture 읽기/검사 포함) | 292.88 s | 260.33 s |
| UV unwrap | 233.74 s | 203.03 s |
| └ xatlas `compute_charts` | **229.25 s** | **198.64 s** |
| └ GPU CuMesh `compute_charts` | 1.20 s | 1.13 s |
| └ xatlas `pack_charts` | 1.04 s | 1.05 s |
| `cv2.inpaint` 4회 합 | **38.04 s** | **38.45 s** |
| Volume attribute sampling | 4.48 s | 1.98 s |
| GLB/WebP 직렬화 | 6.75 s | 7.02 s |

UV 비용은 packing이 아니라 **xatlas chart 생성/parameterization**에 몰려 있었다.
해당 호출의 process CPU 시간은 449.48/389.28초로 wall time의 약 1.96배다.
2-core CPU 제한 아래에서 CPU를 거의 모두 사용한다는 근거이며, GPU/VRAM 연산이
UV 시간을 지배한다고 볼 자료는 아니다. `cv2.inpaint`는 CPU/wall 비율이 약 1이다.
동기화/함수 계측 오버헤드가 포함된 각 1회 수치다.

두 결과는 embedded PBR를 검증했고, 실제 GLB를 Blender로 다시 열어 양쪽 4096²
texture 연결과 렌더를 확인했다. 각각 37,041,768 / 37,027,676 bytes,
972,805 / 972,587 triangles다. **같은 raw 입력이어도 export의 삼각형 수와 GLB hash는
달랐다.** 중간 remesh/simplify/UV 입력까지 고정한 것이 아니므로 시간 차이를 순수
스케줄링 잡음으로만 설명하거나 byte-identical GLB 동등성을 주장하지 않는다.
이는 neural tensor parity와 별개의 native export 재현성 범위다.
삼각형 수는 GLB index accessor 기준이며 Blender import에서는 각각 972,799/972,583개를
읽었다. 두 계측을 혼동하거나 이 렌더를 완벽한 topology 품질 검증으로 간주하지 않는다.

## 커널 관측과 해석

HR/texture 각각 sampling step index 1을 reference 반복/optimized에서 캡처했다.
HR의 `aten::addmm` 호출은 490→455, texture는 245→210으로 줄었지만,
HR flash attention은 약 795/792 ms, texture는 약 397/398 ms로 그대로였다.
Static conditioning cache가 전체 flow/decoder/NAF 비용을 없애는 것은 아니다.

전체 neural 진단 stage는 calibration 159.48 / reference 반복 140.86 / optimized
136.73초였다. CPU tensor snapshot/digest/strict 비교와 두 step의 profiler 비용이
포함되므로 production latency나 반복 가속률로 사용하지 않는다. `aten` attribution과
같은 CUDA kernel event를 중복 합산하지 않는다.

## 다음 최적화 우선순위

1. **Checkpoint coverage를 확인한 no-init construction:** missing/unexpected key와
   초기화가 필요한 buffer를 먼저 audit한다. 단순히 모든 initializer를 끄면 안 된다.
2. **xatlas CPU scaling:** post-remesh/UV 입력을 추가로 고정한 뒤 같은 품질에서
   2/4/8-core를 비교한다. Production CPU를 이번에 늘리지는 않았다.
3. **Inpaint 분리:** 4개 호출을 세분화해 동일한 mask/결과를 유지하는 병렬화 가능성을
   검사한다. 4K 축소나 inpaint 제거는 동일 품질 backend 최적화로 세지 않는다.
4. 그 뒤 cache 예산/FFN/GEMM을 ablation한다. 현재 주요 비용은 더 많은 VRAM 자체보다
   CPU 초기화와 xatlas 쪽에 있다.

## 실행·범위

```bash
python scripts/profile_modal_native.py --scope full \
  --source-job COMPLETED_NATIVE_JOB_ID --output-dir outputs/full-probe
```

기본 dry-run; `--execute`에서만 GPU 요청이 발생한다. L4 한 장, CPU 2개,
host RAM 16–64 GiB, timeout 30분, retry 0이며 운영 GPU가 바쁘면 제출하지 않는다.
Reference fixture payload 상한은 3 GiB다. 실제 캡처는 3,010,327,004 bytes였고,
무거운 `.pt`/Kineto trace/GLB/입력은 private volume 또는 ignored outputs에만 둔다.
CPU/GPU 계측 stage 합은 1,238.96초로 queue/import 등 stage 밖 시간은 제외한다.
GPU 호출 1회로 완료했고 진단 App 종료 및 active container 0을 확인했다.

[Sanitized 측정 기록](evidence/modal-l4-full-2026-11-14.profiling.json).
별도 MLX 저장소와 production runtime/배포 옵션은 수정하지 않았다.
