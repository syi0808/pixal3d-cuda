# 공식 경로의 재사용·투영 최적화

2026-11-13 적용. 별도 runtime이나 이전 포팅 연산을 추가하지 않는다. 고정된
공식 클래스에 worker/job 수명의 hook을 설치하고 완료/실패 시 복원한다.
가중치, dtype, 12-step schedule, texture NAF 1024, native decoder와 remesh/PBR는
바꾸지 않는다. 이후 실제 L4 pilot에서 두 profile의 PBR GLB 생성은 성공했지만
strict native tensor parity는 실패했다. [측정과 한계](performance.md)를 참조한다.

## 적용 항목

| 항목 | 기본값 | 동작 |
| --- | --- | --- |
| DINO backbone 공유 | 켬 | 같은 pinned 경로의 backbone 1개를 4개 extractor가 공유 |
| DINO 특징 재사용 | 128 MiB | 정규화 입력의 실제 값·shape·dtype·device·autocast 검증, 모델별 cache |
| Active projection | 켬 | 원본 grid buffer에서 active 좌표 선택 후 원본 projection/grid_sample 호출 |
| Static conditioning cache | flow당 256 MiB | 원본 proj_linear 및 global to_kv 결과만 재사용 |

DINO cache는 입력 snapshot과 출력 clone을 예산에 포함하고 LRU로 제한한다.
큰 MV batch가 예산을 넘으면 저장하지 않고 같은 원본 연산을 실행한다.
동일 크기의 다른 이미지를 입력 값 비교 없이 hit 처리하지 않는다.

Conditioning cache는 sampler 호출 한 번(한 flow의 step/CFG)에만 유효하다.
positive/negative 입력 snapshot을 root forward마다 확인한다. noise/timestep/
self-attention은 캐시하지 않는다. 원본 BF16 cast와 ElasticModuleMixin을 유지하고
실제 연산 dtype/device/autocast를 구분한다. 순차 traversal에서 thrashing이 없도록
출력은 fixed admission으로 보관한다. 새 context snapshot 공간만 뒤에서 회수한다.
sampler 반환/오류 전에 비우므로 CPU offload 뒤에 GPU 결과가 남지 않는다.

예산은 보관한 tensor payload 상한이며 전체 VRAM/allocator/임시 clone/모델의
hard limit이 아니다. 준비·입력 비교·clone 비용도 GPU 벤치마크에 포함해야 한다.

Active projection은 현재 worker의 batch=1 SV/MV average 경로를 지원한다.
좌표 순서·중복·원본 linspace/회전 grid·camera·border sampling을 유지하며,
SS는 원본 dense 경로다. NAF 맵 생성/NATTEN은 원본 그대로이고 target을 줄이지
않는다. grid override와 buffer 선택은 예외에서도 복원한다.

## 사용과 비교

```bash
pixal3d-cuda generate --image object.png --output-dir outputs/reference --reference --save-reference
pixal3d-cuda generate --image object.png --output-dir outputs/optimized --save-reference
python scripts/compare_native_references.py outputs/reference/reference.pt outputs/optimized/reference.pt
```

`--reference`는 같은 공식 pipeline에서 hook들만 끈다. 다른 runtime이 아니다.
세부 override와 동시에 쓰면 오류다. 개별 제어는 `--no-share-dino`,
`--dense-projection`, `--dino-cache-mb`, `--conditioning-cache-mb`다.
예산 범위는 0–2048 MiB이며 0은 해당 cache를 끈다. 옵션은 run identity에 포함된다.
Modal 기본값은 `modal.toml`, HTTP 원본 비교는 `options.reference=true`다.
HTTP 클라이언트가 cache 예산을 임의로 키울 수는 없다.

report의 `optimizations`에는 DINO load/hit/miss/retained peak, active/dense point 수,
모델별 cache hit/miss/peak 및 sampler 종료 후 0인 retained bytes가 나온다.
reference tensor는 입력에서 파생된 비공개 데이터다. 비교 도구는 source/입력 hash/
camera/설정을 확인하며 좌표·faces는 exact, 부동소수 tensor는 명시적 tolerance로
비교한다. Modal job별 경로 차이는 무시하되 입력 순서/SHA-256은 비교한다.
native tensor 통과가 최종 remesh/UV/PBR 시각적 동등성을 증명하지는 않는다.

## 검증과 남은 후보

후속 [L4 prefix profiling](profiling.md)에서 native SPLITK를 고정한 SS/LR 구간은
기준 반복/개별 hook/전체 hook 모두 73개 체크포인트에서 exact였다. SPLITK만 바꾼
negative control은 decoder 출력과 HR 좌표 수가 달랐다. HR/texture 및 전체 GLB
동등성을 확인한 것은 아니며 production에 고정 SPLITK를 새 기본값으로 넣지 않았다.

CPU 테스트는 pinned upstream 소스의 projection builder, SV/MV extractor,
ProjectAttention/SparseProjectAttention, ElasticModuleMixin을 AST로 읽어 실행한다.
DINO backbone/NAF는 작은 대역이며 sparse tensor도 최소 interface 대역이다.
따라서 pretrained 모델/native CUDA/full DiT parity 검증은 아니다.
SV/MV·grid override·경계/중복 좌표·FP32/BF16 cache·입력 변경·오류 복원·예산·수명을
검증한다. CI도 같은 SHA의 source를 checkout하며 가중치는 받지 않는다.

벤치마크 profile은 reference, reuse(DINO), projected(DINO+active projection),
optimized(cache 포함), standard(전체 GPU 상주)다. 기본은 reference/optimized
교차 순서 반복이며 `--execute` 전에는 실행하지 않는다.

NAF selected-query/타일 분할, native sparse autotuning, compile/CUDA graphs,
T4 precision 호환화는 적용하지 않았다. NATTEN/CUDA 및 pretrained 기준선 없이
연산을 바꾸지 않기 위한 구분이다. Modal SM89 빌드는 통과했고 RMBG 기본 의존성은
제거했다. L4 단일 입력에서 실제 생성/렌더와 raw tensor 비교를 실행했으며,
동등성 실패와 변동 요인을 [실제 배포/진단 범위](validation.md#v02-l4-performance-pilot)에
기록한다. 반복 가속, 원본 CLI/GLB 동등성과 mapped-view 품질은 검증하지 않았다.
