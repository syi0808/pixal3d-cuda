# 공식 runtime 최적화 현황

v0.2부터 기본/대체 backend 구분 없이 공식 Pixal3D 경로 하나만 제공한다.
이전 MLX-derived 연산 구현과 전용 벤치마크는 삭제했다. 과거 측정은
[이전 revision](https://github.com/syi0808/pixal3d-cuda/blob/a6ce8a3c4a68e377569b7965035a8f00d74ff9b0/docs/performance.md)
및 [보존된 증거](validation.md)에 있으며 새 runtime 수치가 아니다.

## 현재 상태

- 공식 low-VRAM / SDPA / FlexGEMM / BF16 flow / FP16 decoder를 사용한다.
- 원본 texture NAF 1024와 CUDA remesh/PBR exporter를 유지한다.
- native 이미지 빌드, 실제 GPU inference, 원본 CLI 동등성, 속도 개선은 아직 미검증이다.
- CPU 계약 테스트는 wiring/입력/상태/출력 구조만 검증한다.
- DINO 공유/특징 cache, active projection, flow-scoped conditioning cache를 적용했다.
  pinned 원본 함수와의 CPU 비교를 추가했지만 pretrained GPU 동등성은 미검증이다.
- GPU/Modal 실행은 이 재구축에서 시작하지 않았다.

## 우선순위

1. 원본 CLI와 wrapper의 동일 1024 입력 baseline 및 raw tensor fixture 확보.
2. DINO backbone 공유 및 이미지 크기별 feature 재사용.
3. active 좌표 projection / target 1024를 유지하는 native NAF query 최적화.
4. per-block static conditioning cache 및 native sparse algorithm tuning.
5. model residency / compile 실험. T4 precision 호환화는 별도 검증.

각 후보의 위험과 통과 조건은 [최적화 재조사](upstream-runtime.md)에 정리했다.
적용된 항목의 옵션/예산/검증 범위는 [최적화 구현](optimizations.md)에 기록한다.

## 실행 계획

```bash
python scripts/benchmark_upstream.py --image object.png
python scripts/benchmark_upstream.py --image object.png --execute --output-dir outputs/native-bench
```

첫 명령은 dry-run이며 모델 다운로드나 CUDA 초기화/Modal 호출이 없다.
둘째 명령만 GPU를 실행한다. 교차 순서 반복, 동일 옵션, 모델 로딩/후처리를 포함한
asset wall time을 기록하고 실패/OOM에서는 중단한다. 아직 측정된 가속 수치는 없다.
