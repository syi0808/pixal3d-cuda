# 공식 native baseline Colab 안내

현재 노트북은 `notebooks/colab_native_baseline.ipynb` 하나다. 이전 T4 실행 경로는
삭제했으며, 공식 BF16 baseline은 SM80/86/89 GPU만 허용한다. 무료 T4에서는 실행할
수 없고 자동 GPU 변경도 없다. GPU 선택/비용은 사용자가 먼저 확인해야 한다.

1. 새 Colab 세션에서 지원 GPU를 명시적으로 선택한다.
2. 설치 셀은 source revision을 출력하고 Torch 2.6/cu124 및 고정된 native 의존성을 빌드한다.
3. HF read token을 Colab Secrets의 `HF_TOKEN`에 등록한다. 토큰을 코드/채팅에 넣지 않는다.
4. pinned 가중치를 prefetch한다. RMBG-2.0은 기본 다운로드/초기화에 포함하지 않는다.
5. 이미 배경이 제거되고 framing된 이미지(투명 RGBA 권장)를 업로드해
   1024 geometry / 4096 PBR / 원본 remesh 경로를 실행한다. 기본값은 자동 matting/crop 없음이다.
6. 실패하면 출력 디렉터리의 `upstream.log`와 `report.json`을 확인한다. 자동 fallback은 없다.

**Modal SM89 이미지 빌드는 통과했지만 Colab 설치와 pretrained 생성은 미검증이다.**
과거 T4 결과를 새 경로의 성능이나 품질
증거로 사용하면 안 된다. `reference.pt`는 입력에서 파생된 비공개 데이터를 포함한다.
상세 제약 및 최적화 순서는 [공식 runtime 조사](upstream-runtime.md)를 참조한다.
