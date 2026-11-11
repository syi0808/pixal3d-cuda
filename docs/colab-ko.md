# Colab T4 실행 안내

1. README의 **Open in Colab**을 누르고 `런타임 → 런타임 유형 변경 → T4 GPU`를 선택합니다. 새 세션에서 시작합니다.
2. 설치 셀부터 차례로 실행합니다. 설치 전에 torch를 import했다면 런타임을 재시작합니다.
3. 공식 DINOv3 모델 페이지에서 접근 권한을 받고, Hugging Face read 토큰을 Colab Secrets의 `HF_TOKEN`에 저장합니다. 토큰을 코드나 출력에 붙여 넣지 않습니다.
4. 모델 다운로드·파라미터 검증 후 이미지를 한 장 업로드합니다. 모델은 약 25–30GB이며, 다운로드 전에 디스크 여유 공간 40GiB 이상을 권장합니다.
5. 기본 설정은 지오메트리 512, PBR 텍스처 1024, 각 flow 12-step입니다. 1024 이름의 refinement/texture 모델도 필요하므로 가중치 다운로드가 512 모델 하나로 줄지는 않습니다.
6. 실행 후 실제 `result.glb`를 미리 보고 다운로드합니다. `report.json`에서 단계별 시간, CUDA allocated/reserved 피크, CPU RSS를 확인합니다.

## 끊긴 실행 재개

런타임 삭제에도 보존하려면 실행 전에 `USE_DRIVE=True`를 선택합니다. 재연결 후 설치·인증을 다시 하고, 원본 이미지·설정·`RUN_NAME`을 동일하게 유지한 채 `RESUME=True`로 실행합니다. Drive에 저장하지 않은 체크포인트와 로컬 모델 캐시는 런타임이 삭제되면 사라집니다.

## 메모리가 부족할 때

- 작은 chunk로 임시 메모리를 줄일 수 있지만 실행 시간이 늘어납니다. chunk를 바꿀 때는 새 출력 디렉터리를 사용합니다.
- token/voxel guard는 품질을 몰래 낮추지 않고 명시적으로 중단합니다. guard를 올리면 CUDA OOM이 발생할 수 있습니다.
- MoGe를 생략하려면 `CAMERA="fixed"`와 수평 FOV 각도를 명시합니다. 자동 카메라 실패 시 고정 FOV로 조용히 대체하지 않습니다.
- 첫 실행은 배경이 단순한 단일 물체를 권장합니다. 투명 배경 RGBA도 사용할 수 있습니다.

## 실제 T4 검증 결과

2026-11-11에 공개 신발 이미지 한 장으로 전체 생성을 완료했습니다. geometry 512, 텍스처 1024×1024 두 장(base color, metallic/roughness), 모든 flow 12-step, 자동 카메라 설정을 유지했습니다.

- 실제 Tesla T4 / PyTorch 2.6.0+cu124 / Python 3.13.16에서 PBR 포함 GLB 생성 완료, CUDA 테스트를 포함한 29개 테스트 통과.
- 전체 1,083.05초(약 18분 3초). 일부 모델의 최초 다운로드가 포함되어 있으므로 순수 추론 시간이나 warm-cache 벤치마크가 아닙니다.
- PyTorch CUDA allocated 피크 5.45GiB / reserved 피크 5.48GiB, 샘플링한 프로세스 RSS 피크 7.21GiB. `nvidia-smi` 전체 사용량이나 강제 메모리 상한이 아닙니다.
- GLB 12,276,444바이트 / 삼각형 150,306개. simplification target은 100,000이지만 엄격한 삼각형 상한은 아닙니다.
- 초기 FP16 DINO/LR overflow를 수정해 DINO·flow·sparse decoder를 FP32로 실행합니다. 구버전 실행을 재개하지 말고 새 출력 디렉터리를 사용하세요.

인증 후 공개 샘플을 재현하려면 `python -u scripts/validate_colab_t4.py --output-dir outputs/t4-acceptance`를 실행할 수 있습니다. [단계별 측정값과 원본 기록](validation.md)을 참고하세요.

**한 입력의 완료만 검증했습니다.** 미리보기에는 표면 노이즈·음영 아티팩트가 있으며, 일반 입력 성공률·품질 동등성·warm-cache 속도·중단 후 실제 GPU 재개는 아직 검증하지 않았습니다. MLX의 메모리 측정치를 CUDA 측정값으로 간주하면 안 됩니다.
