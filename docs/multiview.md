# 이미지/카메라 매핑 입력

공식 `Pixal3DMVImageTo3DPipeline`과 dedicated MV 가중치를 사용한다. 입력 정규화는
경로·index·카메라 검증만 담당하며 별도 conditioning/신경망 구현은 없다.
실제 multi-view pretrained 생성은 아직 검증하지 않았다.

```bash
pixal3d-cuda download --multiview --skip-camera
pixal3d-cuda generate --images front.png right.png back.png left.png \
  --transforms cameras.json --output-dir outputs/native-mv
```

`cameras.json`에는 `frames` 배열이 필요하다. 각 frame은 입력 이미지의 0-based
`image_index`와 카메라 정보를 가진다. 명시적 index를 사용하면 frame 배열 순서는
달라도 된다. 모든 index는 빠짐없이 한 번씩만 나타나야 한다. index를 모두 생략하면
입력 순서로 매핑하며, 일부만 생략하는 것은 허용하지 않는다.

카메라는 다음 두 형식 중 하나다:

- `transform_matrix`: 4x4 camera-to-world matrix + `camera_angle_x`(radians).
- `azimuth_degrees`, `elevation_degrees`, `distance`, `fov_degrees`: 명시적 orbit pose.

같은 frame에서 두 형식을 섞지 않는다. root `mesh_scale`은 선택값이며 기본 1이다.
단위·축은 원본 Blender/NeRF convention(camera looks -Z, +Y up)을 따른다.
frame 0은 canonical front가 권장되며 다른 경우 출력 기준 frame이 달라질 수 있다.

```json
{
  "mesh_scale": 1.0,
  "frames": [
    {"image_index": 0, "azimuth_degrees": 0, "elevation_degrees": 0, "distance": 3.0, "fov_degrees": 20},
    {"image_index": 1, "azimuth_degrees": 90, "elevation_degrees": 0, "distance": 3.0, "fov_degrees": 20}
  ]
}
```

위 JSON은 이미지 두 장에 대응한다. view 수는 4로 고정되지 않는다. Modal에는 별도의
16-view 운영 제한이 있다. `--num-views`는 명시적 prefix 선택이며 자동 token 축소가 아니다.
`--views-dir` + `transforms.json`도 가능하며 각 frame의 상대 `file_path`는 해당 디렉터리
밖으로 나갈 수 없다.

입력 이미지는 카메라에 맞춰 framing되어 있어야 하며 recrop하지 않는다. alpha가
있으면 존중하고, matting은 원본 BiRefNet 경로를 사용한다. `--no-rembg`로 생략할 수 있다.
임의의 여러 사진에 자동 카메라 calibration을 제공하는 기능은 아니다.
기본 출력은 1024 geometry / 4096 texture / 원본 remesh다.
