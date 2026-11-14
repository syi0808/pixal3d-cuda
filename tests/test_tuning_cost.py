import json

import pytest

from scripts.summarize_native_tuning import cost, summarize


def test_normalized_cost_selects_measured_tier_and_requires_cross_tier_pixels(tmp_path):
    for cores, atlas in ((2, 200), (4, 100), (8, 50)):
        report = {"status": "complete", "gpu": "NVIDIA L4", "fixture_hashes": {"charts": "same"},
                  "opencv": "4.12.0", "chart_trials": [{"topology_preserved": True, "compute_seconds": atlas,
                                                         "pack_seconds": 1} for repeat in range(2)],
                  "inpaint_trials": [{"workers": workers, "seconds": 40 / workers, "byte_exact": True,
                                      "output_sha256": ["same"] * 4} for repeat in range(2) for workers in (1, 2, 4)],
                  "stages": [{"name": name, "elapsed_seconds": seconds} for name, seconds in
                             (("load_covered_no_init", 10), ("neural_covered_no_init", 100), ("capture_export", 270))],
                  "baseline_export": {"components": {name: {"inclusive_wall_seconds": value} for name, value in
                                                       (("xatlas.compute_charts", 200), ("xatlas.pack_charts", 1), ("cv2.inpaint", 40))}}}
        (tmp_path / f"result-{cores}.json").write_text(json.dumps(report))
    result = summarize(tmp_path)
    assert result["selected"]["cores"] == 8 and result["selected"]["inpaint_workers"] == 4
    assert cost(10, 2, 16) > cost(10, 2, 0)
    path = tmp_path / "result-8.json"
    report = json.loads(path.read_text())
    report["inpaint_trials"][0]["output_sha256"][0] = "different"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="outputs/version"):
        summarize(tmp_path)
