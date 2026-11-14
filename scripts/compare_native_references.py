"""Compare private native tensors; structural equality is required before tolerances."""

import argparse
import json
import math
from pathlib import Path

import torch


def compare(reference_path, candidate_path, *, atol=1e-6, rtol=1e-5):
    if not all(math.isfinite(value) and value >= 0 for value in (atol, rtol)):
        raise ValueError("Tolerances must be finite and nonnegative")
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
    if reference["schema"] != "pixalcuda.native-reference.v1" or candidate["schema"] != reference["schema"]:
        raise ValueError("Unsupported native reference schema")
    ignored = {"optimizations", "save_reference", "skip_weight_init", "inpaint_workers"}
    identities = [{key: value for key, value in record["identity"].items() if key not in ignored}
                  for record in (reference, candidate)]
    for identity in identities:
        if "inputs" in identity:
            identity["inputs"] = [{"sha256": item["sha256"]} for item in identity["inputs"]]
    if identities[0] != identities[1] or reference["actual_geometry_resolution"] != candidate["actual_geometry_resolution"]:
        raise ValueError("Input/source/camera/precision/geometry settings differ; not a controlled comparison")
    results = {}
    for group, names in {"shape": ("coordinates", "features"), "texture": ("coordinates", "features"),
                         "mesh": ("vertices", "faces", "attrs", "coords")}.items():
        for name in names:
            expected, actual = reference[group][name], candidate[group][name]
            key = f"{group}.{name}"
            if expected.shape != actual.shape or expected.dtype != actual.dtype:
                results[key] = {"passed": False, "reason": "shape/dtype mismatch"}
                continue
            if name in ("coordinates", "coords", "faces"):
                results[key] = {"passed": bool(torch.equal(expected, actual)), "comparison": "exact"}
                continue
            expected, actual = expected.double(), actual.double()
            finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
            difference = actual - expected
            results[key] = {"passed": finite and bool(torch.allclose(expected, actual, atol=atol, rtol=rtol)),
                            "finite": finite, "max_abs": float(difference.abs().max()) if finite and difference.numel() else None,
                            "relative_l2": float(difference.norm() / expected.norm().clamp_min(1e-30)) if finite else None}
    return {"passed": all(record["passed"] for record in results.values()), "atol": atol, "rtol": rtol, "tensors": results,
            "scope": "Native tensor parity only; not a visual-quality, remesh or PBR bake equivalence result."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    arguments = parser.parse_args(argv)
    result = compare(arguments.reference, arguments.candidate, atol=arguments.atol, rtol=arguments.rtol)
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result["passed"]:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    main()
