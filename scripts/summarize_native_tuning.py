"""Compare measured L4 CPU tiers using explicit, normalized list-price estimates."""

import argparse
import json
from pathlib import Path
import statistics


RATES = {"l4_per_second": 0.000222, "cpu_core_per_second": 0.0000131, "gib_per_second": 0.00000222}


def cost(seconds, cores, memory_gib):
    return seconds * (RATES["l4_per_second"] + cores * RATES["cpu_core_per_second"] + memory_gib * RATES["gib_per_second"])


def summarize(directory):
    directory = Path(directory)
    reports = {cores: json.loads((directory / f"result-{cores}.json").read_text()) for cores in (2, 4, 8)}
    first = reports[2]
    baseline_pixels = first["inpaint_trials"][0]["output_sha256"]
    for cores, report in reports.items():
        if report["status"] != "complete" or report["gpu"] != "NVIDIA L4" or report["fixture_hashes"] != first["fixture_hashes"]:
            raise ValueError("CPU comparison requires matching fixtures and completed L4 measurements")
        if not all(trial["topology_preserved"] for trial in report["chart_trials"]):
            raise ValueError("UV topology check failed")
        if not all(trial["byte_exact"] for trial in report["inpaint_trials"]):
            raise ValueError("Inpaint pixel check failed")
        if (len(report["chart_trials"]) != 2 or len(report["inpaint_trials"]) != 6
                or report["opencv"] != first["opencv"]
                or any(trial["output_sha256"] != baseline_pixels for trial in report["inpaint_trials"])):
            raise ValueError("Inpaint outputs/version or repeat counts differ across CPU tiers")
    stages = {record["name"]: record["elapsed_seconds"] for record in first["stages"]}
    components = first["baseline_export"]["components"]
    original_hotspots = sum(components[name]["inclusive_wall_seconds"] for name in
                            ("xatlas.compute_charts", "xatlas.pack_charts", "cv2.inpaint"))
    common = stages["load_covered_no_init"] + stages["neural_covered_no_init"] + max(0, stages["capture_export"] - original_hotspots)
    rows = []
    for cores, report in reports.items():
        atlas = statistics.median(trial["compute_seconds"] + trial["pack_seconds"] for trial in report["chart_trials"])
        for workers in (1, 2, 4):
            inpaint = statistics.median(trial["seconds"] for trial in report["inpaint_trials"] if trial["workers"] == workers)
            total = common + atlas + inpaint
            rows.append({"cores": cores, "inpaint_workers": workers, "atlas_median_seconds": atlas,
                         "inpaint_median_seconds": inpaint, "modeled_asset_seconds": total,
                         "normalized_estimated_asset_usd": {str(memory): cost(total, cores, memory) for memory in (16, 32, 64)}})
    selected = min(rows, key=lambda row: row["normalized_estimated_asset_usd"]["32"])
    return {"schema": "pixalcuda.cpu-cost-comparison.v1", "rates_usd": RATES,
            "pricing_source": "https://modal.com/pricing", "pricing_checked_date": "2026-11-14",
            "common_seconds_from_2core_diagnostic": common, "rows": rows, "selected": selected,
            "sensitivity": [{"common_seconds": shared, "memory_gib": memory,
                             "selected": min(rows, key=lambda row: cost(shared + row["atlas_median_seconds"] + row["inpaint_median_seconds"], row["cores"], memory))["cores"]}
                            for shared in (0, 150, 300) for memory in (16, 32, 64)],
            "scope": "List-price normalized model, not an invoice or measured end-to-end baseline. Memory is fixed at 16/32/64 GiB for sensitivity. Diagnostic common time includes snapshot/capture overhead; excludes startup/queue/network/idle. Actual Modal charges max(request, usage), and host/ordering effects are not isolated."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args(argv)
    result = summarize(arguments.directory)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
