"""Reclassify compact immutable saved traces; no images or inference."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from audit_controls import BUTTONS, direction_oracle
from audit_long import summarize


def reclassify(events, previous_button):
    result = copy.deepcopy(events)
    for event in result:
        current = next((name for name, action in BUTTONS.items() if action == event["button"]), None)
        if (event.get("reason") == "uncommanded-horizontal-motion" and event["status"] == "violation"
                and event.get("old_cells") is not None and event.get("new_cells") is not None):
            reason, _ = direction_oracle(event["old_cells"], event["new_cells"], current, 0, previous_button)
            if reason == "possible-delayed-horizontal":
                event.update(status="unknown", reason=reason, original_status="violation")
        previous_button = current
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("docs/latency-historical-inputs.json"))
    parser.add_argument("--gt-support", type=Path, default=Path("docs/horizontal-latency-gt-support.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    payload, gt_bytes = args.inputs.read_bytes(), args.gt_support.read_bytes()
    inputs, support = json.loads(payload), json.loads(gt_bytes)
    if len(payload) > 1024 ** 2 or inputs["test_used"] or any(support[key] != value for key, value in
            (("train_lag_max", 1), ("train_amplitude_max", 1), ("train_unexplained", 0), ("validation_outside", 0))):
        raise ValueError("compact input/GT fixed lag1 proof mismatch")
    report = {"status": "historical-trace-reclassification-no-inference-no-success", "test_used": False,
              "model_forward_used": False, "input_sha256": hashlib.sha256(payload).hexdigest(),
              "gt_support_sha256": hashlib.sha256(gt_bytes).hexdigest(), "gt_source_report_sha256": support["source_report_sha256"],
              "seed_report_sha256": inputs["seed_report_sha256"],
              "sources": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                          ("audit_controls.py", "audit_long.py", "reclassify_latency.py")}, "traces": []}
    for trace in inputs["traces"]:
        if trace["manifest_sha256"] != support["manifest_sha256"]: raise ValueError("historical dataset lineage mismatch")
        result = {key: value for key, value in trace.items() if key != "seeds"}; result["seeds"] = []
        for seed in trace["seeds"]:
            if seed["previous_button"] != "noop": raise ValueError("fixed seed release proof mismatch")
            paths = {}
            for name, path in seed["paths"].items():
                original = path["events"]; events = reclassify(original, seed["previous_button"])
                paths[name] = {"original": summarize(original), "reclassified": summarize(events),
                               "changed_steps": [i for i, (a, b) in enumerate(zip(original, events, strict=True)) if a["status"] != b["status"]],
                               "events": events}
            result["seeds"].append({"file": seed["file"], "frame": seed["frame"], "paths": paths})
        report["traces"].append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream: json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output), "traces": len(report["traces"])}))


if __name__ == "__main__": main()
