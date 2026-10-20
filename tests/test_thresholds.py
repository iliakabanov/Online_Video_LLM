import json

import pytest

from online_vllm.eval.thresholds import resolve_thresholds, sweep


def payload():
    return {"model": "uhead", "tags": "sigmoid", "split": "validation", "probabilities": [[.1, .4, .8]],
            "target_pairs": [2], "visible_pairs": [1], "temporal_patch": 2, "frame_stride": 1}


def test_threshold_grid_and_explicit_values():
    assert resolve_thresholds([.5, .1, .5]) == [.1, .5]
    assert resolve_thresholds(start=.1, stop=.3, step=.1) == [.1, .2, .3]
    with pytest.raises(ValueError):
        resolve_thresholds([.5], .1, .2, .1)


def test_sweep_preserves_all_threshold_results_and_json(tmp_path):
    result = sweep(payload(), [.3, .7])
    assert set(result["results"]) == {"0.3", "0.7"}
    assert result["results"]["0.3"]["early_rate"] == 1
    path = tmp_path / "sweep.json"; path.write_text(json.dumps(result, allow_nan=False))
    assert json.loads(path.read_text())["classification"]["count"] == 3


def test_export_format_is_accepted_and_padding_is_internal_only():
    exported = {"metadata": {"model": "uhead", "tags": "sigmoid", "split": "test", "temporal_patch": 2,
                             "frame_stride": 1},
                "predictions": [{"record_id": "short", "probabilities": [.1, .9], "target_pair": 1,
                                 "visible_pair": 1, "fps": 30.0},
                                {"record_id": "long", "probabilities": [.1, .2, .9], "target_pair": 2,
                                 "visible_pair": 1, "fps": 30.0}]}
    result = sweep(exported, [.5])
    assert result["split"] == "test"
    assert result["classification"]["count"] == 5
    assert result["results"]["0.5"]["on_target_rate"] == 1.0
