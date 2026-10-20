import numpy as np
import pytest
import torch

from online_vllm.eval.metrics import _binary_curves, evaluate_readiness, statehead_probabilities, uhead_probabilities


def test_ideal_timing_and_seconds():
    probabilities = np.array([[.1, .2, .9], [.1, .8, .9]])
    result = evaluate_readiness(probabilities, [2, 1], [1, 1], [0.5], temporal_patch=2, frame_stride=3,
                                fps=[30, 30])
    metrics = result["thresholds"]["0.5"]
    assert metrics["on_target_rate"] == 1
    assert metrics["timing_mean_frames"] == metrics["timing_std_frames"] == metrics["timing_mae_frames"] == 0
    assert metrics["timing_mean_seconds"] == 0


def test_early_late_and_miss_timing_is_conditional():
    probabilities = np.array([[.9, .1, .1, .1], [.1, .1, .1, .9], [.1, .1, .1, .1]])
    result = evaluate_readiness(probabilities, [2, 2, 2], [1, 1, 1], [0.5], 2, 1)
    metrics = result["thresholds"]["0.5"]
    assert metrics["premature_rate"] == 1 / 3 and metrics["late_rate"] == 1 / 3 and metrics["miss_rate"] == 1 / 3
    assert metrics["timing_count"] == 2 and metrics["timing_mean_frames"] == -1


def test_adapters_padding_and_shared_hard_targets():
    state = torch.tensor([[[0., 0., 1.], [0., 0., 2.]]])
    binary = torch.tensor([[1., 2.]])
    assert statehead_probabilities(state, ["silent", "prepare", "speak"]).shape == (1, 2)
    # Плоские logits позиций решений из train/head.py и train/loop.py.
    flat = statehead_probabilities(state[0], ["silent", "prepare", "speak"])
    assert flat.shape == (2,)
    np.testing.assert_allclose(flat, statehead_probabilities(state, ["silent", "prepare", "speak"])[0])
    np.testing.assert_allclose(uhead_probabilities(binary), torch.sigmoid(binary).numpy())
    result = evaluate_readiness(np.array([[.1, .9, .0]]), [1], [1], [.5], 1, 1,
                                valid_mask=np.array([[True, True, False]]))
    assert result["classification"]["count"] == 2


def test_single_class_curves_are_undefined():
    result = evaluate_readiness(np.array([[.9, .8]]), [0], [0], [.5], 1, 1)
    assert result["classification"]["auroc"] is None and result["classification"]["auprc"] is None


def test_tied_scores_perfect_and_random_binary_curves():
    assert _binary_curves(np.array([0, 1, 0, 1]), np.full(4, .5)) == (.5, .5)
    assert _binary_curves(np.array([0, 1, 0, 1]), np.array([.1, .9, .2, .8])) == (1., 1.)
    auroc, auprc = _binary_curves(np.array([0, 1, 0, 1]), np.array([.9, .8, .7, .6]))
    # Детерминированный "случайный" порядок: две пары positive/negative
    # инвертированы, поэтому сверяем известные значения, а не tie-case выше.
    assert auroc == .25 and auprc == .5


def test_same_probabilities_have_same_metrics_for_both_adapters():
    probability = np.array([[.1, .2, .8, .9]])
    state_logits = torch.log(torch.tensor([[[.45, .45, .1], [.4, .4, .2], [.1, .1, .8], [.05, .05, .9]]]))
    binary_logits = torch.logit(torch.tensor(probability))
    state = statehead_probabilities(state_logits, ["silent", "prepare", "speak"])
    binary = uhead_probabilities(binary_logits)
    np.testing.assert_allclose(state, binary)
    state_result = evaluate_readiness(state, [2], [1], [.5], 1, 1)
    binary_result = evaluate_readiness(binary, [2], [1], [.5], 1, 1)
    for key in state_result["classification"]:
        np.testing.assert_allclose(state_result["classification"][key], binary_result["classification"][key])
    assert state_result["thresholds"] == binary_result["thresholds"]


def test_evaluation_targets_do_not_depend_on_training_tag_mode():
    probability = np.array([[.1, .2, .8, .9]])
    results = [evaluate_readiness(probability, [2], [1], [.5], 1, 1, model="uhead", tags=tags)
               for tags in ("hard", "step", "sigmoid")]
    assert results[0]["classification"] == results[1]["classification"] == results[2]["classification"]


def test_non_right_padded_mask_is_rejected():
    with pytest.raises(ValueError, match="right-padding"):
        evaluate_readiness(np.array([[.1, .2, .9]]), [1], [1], [.5], 1, 1,
                           valid_mask=np.array([[True, False, True]]))
