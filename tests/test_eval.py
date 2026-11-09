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
               for tags in ("hard", "step", "sigmoid", "gaussian")]
    assert results[0]["classification"] == results[1]["classification"] == results[2]["classification"]


def test_non_right_padded_mask_is_rejected():
    with pytest.raises(ValueError, match="right-padding"):
        evaluate_readiness(np.array([[.1, .2, .9]]), [1], [1], [.5], 1, 1,
                           valid_mask=np.array([[True, False, True]]))


def test_answer_metrics_at_the_firing_pair():
    # Вопрос 0 срабатывает на целевой паре 2 (ответ верный), 1 — преждевременно на паре 0 (неверный),
    # 2 — поздно на паре 4 (верный, но вне ±2), 3 — ни разу (пропуск считается неверным ответом).
    probabilities = np.array([[.1, .1, .9, .9, .9], [.9, .1, .1, .1, .1], [.1, .1, .1, .1, .9], [.1] * 5])
    correct = [[0, 0, 1, 1, 1], [0, 1, 1, 1, 1], [0, 0, 0, 0, 1], [1] * 5]
    metrics = evaluate_readiness(probabilities, [2, 2, 1, 2], [2, 2, 1, 2], [.5], 1, 1,
                                 correct=correct)["thresholds"]["0.5"]
    assert metrics["answer_accuracy"] == 2 / 4
    assert metrics["answer_accuracy_answered"] == 2 / 3
    assert metrics["timely_correct_pm2"] == 1 / 4
    assert metrics["answer_accuracy_on_target"] == 1 and metrics["answer_accuracy_premature"] == 0
    assert metrics["answer_accuracy_late"] == 1 and metrics["answer_accuracy_early"] is None
    assert "answer_accuracy" not in evaluate_readiness(probabilities, [2, 2, 1, 2], [2, 2, 1, 2], [.5], 1, 1)[
        "thresholds"]["0.5"]


def test_choice_probabilities_and_answer_curves():
    from online_vllm.eval.answers import AnswerTrace, answer_curves, choice_probabilities
    raw = np.array([[np.log(.2), np.log(.6), np.nan, np.nan, -1.0], [np.log(.3), np.log(.1), np.nan, np.nan, -1.0]])
    probs = choice_probabilities(raw, 2)
    np.testing.assert_allclose(probs, [[.25, .75], [.75, .25]])
    # Правильный вариант 1: модель права на паре 0 и ошибается на паре 1.
    trace = AnswerTrace(probs, correct=1, target_pair=1, visible_pair=1)
    result = answer_curves([trace], offsets=range(-1, 2))
    assert result["levels"]["chance"] == .5
    assert result["levels"]["before_visible"]["accuracy"] == 1.0
    assert result["levels"]["first_pair"]["accuracy"] == 1.0 and result["levels"]["first_pair"]["count"] == 1
    assert result["levels"]["at_target"] == pytest.approx({"count": 1, "accuracy": 0.0, "p_correct": .25,
                                                           "p_answer": .75, "chance": .5})
    assert [row["count"] for row in result["curve"]] == [1, 1, 0]
    assert result["curve"][0]["p_correct"] == pytest.approx(.75)


def test_readiness_curves_agree_with_online_metrics():
    from online_vllm.eval.readiness_plot import load_traces, readiness_curves
    probabilities = [[.1, .1, .9, .9, .9], [.9, .1, .1, .1, .1], [.1, .1, .1, .1, .9], [.1] * 5]
    payload = {"predictions": [{"record_id": str(i), "probabilities": p, "target_pair": target, "visible_pair": 2,
                                "answer_correct": [False, False, True, True, True]}
                               for i, (p, target) in enumerate(zip(probabilities, [2, 2, 2, 2]))]}
    result = readiness_curves(load_traces(payload), offsets=range(-2, 3), taus=[.5], examples=4)
    online = evaluate_readiness(np.array(probabilities), [2] * 4, [2] * 4, [.5], 1, 1)["thresholds"]["0.5"]
    firing = result["firing"]["0.5"]
    for outcome in ("premature", "on_target", "late", "missed"):
        assert firing["outcomes"][outcome] == online[f"{outcome}_rate"]
    # Сдвиги срабатываний: 0, −2, +2; четвёртый вопрос не сработал.
    assert firing["share"] == [.25, 0, .25, 0, .25]
    assert result["curve"][2]["median"] == pytest.approx(.1) and result["curve"][2]["answer_accuracy"] == 1
    # До visible_pair = 2: всплеск только у вопроса 1, на одной паре из двух.
    background = next(row for row in result["background"] if row["tau"] == .5)
    assert background == {"tau": .5, "any_pair": .25, "pair_share": .125}
    assert sorted(e["outcome"] for e in result["examples"]) == ["late", "missed", "on_target", "premature"]


def test_shift_by_target_time_separates_event_following_from_fixed_time():
    from online_vllm.eval.readiness_plot import ReadinessTrace, by_target_time
    pairs = np.arange(20)
    follows = [ReadinessTrace((pairs >= t).astype(float), t, t) for t in (4, 8, 12, 16)]
    fixed = [ReadinessTrace((pairs >= 10).astype(float), t, t) for t in (4, 8, 12, 16)]
    shift = lambda traces: [row["median"] for row in by_target_time(traces, [0], .5, groups=2)["shift"]]
    assert shift(follows) == [0, 0, 0, 0]
    assert shift(fixed) == [6, 2, -2, -6]
    groups = by_target_time(fixed, [0], .5, groups=2)["groups"]
    assert [(g["first_target"], g["last_target"], g["count"]) for g in groups] == [(4, 8, 2), (12, 16, 2)]
    assert groups[0]["outcomes"]["late"] == 1 and groups[1]["outcomes"]["premature"] == 1
