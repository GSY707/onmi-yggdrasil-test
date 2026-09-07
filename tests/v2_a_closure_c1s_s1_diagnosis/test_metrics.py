import pytest

from yggdrasil_v2.v2_a.closure_c1s_s1_diagnosis.metrics import (
    answer_margin,
    balanced_accuracy,
    family_binary_summary,
    paired_cluster_bootstrap,
    wilson_interval,
)


def test_answer_margin_is_correct_logsumexp_wrong() -> None:
    result = answer_margin([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0]], [0, 1])
    assert result[0] == pytest.approx(3.0 - __import__("math").log(2.0))
    assert result[1] == pytest.approx(2.0 - __import__("math").log(2.0))


def test_wilson_and_cluster_bootstrap_are_deterministic() -> None:
    interval = wilson_interval(2, 4)
    assert interval["point"] == 0.5
    left = paired_cluster_bootstrap([1, 2, 10, 20], [0, 1, 8, 18], ["a", "a", "b", "b"], seed=7, samples=200)
    right = paired_cluster_bootstrap([1, 2, 10, 20], [0, 1, 8, 18], ["a", "a", "b", "b"], seed=7, samples=200)
    assert left == right
    assert left["point"] == pytest.approx(1.5)


def test_balanced_accuracy_refuses_one_class_as_valid_balanced_metric() -> None:
    result = balanced_accuracy([1, 1], [1, 1])
    assert result["both_classes"] is False
    assert result["balanced_accuracy"] is None


def test_family_summary_does_not_hide_cps_failure_with_overall_pool() -> None:
    # ERE is 16/16 while CPS is 12/16; the family cells remain visible.
    predicted = [1] * 16 + [1] * 12 + [0] * 4
    target = [1] * 32
    families = ["ERE"] * 16 + ["CPS"] * 16
    report = family_binary_summary(predicted, target, families)
    assert report["families"]["ERE"]["balanced_accuracy"] is None
    assert report["families"]["CPS"]["accuracy"]["point"] == pytest.approx(0.75)
    assert report["overall"]["accuracy"]["point"] == pytest.approx(28 / 32)


def test_nonfinite_and_shape_errors_are_rejected() -> None:
    with pytest.raises(ValueError):
        answer_margin([[1.0, float("nan")]], [0])
    with pytest.raises(ValueError):
        answer_margin([[1.0, 0.0]], [0, 1])
    with pytest.raises(ValueError):
        paired_cluster_bootstrap([1.0], [0.0], [], seed=1, samples=2)
