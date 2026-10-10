import numpy as np
import pytest

from hlo.eval.agreement import cohen_kappa, gwet_ac1, krippendorff_alpha_nominal, percent_agreement


def test_perfect_agreement():
    a = ["Pos", "Neg", "Neu", "Pos"]
    assert percent_agreement(a, a) == 1
    assert cohen_kappa(a, a) == pytest.approx(1)
    assert gwet_ac1(a, a) == pytest.approx(1)
    assert krippendorff_alpha_nominal(a, a) == pytest.approx(1)


def test_hand_computed_two_by_two():
    # 10 items: both "y" 6, both "n" 2, disagree 2 (y/n, n/y)
    a = ["y"] * 6 + ["n"] * 2 + ["y", "n"]
    b = ["y"] * 6 + ["n"] * 2 + ["n", "y"]
    p_o = 0.8
    p_e_kappa = 0.7 * 0.7 + 0.3 * 0.3
    assert cohen_kappa(a, b) == pytest.approx((p_o - p_e_kappa) / (1 - p_e_kappa))
    pi = np.array([0.7, 0.3])
    p_e_ac1 = (pi * (1 - pi)).sum() / 1
    assert gwet_ac1(a, b) == pytest.approx((p_o - p_e_ac1) / (1 - p_e_ac1))
    # Krippendorff nominal, 2 raters, no missing: 1 - (n-1) * D_o / D_e with pairable values
    n = 20
    d_o = 4  # 2 disagreeing units contribute 2 ordered pairs each
    n_y, n_n = 14, 6
    d_e = (n * n - n_y ** 2 - n_n ** 2) / (n - 1)
    assert krippendorff_alpha_nominal(a, b) == pytest.approx(1 - d_o / d_e)


def test_alpha_handles_missing():
    a = ["x", "y", None, "x"]
    b = ["x", "y", "y", "x"]
    assert krippendorff_alpha_nominal(a, b) == pytest.approx(1)
