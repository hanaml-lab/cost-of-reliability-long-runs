"""Closed-form results of the report (Section 3 and Appendix B), checked against worked examples."""
import math

import pytest

from belay.theory import break_even_recall, check_gain, half_life_multiplier, retry_success, young_interval


def test_a_check_on_a_reliable_step_needs_recall_above_the_break_even():
    # p = 0.98, 3% false flags, retries that succeed 80% of the time: break-even recall 0.98*0.03*0.2/(0.02*0.8).
    assert break_even_recall(p=0.98, f=0.03, q=0.8) == pytest.approx(0.3675)
    assert check_gain(p=0.98, r=0.30, f=0.03, q=0.8) < 0 < check_gain(p=0.98, r=0.45, f=0.03, q=0.8)


def test_no_recall_rescues_a_check_with_poor_retries_and_false_flags():
    # 5% false flags and retries at 70%: the break-even recall exceeds one.
    assert break_even_recall(p=0.98, f=0.05, q=0.7) > 1.0
    assert check_gain(p=0.98, r=1.0, f=0.05, q=0.7) < 0


def test_retry_after_a_failure_is_less_reliable_under_frailty():
    # Exponential frailty (variance 1): E[exp(-sZ)] = 1/(1+s). First attempt 1/1.1; retry (1/1.1 - 1/1.2)/(1 - 1/1.1) = 5/6.
    first, retry = retry_success(eta=0.1, frailty_var=1.0, rho=1.0)
    assert first == pytest.approx(1 / 1.1)
    assert retry == pytest.approx(5 / 6)


def test_without_frailty_a_retry_is_a_fresh_draw():
    first, retry = retry_success(eta=0.1, frailty_var=0.0, rho=1.0)
    assert retry == pytest.approx(first) == pytest.approx(math.exp(-0.1))


def test_check_interval_follows_young():
    # sqrt(2 * 0.5 / (0.04 * 1.0)) = 5
    assert young_interval(check_cost=0.5, hazard=0.04, step_cost=1.0) == pytest.approx(5.0)


def test_checks_stretch_the_half_life_by_at_most_one_over_one_minus_recall():
    assert half_life_multiplier(hazard=1e-6, recall=0.75) == pytest.approx(4.0, rel=1e-4)
    assert half_life_multiplier(hazard=0.05, recall=0.75) < 4.0
