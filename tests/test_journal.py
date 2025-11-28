"""Durable execution (Section 5): journalled steps, deterministic replay and idempotency keys."""
import pytest

from belay.journal import Crash, Journal, Service, Step, run


def workflow(model_calls):
    def think(state):
        model_calls.append(len(state))
        return f"plan-{len(model_calls)}"

    return [
        Step("plan", "model", think),
        Step("read", "tool", lambda state: "invoice 42"),
        Step("pay", "effect", lambda state: {"payee": "acme", "amount": 120}),
        Step("notify", "effect", lambda state: {"to": "ap@acme"}),
        Step("close", "model", think),
    ]


def crash_then_resume(idempotency, crash):
    calls, service, journal = [], Service(), Journal()
    with pytest.raises(Crash):
        run(workflow(calls), journal, service, job="job-7", idempotency=idempotency, crash=crash)
    result = run(workflow(calls), journal, service, job="job-7", idempotency=idempotency)
    return result, service, calls


def test_a_crash_in_flight_with_keys_pays_once():
    result, service, _ = crash_then_resume(idempotency=True, crash=("pay", "in-flight"))
    assert service.executed("pay") == 1
    assert service.duplicates() == 0
    reference, _, _ = crash_then_resume(idempotency=True, crash=("close", "before"))
    assert result == reference


def test_without_keys_the_in_flight_payment_runs_twice():
    _, service, _ = crash_then_resume(idempotency=False, crash=("pay", "in-flight"))
    assert service.executed("pay") == 2
    assert service.duplicates() == 1


def test_replay_reuses_recorded_model_calls():
    result, _, calls = crash_then_resume(idempotency=True, crash=("notify", "before"))
    assert len(calls) == 2  # "plan" once before the crash, "close" once after; plan is replayed from the journal
    assert result[0] == "plan-1"


def test_restarting_an_in_memory_loop_repeats_every_side_effect():
    calls, service = [], Service()
    with pytest.raises(Crash):
        run(workflow(calls), Journal(), service, job="job-7", idempotency=False, crash=("close", "before"))
    run(workflow(calls), Journal(), service, job="job-7", idempotency=False)  # a fresh journal: start over
    assert service.duplicates() == 2
