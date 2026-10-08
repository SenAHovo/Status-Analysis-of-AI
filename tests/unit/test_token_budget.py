"""Token budget: estimator, provider usage and the ledger."""

import pytest

from ai_status_report.token_budget import (
    RUN,
    BudgetExceeded,
    BudgetLimits,
    TokenLedger,
    estimate_messages,
    estimate_tokens,
    parse_usage,
    usage_tokens,
)

LIMITS = BudgetLimits(capacities={"run": 100, "output": 20, "search_requests": 3, "window": 1000})


# --- estimator ---

def test_estimate_tokens_grows_with_text():
    small = estimate_tokens("short")
    large = estimate_tokens("short " + "词" * 500)
    assert 0 < small < large


def test_estimate_messages_adds_overhead():
    single = estimate_messages([{"role": "user", "content": "hi"}])
    assert single > estimate_tokens("hi")


# --- provider usage parsing ---

def test_usage_tokens_prefers_total():
    assert usage_tokens({"total_tokens": 9, "prompt_tokens": 4, "completion_tokens": 5}) == 9
    assert usage_tokens({"prompt_tokens": 4, "completion_tokens": 5}) == 9
    assert usage_tokens({}) == 0
    assert usage_tokens({"total_tokens": -1}) == 0


def test_parse_usage_allowlists_non_negative_ints():
    result = parse_usage({"usage": {"total_tokens": 5, "other": "secret", "prompt_tokens": True}})
    assert result == {"total_tokens": 5}


# --- ledger ---

def test_reserve_settle_releases_capacity():
    ledger = TokenLedger("run-1", LIMITS)
    token = ledger.reserve(RUN, 40)
    assert ledger.remaining(RUN) == 60
    ledger.settle(token, 25)
    assert ledger.actual[RUN] == 25
    assert ledger.remaining(RUN) == 75


def test_reserve_over_limit_raises():
    ledger = TokenLedger("run-1", LIMITS)
    with pytest.raises(BudgetExceeded):
        ledger.reserve(RUN, 101)


def test_cancel_releases_reservation():
    ledger = TokenLedger("run-1", LIMITS)
    token = ledger.reserve(RUN, 40)
    ledger.cancel(token)
    assert ledger.remaining(RUN) == 100
    with pytest.raises(KeyError):
        ledger.settle(token, 10)


def test_unknown_usage_is_kept_as_consumed():
    ledger = TokenLedger("run-1", LIMITS)
    token = ledger.reserve(RUN, 30)
    ledger.settle(token, None)  # provider never reported actuals
    assert ledger.unknown[RUN] == 30
    assert ledger.remaining(RUN) == 70


def test_meter_records_metered_actions():
    ledger = TokenLedger("run-1", LIMITS)
    assert ledger.meter("search_requests") is True
    assert ledger.remaining("search_requests") == 2
    ledger.meter("search_requests")
    ledger.meter("search_requests")
    with pytest.raises(BudgetExceeded):
        ledger.meter("search_requests")


def test_estimates_and_actuals_stay_separate():
    ledger = TokenLedger("run-1", LIMITS)
    token = ledger.reserve(RUN, 50)
    ledger.settle(token, 35)
    assert ledger.estimated[RUN] == 50
    assert ledger.actual[RUN] == 35
    assert RUN not in ledger.unknown


def test_snapshot_round_trip_preserves_ledger():
    ledger = TokenLedger("run-1", LIMITS)
    token = ledger.reserve(RUN, 30)
    ledger.settle(token, 22)
    restored = TokenLedger.from_snapshot(ledger.snapshot())
    assert restored.run_id == "run-1"
    assert restored.actual[RUN] == 22
    assert restored.remaining(RUN) == 78


def test_unknown_budget_kind_raises():
    ledger = TokenLedger("run-1", LIMITS)
    with pytest.raises(KeyError):
        ledger.check("nope", 1)
