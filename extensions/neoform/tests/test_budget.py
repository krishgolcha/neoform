import pytest

from neoform.budget import BudgetExceeded, BudgetLedger


@pytest.mark.asyncio
async def test_budget_reservations_are_idempotent_and_reconciled():
    ledger = BudgetLedger(5)
    first = await ledger.reserve("candidate", 2)
    duplicate = await ledger.reserve("candidate", 2)
    assert first == duplicate
    assert ledger.reserved == 2

    await ledger.reconcile(first, 1.25)
    assert ledger.spent == 1.25
    assert ledger.reserved == 0
    assert ledger.remaining == 3.75


@pytest.mark.asyncio
async def test_budget_rejects_work_before_limit_is_exceeded():
    ledger = BudgetLedger(1)
    with pytest.raises(BudgetExceeded, match="exceeds"):
        await ledger.reserve("too-large", 1.01)
