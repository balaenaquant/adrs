"""`run_portfolio` builds the `PortfolioExecutor`, so it is the only place a
caller can hand Prime's URL and key to it. They were accepted by the executor
and never forwarded, which left `_publish_target_to_prime` returning early on
every deployment."""

import asyncio
import logging
from typing import Any

import pytest

import adrs.execution.runner as runner
from adrs.execution.executor import PortfolioExecutor


def _capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    class FakeExecutor:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

        async def start(self) -> None:
            return None

    monkeypatch.setattr(runner, "PortfolioExecutor", FakeExecutor)
    return seen


def _run(**kwargs: Any) -> None:
    asyncio.run(
        runner.run_portfolio(
            portfolio=object(),  # type: ignore[arg-type]
            alphas=[],
            dataloader=object(),  # type: ignore[arg-type]
            metric_stream=object(),  # type: ignore[arg-type]
            datasource_stream=object(),  # type: ignore[arg-type]
            run_alphas=False,
            **kwargs,
        )
    )


def test_prime_credentials_reach_the_executor(monkeypatch: pytest.MonkeyPatch):
    seen = _capture(monkeypatch)
    _run(prime_url="https://prime.example", prime_api_key="k")
    assert seen["prime_url"] == "https://prime.example"
    assert seen["prime_api_key"] == "k"


def test_they_default_to_off(monkeypatch: pytest.MonkeyPatch):
    seen = _capture(monkeypatch)
    _run()
    assert seen["prime_url"] is None
    assert seen["prime_api_key"] is None


def _startup_log(caplog: pytest.LogCaptureFixture, **attrs: Any) -> list[str]:
    executor = PortfolioExecutor.__new__(PortfolioExecutor)
    executor.prime_url = attrs.get("prime_url")
    executor.prime_api_key = attrs.get("prime_api_key")

    class Portfolio:
        id = "bqp_test"

    executor.portfolio = Portfolio()  # type: ignore[assignment]
    with caplog.at_level(logging.DEBUG):
        executor._log_prime_target_mode()
    return [r.getMessage() for r in caplog.records if "[prime]" in r.getMessage()]


def test_startup_says_when_the_target_is_written(caplog: pytest.LogCaptureFixture):
    (line,) = _startup_log(
        caplog, prime_url="https://prime.example", prime_api_key="sekrit-XYZ"
    )
    assert "writing portfolio target" in line
    assert "bqp_test" in line
    assert "sekrit-XYZ" not in line  # never the key


def test_startup_says_when_it_is_not(caplog: pytest.LogCaptureFixture):
    (line,) = _startup_log(caplog)
    assert "not written" in line


def test_startup_warns_on_a_half_configured_pair(caplog: pytest.LogCaptureFixture):
    (line,) = _startup_log(caplog, prime_url="https://prime.example")
    assert "NOT written" in line
    assert "prime_api_key" in line
