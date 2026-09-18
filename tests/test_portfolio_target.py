"""Writing the portfolio target to Prime, alongside the NATS broadcast.

The load-bearing test here is `TestTheNatsBroadcastIsUnchanged`. Other
deployments still run the OMS that subscribes to that subject and reads schema
1, so this change is only safe if the old path is byte-identical. Everything
else is about the new one.
"""

from __future__ import annotations

import json
from datetime import timedelta

import httpx
import pytest
from aion import Trigger

from adrs.execution.executor import _declared_gap_ns


class TestTheDeclaredDeadline:
    """`next_expected_at`, so a consumer knows when to stop believing a target.

    The consumer multiplies this gap by three before calling a publisher dead.
    Getting it wrong in either direction is bad: too long and a dead publisher
    goes unnoticed, too short and every ordinary pause reads as an outage.
    """

    def test_a_recurring_cron_states_its_gap(self) -> None:
        assert _declared_gap_ns(Trigger.Cron("*/4 * * * *")) == 240_000_000_000

    def test_an_interval_trigger_states_its_own(self) -> None:
        gap = _declared_gap_ns(Trigger.Interval(timedelta(minutes=5)))

        assert gap == 300_000_000_000

    def test_a_cron_it_cannot_read_yields_nothing(self) -> None:
        # Not a failure. Schema 2 permits a null deadline and the consumer then
        # infers cadence from observed gaps -- correct, just less sharp. The
        # alternative was a cron-parsing dependency inside a live trading
        # process, to serve a field that already degrades gracefully.
        assert _declared_gap_ns(Trigger.Cron("15 3 * * 1")) is None

    def test_something_that_is_not_a_trigger_yields_nothing(self) -> None:
        assert _declared_gap_ns(object()) is None


class Recorder:
    """Stands in for the NATS metric stream, keeping what was published."""

    def __init__(self) -> None:
        self.published: list[tuple[str, bytes]] = []

    async def publish(self, subject: str, payload: bytes, **_kwargs: object) -> None:
        self.published.append((subject, payload))


class FakeMetricBuilder:
    def __init__(self, stream: Recorder) -> None:
        self.metric_stream = stream
        self.signals: list[dict] = []
        self.alerts: list[str] = []

    async def create_portfolio_signal(self, portfolio_id: str, signals: dict) -> None:
        self.signals.append(dict(signals))

    async def create_portfolio_alert(self, **kwargs: object) -> None:
        self.alerts.append(str(kwargs.get("description")))


def aggregating_executor(weights: dict[str, float], **attrs: object):
    """A `PortfolioExecutor` wired just far enough to run `on_aggregate`.

    Constructed past `__init__` because a real one needs a Portfolio with a
    populated, non-stale signal frame, and none of that is what these tests are
    about. Everything `on_aggregate` actually touches is supplied.
    """
    import polars as pl

    from adrs.execution.executor import PortfolioExecutor

    executor = PortfolioExecutor.__new__(PortfolioExecutor)
    recorder = Recorder()
    executor.metric_builder = FakeMetricBuilder(recorder)
    executor.signal_namespace = "aegis_ts"
    executor.aggregate_window = Trigger.Cron("*/4 * * * *")
    executor.prime_url = None
    executor.prime_api_key = None
    executor._prime_epoch_ns = 1_758_067_200_000_000_000
    executor._prime_sequence = 0

    frame = pl.DataFrame(
        {
            "base_asset": list(weights),
            "weighted_signal": list(weights.values()),
        }
    )

    class Portfolio:
        id = "bqp_test"

        def get_signal(self):  # noqa: ANN201
            return frame

    executor.portfolio = Portfolio()
    for name, value in attrs.items():
        setattr(executor, name, value)
    return executor, recorder


class TestTheNatsBroadcastIsUnchanged:
    """The reason this change is safe, exercised rather than described.

    Other deployments subscribe to this subject with an OMS that reads schema 1,
    where an omitted asset keeps its position. If anything here altered that
    payload -- its shape, its rounding, its subject -- those deployments would
    change behaviour without anyone asking them to.

    These run the real `on_aggregate` and read what it actually published.
    """

    def broadcast(self, weights: dict[str, float], **attrs: object) -> tuple:
        import asyncio

        executor, recorder = aggregating_executor(weights, **attrs)
        asyncio.run(executor.on_aggregate())
        assert recorder.published, "on_aggregate broadcast nothing"
        subject, payload = recorder.published[0]
        return subject, json.loads(payload)

    def test_the_payload_is_still_schema_1(self) -> None:
        # Exactly two keys. The old OMS parses `assets` and `timestamp` and
        # nothing else, and a schema 2 envelope on this subject would be read
        # by it as a partial target.
        _subject, body = self.broadcast({"BTC": 0.5})

        assert set(body) == {"assets", "timestamp"}

    def test_weights_are_still_two_places(self) -> None:
        # Four places go to Prime and two stay here, deliberately: widening
        # this rendering would change what the existing OMS receives.
        _subject, body = self.broadcast({"BTC": 0.123456})

        assert body["assets"]["BTC"] == "0.12"

    def test_the_subject_is_unchanged(self) -> None:
        subject, _body = self.broadcast({"BTC": 0.5})

        assert subject == "portfolio_signal.aegis_ts.bqp_test"

    def test_it_still_broadcasts_when_prime_is_unreachable(self) -> None:
        # The ordering guarantee. The broadcast goes out first and the Prime
        # write cannot undo it, so a Prime outage costs the new consumer its
        # target and costs the old one nothing.
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        original = httpx.AsyncClient
        httpx.AsyncClient = lambda *a, **k: original(  # type: ignore[misc]
            transport=httpx.MockTransport(refuse)
        )
        try:
            _subject, body = self.broadcast(
                {"BTC": 0.5},
                prime_url="https://prime.example",
                prime_api_key="k",
            )
        finally:
            httpx.AsyncClient = original  # type: ignore[misc]

        assert body["assets"]["BTC"] == "0.50"

    def test_a_prime_failure_files_no_portfolio_alert(self) -> None:
        # `on_aggregate`'s handler files an alert on any exception. If the
        # Prime write could raise, a Prime outage would file one every four
        # minutes saying the portfolio failed, which it did not.
        import asyncio

        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        executor, _recorder = aggregating_executor(
            {"BTC": 0.5}, prime_url="https://prime.example", prime_api_key="k"
        )
        original = httpx.AsyncClient
        httpx.AsyncClient = lambda *a, **k: original(  # type: ignore[misc]
            transport=httpx.MockTransport(refuse)
        )
        try:
            asyncio.run(executor.on_aggregate())
        finally:
            httpx.AsyncClient = original  # type: ignore[misc]

        assert executor.metric_builder.alerts == []

    def test_the_aegis_metric_still_goes_out(self) -> None:
        # The dashboard's per-asset rows, untouched by any of this.
        import asyncio

        executor, _recorder = aggregating_executor({"BTC": 0.5})
        asyncio.run(executor.on_aggregate())

        assert executor.metric_builder.signals == [{"BTC": 0.5}]


class TestThePrimeWriteIsOptional:
    """Configured off by default, so an existing deployment is untouched."""

    def test_no_url_means_no_call(self) -> None:
        from adrs.execution.executor import PortfolioExecutor

        # Reaching past __init__ deliberately: constructing a real
        # PortfolioExecutor needs a Portfolio with a populated signal frame, and
        # what is under test is one guard.
        executor = PortfolioExecutor.__new__(PortfolioExecutor)
        executor.prime_url = None
        executor.prime_api_key = "key"

        import asyncio

        asyncio.run(executor._publish_target_to_prime({"BTC": 0.5}))

    def test_no_key_means_no_call(self) -> None:
        from adrs.execution.executor import PortfolioExecutor

        executor = PortfolioExecutor.__new__(PortfolioExecutor)
        executor.prime_url = "https://prime.example"
        executor.prime_api_key = None

        import asyncio

        asyncio.run(executor._publish_target_to_prime({"BTC": 0.5}))


class TestWhatReachesPrime:
    def executor(self, handler, **attrs):
        import asyncio

        from adrs.execution.executor import PortfolioExecutor

        executor = PortfolioExecutor.__new__(PortfolioExecutor)
        executor.prime_url = "https://prime.example"
        executor.prime_api_key = "not-a-real-key"
        executor._prime_epoch_ns = 1_758_067_200_000_000_000
        executor._prime_sequence = 0
        executor.aggregate_window = Trigger.Cron("*/4 * * * *")

        class Portfolio:
            id = "bqp_test"

        executor.portfolio = Portfolio()
        for name, value in attrs.items():
            setattr(executor, name, value)

        seen: list[httpx.Request] = []

        def record(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return handler(request)

        original = httpx.AsyncClient

        def patched(*args: object, **kwargs: object) -> httpx.AsyncClient:
            return original(transport=httpx.MockTransport(record))

        httpx.AsyncClient = patched  # type: ignore[misc]
        try:
            asyncio.run(executor._publish_target_to_prime({"BTC": 0.5, "ETH": -0.25}))
        finally:
            httpx.AsyncClient = original  # type: ignore[misc]
        return seen

    def ok(self, _request: httpx.Request) -> httpx.Response:
        return httpx.Response(202)

    def body(self, seen: list[httpx.Request]) -> dict:
        assert seen, "nothing was sent to Prime"
        return json.loads(seen[0].content)

    def test_it_is_schema_2(self) -> None:
        body = self.body(self.executor(self.ok))

        assert body["schema"] == 2
        assert body["portfolio_id"] == "bqp_test"

    def test_the_sequence_advances(self) -> None:
        # Without it, every target would carry the same ordering key and a
        # consumer would apply the first and discard the rest as redelivery.
        body = self.body(self.executor(self.ok))

        assert body["sequence"] == 1

    def test_weights_are_four_places(self) -> None:
        body = self.body(self.executor(self.ok))

        assert body["assets"] == {"BTC": "0.5000", "ETH": "-0.2500"}

    def test_a_zero_weight_is_sent_rather_than_dropped(self) -> None:
        # Schema 2's meaning depends on it: an absent asset is an instruction
        # to go flat, so an asset held at zero must be present and zero, not
        # missing. `Portfolio.get_signal` already returns every asset in the
        # roster; this proves nothing filters them out on the way.
        import asyncio

        from adrs.execution.executor import PortfolioExecutor

        executor = PortfolioExecutor.__new__(PortfolioExecutor)
        executor.prime_url = "https://prime.example"
        executor.prime_api_key = "k"
        executor._prime_epoch_ns = 1
        executor._prime_sequence = 0
        executor.aggregate_window = Trigger.Cron("*/4 * * * *")

        class Portfolio:
            id = "p"

        executor.portfolio = Portfolio()
        seen: list[httpx.Request] = []
        original = httpx.AsyncClient
        httpx.AsyncClient = lambda *a, **k: original(  # type: ignore[misc]
            transport=httpx.MockTransport(
                lambda r: (seen.append(r), httpx.Response(202))[1]
            )
        )
        try:
            asyncio.run(executor._publish_target_to_prime({"BTC": 0.0, "ETH": 0.3}))
        finally:
            httpx.AsyncClient = original  # type: ignore[misc]

        assert json.loads(seen[0].content)["assets"]["BTC"] == "0.0000"

    def test_the_deadline_comes_from_the_schedule(self) -> None:
        body = self.body(self.executor(self.ok))

        assert body["next_expected_at"] - body["published_at"] == 240_000_000_000

    def test_the_key_is_sent(self) -> None:
        seen = self.executor(self.ok)

        assert seen[0].headers["x-api-key"] == "not-a-real-key"

    def test_it_posts_to_the_portfolio_s_own_path(self) -> None:
        seen = self.executor(self.ok)

        assert str(seen[0].url).endswith("/api/portfolio-target/bqp_test")


class TestItCannotDisturbTheBroadcast:
    """A Prime failure must not raise.

    `on_aggregate` wraps everything in one handler that logs and files a
    portfolio alert. If this raised, a Prime outage would become an alert every
    four minutes -- and the alert would say the portfolio failed, which it did
    not. The broadcast has already gone out by the time this runs.
    """

    def run_against(self, handler) -> None:
        import asyncio

        from adrs.execution.executor import PortfolioExecutor

        executor = PortfolioExecutor.__new__(PortfolioExecutor)
        executor.prime_url = "https://prime.example"
        executor.prime_api_key = "k"
        executor._prime_epoch_ns = 1
        executor._prime_sequence = 0
        executor.aggregate_window = Trigger.Cron("*/4 * * * *")

        class Portfolio:
            id = "p"

        executor.portfolio = Portfolio()
        original = httpx.AsyncClient
        httpx.AsyncClient = lambda *a, **k: original(  # type: ignore[misc]
            transport=httpx.MockTransport(handler)
        )
        try:
            asyncio.run(executor._publish_target_to_prime({"BTC": 0.5}))
        finally:
            httpx.AsyncClient = original  # type: ignore[misc]

    @pytest.mark.parametrize("status", [400, 401, 404, 500, 503])
    def test_a_rejection_is_survived(self, status: int) -> None:
        self.run_against(lambda _r: httpx.Response(status))

    def test_a_connection_error_is_survived(self) -> None:
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        self.run_against(refuse)

    def test_a_nonsense_response_is_survived(self) -> None:
        self.run_against(lambda _r: httpx.Response(200, content=b"\xff\xfe not json"))
