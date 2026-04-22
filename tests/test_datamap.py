import json
import pytest
from datetime import datetime, timezone

from adrs.data.datamap import Datamap
from adrs.data.dataloader import DataLoader
from adrs.data.types import DataInfo, DataColumn


def credentials() -> dict:
    return json.load(open("credentials.json"))


def make_dataloader() -> DataLoader:
    return DataLoader(
        data_dir="data",
        credentials=credentials(),
    )


def make_info(lookback_size: int = 1) -> DataInfo:
    return DataInfo(
        topic="cryptoquant|btc/market-data/price-ohlcv?exchange=binance&market=spot&window=hour",
        columns=[DataColumn(src="close", dst="price")],
        lookback_size=lookback_size,
    )


START = datetime(2025, 1, 1, tzinfo=timezone.utc)
END = datetime(2025, 1, 8, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_datamap_init():
    """Datamap initializes and is_ready after init."""
    dm = Datamap()
    info = make_info(lookback_size=24)
    await dm.init(make_dataloader(), [info], START, END)
    assert dm.is_ready()


@pytest.mark.asyncio
async def test_datamap_get():
    """get() returns a DataFrame with the expected columns."""
    dm = Datamap()
    info = make_info(lookback_size=1)
    await dm.init(make_dataloader(), [info], START, END)
    df = dm.get(info)
    assert "start_time" in df.columns
    assert "price" in df.columns
    assert len(df) > 0


@pytest.mark.asyncio
async def test_datamap_ipc_roundtrip():
    """write_ipc / read_ipc round-trip preserves data."""
    dm = Datamap()
    info = make_info(lookback_size=1)
    await dm.init(make_dataloader(), [info], START, END)

    raw = dm.write_ipc()
    dm2 = Datamap.read_ipc(raw)

    df1 = dm.get(info)
    df2 = dm2.get(info)
    assert df1.equals(df2)
