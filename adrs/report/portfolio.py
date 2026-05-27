import io
import polars as pl
from typing import Self
from datetime import datetime
from pydantic import BaseModel
from pandera.typing.polars import DataFrame

from adrs import json_utils as json
from adrs.execution.portfolio import (
    Portfolio,
    PortfolioPerformance,
)


class PortfolioReportV1Performance(BaseModel):
    performance: PortfolioPerformance


class PortfolioReportV1(BaseModel):
    portfolio_id: str
    # long_alphas: int
    # short_alphas: int
    # both_alphas: int
    back: PortfolioReportV1Performance
    forward: PortfolioReportV1Performance

    @staticmethod
    def version() -> str:
        return "portfolio-report-v1"

    @classmethod
    def compute(
        cls,
        id: str,
        signal_df: pl.DataFrame,
        metadata_df: pl.DataFrame,
        weight_df: pl.DataFrame,
        prices_df: pl.DataFrame,
        back_pct: float = 0.7,
    ) -> Self:
        timestamps = signal_df["start_time"].sort()
        n = len(timestamps)
        split_idx = int(n * back_pct)
        split_time = timestamps[split_idx]

        def make_portfolio(
            lf: pl.DataFrame, pf: pl.DataFrame
        ) -> tuple[Portfolio, pl.DataFrame]:
            return Portfolio(
                id=id,
                signal_df=lf,
                metadata_df=metadata_df,
                weight_df=weight_df,
            ), pf

        back_portfolio, back_prices = make_portfolio(
            signal_df.filter(pl.col("start_time") < split_time),
            prices_df.filter(pl.col("start_time") < split_time),
        )
        forward_portfolio, forward_prices = make_portfolio(
            signal_df.filter(pl.col("start_time") >= split_time),
            prices_df.filter(pl.col("start_time") >= split_time),
        )

        return cls(
            portfolio_id=id,
            back=PortfolioReportV1Performance(
                performance=back_portfolio.backtest(back_prices)
            ),
            forward=PortfolioReportV1Performance(
                performance=forward_portfolio.backtest(
                    forward_prices
                )
            ),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PortfolioReportV1):
            return False
        return (
            self.portfolio_id == other.portfolio_id
            # and self.long_alphas == other.long_alphas
            # and self.short_alphas == other.short_alphas
            # and self.both_alphas == other.both_alphas
            and self.back.model_dump(exclude={"performance_df"})
            == other.back.model_dump(exclude={"performance_df"})
            and self.back.performance_df.equals(other.back.performance_df)
            and self.forward.model_dump(exclude={"performance_df"})
            == other.forward.model_dump(exclude={"performance_df"})
            and self.forward.performance_df.equals(other.forward.performance_df)
        )

    def serialize(self) -> bytes:
        back_pdf_buf = io.BytesIO()
        self.back.performance_df.write_parquet(back_pdf_buf)
        forward_pdf_buf = io.BytesIO()
        self.forward.performance_df.write_parquet(forward_pdf_buf)

        buf = io.BytesIO()
        df = pl.DataFrame(
            {
                "portfolio_id": [self.portfolio_id],
                # "long_alphas": [self.long_alphas],
                # "short_alphas": [self.short_alphas],
                # "both_alphas": [self.both_alphas],
                "back": [self.back.model_dump_json(exclude={"performance_df"})],
                "back_pdf_buf": [back_pdf_buf.getvalue()],
                "forward": [self.forward.model_dump_json(exclude={"performance_df"})],
                "forward_pdf_buf": [forward_pdf_buf.getvalue()],
            }
        )
        df.write_parquet(buf)

        return buf.getvalue()

    @classmethod
    def deserialize(cls, buf: bytes) -> Self:
        df = pl.read_parquet(buf)

        back_pdf = pl.read_parquet(df["back_pdf_buf"][0])
        forward_pdf = pl.read_parquet(df["forward_pdf_buf"][0])

        return cls(
            portfolio_id=df["portfolio_id"][0],
            long_alphas=df["long_alphas"][0],
            short_alphas=df["short_alphas"][0],
            both_alphas=df["both_alphas"][0],
            back=PortfolioReportV1Performance(
                **json.loads(df["back"][0]),
                performance_df=PortfolioPerformanceDF.validate(back_pdf),
            ),
            forward=PortfolioReportV1Performance(
                **json.loads(df["forward"][0]),
                performance_df=PortfolioPerformanceDF.validate(forward_pdf),
            ),
        )
