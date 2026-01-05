import io
import inspect
import polars as pl
from datetime import datetime, timedelta
from pydantic import BaseModel, TypeAdapter
from typing import Self, cast, Unpack, NotRequired, TypedDict

from adrs import json
from adrs.alpha import Alpha
from adrs.data import Datamap
from adrs.types import Performance
from adrs.performance import Evaluator
from adrs.tests import Sensitivity, SensitivityParameter

type Params = dict[str, float | int | timedelta]


class SensitivitySharpeRatioSummary(BaseModel):
    best_param: float
    mean: float
    median: float
    std: float
    min: float
    max: float
    p25: float
    p75: float
    num_negative: int
    num_positive: int
    total_permutations: int
    score: float

    @classmethod
    def compute(
        cls, performance: Performance, results: list[tuple[Params, Performance]]
    ) -> Self:
        df = pl.DataFrame(map(lambda t: t[1], results))

        best_param = performance.sharpe_ratio
        mean = cast(float, df["sharpe_ratio"].mean() or 0.0)
        std = cast(float, df["sharpe_ratio"].std() or 0.0)
        num_positive = df.filter(pl.col("sharpe_ratio") > 0).shape[0]
        total_permutations = df.shape[0]
        # fmt: off
        score = (
            ((1 / std) if std != 0 else 0) * 0.4
            + ((mean / best_param) if best_param != 0 else 0) * 0.3
            + ((num_positive / total_permutations) if total_permutations != 0 else 0) * 0.3
        )
        # fmt: on

        return cls(
            best_param=best_param,
            mean=mean,
            median=cast(float, df["sharpe_ratio"].median() or 0.0),
            std=std,
            min=cast(float, df["sharpe_ratio"].min() or 0.0),
            max=cast(float, df["sharpe_ratio"].max() or 0.0),
            p25=df["sharpe_ratio"].quantile(0.25) or 0.0,
            p75=df["sharpe_ratio"].quantile(0.75) or 0.0,
            num_negative=df.filter(pl.col("sharpe_ratio") < 0).shape[0],
            num_positive=num_positive,
            total_permutations=total_permutations,
            score=score,
        )


class AlphaReportV1Performance(BaseModel):
    performance: Performance
    performance_df: pl.DataFrame
    sensitivity: list[tuple[Params, Performance]]
    sensitivity_sr_summary: SensitivitySharpeRatioSummary

    class Config:
        arbitrary_types_allowed = True


class AlphaBacktestArgsWithoutDates(TypedDict):
    evaluator: Evaluator
    base_asset: str
    datamap: Datamap
    fees: float
    data_df: NotRequired[pl.DataFrame]
    price_shift: NotRequired[int]
    output_columns: NotRequired[list[pl.Expr]]


class AlphaReportV1(BaseModel):
    alpha_id: str
    params: Params
    back: AlphaReportV1Performance
    forward: AlphaReportV1Performance
    sensitivity_params: dict[str, SensitivityParameter]

    @staticmethod
    def version() -> str:
        return "alpha-report-v1"

    @classmethod
    def compute(
        cls,
        alpha: Alpha,
        B_start: datetime,
        B_end: datetime,
        F_start: datetime,
        F_end: datetime,
        sensitivity: Sensitivity,
        **kwargs: Unpack[AlphaBacktestArgsWithoutDates],
    ) -> Self:
        # get alpha params
        params = {
            name: getattr(alpha, name)
            for name in inspect.signature(alpha.__init__).parameters.keys()
        }

        # run backtest
        back, back_df = alpha.backtest(start_time=B_start, end_time=B_end, **kwargs)
        back_sensitivity = [
            (x[0], x[1])
            for x in sensitivity.test(start_time=B_start, end_time=B_end, **kwargs)
        ]

        # run forward test
        forward, forward_df = alpha.backtest(
            start_time=F_start, end_time=F_end, **kwargs
        )
        forward_sensitivity = [
            (x[0], x[1])
            for x in sensitivity.test(start_time=F_start, end_time=F_end, **kwargs)
        ]

        return cls(
            alpha_id=alpha.id,
            params=params,
            sensitivity_params=sensitivity.sensitivity_parameters,
            back=AlphaReportV1Performance(
                performance=back,
                performance_df=back_df,
                sensitivity=back_sensitivity,
                sensitivity_sr_summary=SensitivitySharpeRatioSummary.compute(
                    back, back_sensitivity
                ),
            ),
            forward=AlphaReportV1Performance(
                performance=forward,
                performance_df=forward_df,
                sensitivity=forward_sensitivity,
                sensitivity_sr_summary=SensitivitySharpeRatioSummary.compute(
                    forward, forward_sensitivity
                ),
            ),
        )

    def write_parquet(self, path: str):
        """Write the report to a parquet file."""
        with open(path, "wb+") as f:
            f.write(self.serialize())

    def serialize(self) -> bytes:
        back_pdf_buf = io.BytesIO()
        self.back.performance_df.write_parquet(back_pdf_buf)
        forward_pdf_buf = io.BytesIO()
        self.forward.performance_df.write_parquet(forward_pdf_buf)

        buf = io.BytesIO()
        df = pl.DataFrame(
            {
                "alpha_id": [self.alpha_id],
                "params": [json.dumps(self.params)],
                "sensitivity_params": [
                    TypeAdapter(dict[str, SensitivityParameter]).dump_json(
                        self.sensitivity_params
                    )
                ],
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
            alpha_id=df["alpha_id"][0],
            params=json.loads(df["params"][0]),
            sensitivity_params=TypeAdapter(
                dict[str, SensitivityParameter]
            ).validate_json(df["sensitivity_params"][0]),
            back=AlphaReportV1Performance(
                **json.loads(df["back"][0]),
                performance_df=back_pdf,
            ),
            forward=AlphaReportV1Performance(
                **json.loads(df["forward"][0]),
                performance_df=forward_pdf,
            ),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AlphaReportV1):
            return False
        return (
            self.alpha_id == other.alpha_id
            and self.params == other.params
            and self.sensitivity_params == other.sensitivity_params
            and self.back.performance == other.back.performance
            and self.back.sensitivity == other.back.sensitivity
            and self.back.sensitivity_sr_summary == other.back.sensitivity_sr_summary
            and self.back.performance_df.equals(other.back.performance_df)
            and self.forward.performance == other.forward.performance
            and self.forward.sensitivity == other.forward.sensitivity
            and self.forward.sensitivity_sr_summary
            == other.forward.sensitivity_sr_summary
            and self.forward.performance_df.equals(other.forward.performance_df)
        )
