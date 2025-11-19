from pydantic import BaseModel

from cybotrade import Topic


class DataColumn(BaseModel):
    src: str  # source (original column)
    dst: str  # destination (final column)

    def __hash__(self):
        return hash(f"{self.src}-{self.dst}")

    def __repr__(self):
        return f"{self.src}-{self.dst}"

    def __lt__(self, value):
        return hash(self) < hash(value)


class DataInfo(BaseModel):
    topic: str
    columns: list[DataColumn]
    lookback_size: int
    # resample_methods: str
    # resample_interval: str
    # shift_min: int

    def __hash__(self):
        return hash(f"{Topic.from_str(self.topic)}-{sorted(self.columns)}")

    def __eq__(self, value):
        return hash(self) == hash(value)
