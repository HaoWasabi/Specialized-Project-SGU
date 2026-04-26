from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class GoldInfo:
    date: datetime
    gold_close: float
    dxy: float
    oil: float
    sp500: float
    bond_yield: float
    gold_volume: float

    def to_dict(self) -> dict[str, object]:
        return {
            "Date": self.date,
            "Gold_Close": self.gold_close,
            "DXY": self.dxy,
            "Oil": self.oil,
            "SP500": self.sp500,
            "Bond_Yield": self.bond_yield,
            "Gold_Volume": self.gold_volume,
        }