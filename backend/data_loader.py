import re
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
FAKE_CSV = DATA_DIR / "Fake.csv"
TRUE_CSV = DATA_DIR / "True.csv"

_MONTH_MAP = {
    "Jan": "January",
    "Feb": "February",
    "Mar": "March",
    "Apr": "April",
    "May": "May",
    "Jun": "June",
    "Jul": "July",
    "Aug": "August",
    "Sep": "September",
    "Oct": "October",
    "Nov": "November",
    "Dec": "December",
}
_MONTH_PATTERN = re.compile(r"\\b(" + "|".join(_MONTH_MAP.keys()) + r")\\b")


def _normalize_month_names(date_series: pd.Series) -> pd.Series:
    def _replace(match: re.Match) -> str:
        return _MONTH_MAP[match.group(0)]

    return date_series.fillna("").str.replace(_MONTH_PATTERN, _replace, regex=True)


def load_dataset() -> pd.DataFrame:
    fake_df = pd.read_csv(FAKE_CSV)
    fake_df["isReal"] = 0

    true_df = pd.read_csv(TRUE_CSV)
    true_df["isReal"] = 1

    df = pd.concat([fake_df, true_df], axis=0)
    df = df.sample(frac=1, random_state=1808).reset_index(drop=True)

    normalized_date = _normalize_month_names(df["date"]).str.strip()
    df["date"] = pd.to_datetime(normalized_date, format="%B %d, %Y", errors="coerce")

    df = df.dropna(subset=["date", "title", "text", "subject"])
    return df
