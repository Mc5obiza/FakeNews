import pandas as pd


def _contains_number(text: str) -> int:
    for char in str(text):
        if char.isdigit():
            return 1
    return 0


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = pd.DataFrame(
        {
            "text": (
                df["title"].astype(str)
                + " "
                + df["text"].astype(str)
                + " "
                + df["subject"].astype(str)
            ),
            "year": df["date"].dt.year,
            "month": df["date"].dt.month,
            "day": df["date"].dt.day,
            "title_length": df["title"].astype(str).str.len(),
            "has_num": df["text"].astype(str).apply(_contains_number),
        }
    ).dropna()

    labels = df.loc[features.index, "isReal"].astype(int)
    return features, labels
