from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.85, min_df=3), "text"),
            ("onehot", OneHotEncoder(handle_unknown="ignore"), ["month", "day"]),
            ("extra", "passthrough", ["title_length", "has_num", "year"]),
        ]
    )

    classifier = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=2,
        max_features="log2",
        random_state=2,
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("gbc", classifier),
    ])
