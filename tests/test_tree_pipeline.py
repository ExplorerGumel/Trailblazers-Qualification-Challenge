import numpy as np
import pandas as pd
import pytest

from src.tree_pipeline import TreePreprocessor, preprocess_tree_data


def test_preprocessor_fits_training_statistics_only():
    train = pd.DataFrame(
        {"numeric": [1.0, np.nan, 3.0], "category": ["a", "b", None]}
    )
    validation = pd.DataFrame(
        {"numeric": [np.nan], "category": ["unseen"]}
    )

    preprocessor = TreePreprocessor().fit(train)
    transformed = preprocessor.transform(validation)

    assert transformed.loc[0, "numeric"] == 2.0
    assert transformed.loc[0, "category"] == -1
    assert transformed.columns.tolist() == ["numeric", "category"]


def test_preprocessor_rejects_missing_required_columns():
    preprocessor = TreePreprocessor().fit(pd.DataFrame({"numeric": [1.0, 2.0]}))

    with pytest.raises(ValueError, match="Missing required feature columns"):
        preprocessor.transform(pd.DataFrame({"different": [1.0]}))


def test_preprocess_tree_data_rejects_invalid_target():
    df = pd.DataFrame({"feature": [1, 2], "target": [1.0, np.nan]})

    with pytest.raises(ValueError, match="numeric, non-null"):
        preprocess_tree_data(df)
