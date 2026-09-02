import pytest
import pandas as pd
import numpy as np
from src.data.nasa_loader import NASADataLoader
from src.data.calce_loader import CALCEDataLoader
from src.data.validator import DataQualityValidator
from src.features.extractor import FeatureExtractor
from src.data.unified_dataset import prepare_temporal_splits

def test_nasa_loader():
    loader = NASADataLoader()
    df = loader.load_cell("B0005")
    assert not df.empty
    assert "capacity" in df.columns
    assert "cycle" in df.columns
    assert len(df) == 168

def test_calce_loader():
    loader = CALCEDataLoader()
    df = loader.load_cell("CS2_35")
    assert not df.empty
    assert "capacity" in df.columns
    assert len(df) == 650

def test_data_validator():
    loader = NASADataLoader()
    df = loader.load_cell("B0005")
    validator = DataQualityValidator()
    res = validator.validate_dataframe(df, "Test_NASA")
    assert res["status"] in ["PASS", "WARNING"]
    assert res["total_records"] == 168

def test_feature_extractor():
    loader = NASADataLoader()
    df = loader.load_cell("B0005")
    extractor = FeatureExtractor()
    feat_df = extractor.extract_features(df)
    assert "capacity_prev" in feat_df.columns
    assert "temp_rise" in feat_df.columns
    assert len(feat_df) == len(df)

def test_temporal_splits_no_leakage():
    loader = NASADataLoader()
    df = loader.load_cell("B0005")
    extractor = FeatureExtractor()
    feat_df = extractor.extract_features(df)
    splits = prepare_temporal_splits(feat_df, FeatureExtractor.DEFAULT_FEATURE_COLS)
    assert "train" in splits and "test" in splits and "scaler" in splits
    assert splits["scaler"].is_fitted
