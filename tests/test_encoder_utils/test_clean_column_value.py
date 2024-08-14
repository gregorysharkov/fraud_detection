import pandas as pd
from src.fraud_detection.utils.encoder_utils import clean_column_value

def test_clean_column_value_single_string():
    input_value = "Fraud_Transaction 123"
    expected_output = ["transaction", "123"]
    assert clean_column_value(input_value) == expected_output

def test_clean_column_value_series():
    input_value = pd.Series(["Fraud_Transaction_123", "Another_Value"])
    expected_output = pd.Series([["transaction", "123"], ["another", "value"]])
    pd.testing.assert_series_equal(clean_column_value(input_value), expected_output)

def test_clean_column_value_special_characters():
    input_value = "Fraud_Transaction!@#$%^&*()"
    expected_output = ["transaction"]
    assert clean_column_value(input_value) == expected_output

def test_clean_column_value_multiple_spaces():
    input_value = "Fraud_Transaction   123"
    expected_output = ["transaction", "123"]
    assert clean_column_value(input_value) == expected_output

def test_clean_column_value_empty_string():
    input_value = ""
    expected_output = []
    assert clean_column_value(input_value) == expected_output