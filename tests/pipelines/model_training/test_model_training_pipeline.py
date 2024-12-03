"""
This is a boilerplate test file for pipeline 'model_training'
generated using Kedro 0.19.9.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

import pytest
import time
import random

@pytest.mark.parametrize(
    "input_data",
    ["a", "b", "c", "d"] * 5
)
def test_model_training_pipeline(input_data) -> None:
    assert True

    time.sleep(random.uniform(.1, .5))  # Simulate pipeline execution time
