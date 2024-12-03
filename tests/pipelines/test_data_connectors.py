import pytest
import time
import random


@pytest.mark.parametrize("input_data", ["a", "b", "c", "d"] * 5)
def test_model_training_pipeline(input_data, wait_time_seconds) -> None:
    assert False

    time.sleep(random.uniform(0.1, wait_time_seconds))  # Simulate pipeline execution time
