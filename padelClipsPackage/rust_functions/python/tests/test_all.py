import pytest
import rust_functions


def test_sum_as_string():
    assert rust_functions.sum_as_string(1, 1) == "2"
