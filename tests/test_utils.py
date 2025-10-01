from fractions import Fraction

import pytest

from btcmi.utils import is_number


pytestmark = pytest.mark.smoke

@pytest.mark.parametrize("val", [1, 1.0, Fraction(1, 3)])
def test_is_number_with_real_numbers(val):
    """``is_number`` should return True for real numeric objects."""
    assert is_number(val)

@pytest.mark.parametrize("val", [1 + 2j, 2 + 0j])
def test_is_number_rejects_complex_numbers(val):
    """``is_number`` should return False for complex numbers."""
    assert not is_number(val)


@pytest.mark.parametrize("val", [True, False, "1", None, [], object()])
def test_is_number_with_non_numeric_types(val):
    """``is_number`` should return False for non-numeric objects."""
    assert not is_number(val)
