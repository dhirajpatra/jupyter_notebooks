from utils2 import str_to_int


# This will test the str_to_int function
def test_str_to_int():
    result = str_to_int("14")
    assert result == 14


def test_str_to_int_fails():
    result = str_to_int("14, 44")
    assert result == 14


def test_str_to_int_decimals():
    result = str_to_int("14.44")
    assert result == 14


def test_str_to_int_with_integers():
    result = str_to_int(10)
    assert result == 10
