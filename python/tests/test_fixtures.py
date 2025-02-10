import os
from utils import str_to_bool  # Importing a utility function for string-to-boolean conversion


# Function to convert a string to a boolean and write it to a file
def write_integer(string, path):
    """
    Converts a string to a boolean using `str_to_bool` and writes the result to a file.

    Args:
        string (str): The input string to be converted.
        path (str): The file path where the boolean value should be written.

    If conversion fails due to a RuntimeError, it writes '0' instead.
    """
    with open(path, "w") as _f:
        try:
            _f.write(str(str_to_bool(string)))  # Convert string to boolean and write to file
        except RuntimeError:
            _f.write("0")  # Handle conversion failure by writing '0'


"""
Pytest Fixtures Explanation:

A pytest fixture is a way to set up test dependencies before running test functions.
Fixtures help initialize data, resources, or configurations that tests need.

The tests below validate the `write_integer` function using both manual setup and pytest's `tmpdir` fixture.
"""


class TestWriteBooleans:
    """
    Test cases for `write_integer` function using manual setup/teardown.
    """

    def setup(self):
        """
        Setup method to ensure the test file does not exist before running tests.
        This prevents issues caused by leftover data from previous test runs.
        """
        if os.path.exists("/tmp/test_value"):
            os.remove("/tmp/test_value")

    def test_write_Yes(self):
        """
        Test case: Writing 'Yes' should result in 'True' being stored in the file.
        """
        write_integer("Yes", "/tmp/test_value")

        with open("/tmp/test_value", "r") as _f:
            value = _f.read()

        assert value == "True"

    def test_write_n(self):
        """
        Test case: Writing 'n' should result in 'False' being stored in the file.
        """
        write_integer("n", "/tmp/test_value")

        with open("/tmp/test_value", "r") as _f:
            value = _f.read()

        assert value == "False"


class TestFixtures:
    """
    Test cases for `write_integer` function using pytest's `tmpdir` fixture.
    """

    def test_write_Yes(self, tmpdir):
        """
        Test case: Writing 'Yes' should store 'True' in a temporary file.

        - `tmpdir` creates a unique temporary directory for each test.
        - Ensures test isolation without relying on a fixed file path.
        """
        path = str(tmpdir.join("test_value"))  # Create a temporary file path

        write_integer("Yes", path)

        with open(path, "r") as _f:
            value = _f.read()

        assert value == "True"
