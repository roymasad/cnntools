import pytest
from unittest.mock import patch
import main

def test_quit_app():
    """
    Test quitting the App.
    """
    print("Running test: Quitting the App.")
    
    # Mock user input to return '3' for quitting the app
    with patch('builtins.input', return_value='3'):
        with pytest.raises(SystemExit) as e:
            main.main_loop()  # Assuming the while loop is in a function called main_loop
        assert e.type == SystemExit
        assert e.value.code == 0  # A SystemExit with code 0 is a successful exit