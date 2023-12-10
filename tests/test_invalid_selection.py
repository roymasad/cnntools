import pytest
from unittest.mock import patch, call
from io import StringIO
from main import main_loop

def test_invalid_selection():
    """
    Test Invalid selection in the Menu loop.
    """
    print("Running test: Invalid selection in the Menu loop.")
    
    # Mock user input to return an invalid selection
    with patch('builtins.input', side_effect=['4', '3']):  # '4' is invalid, '3' to quit
        with patch('sys.stdout', new=StringIO()) as fake_out:
            with pytest.raises(SystemExit) as e:
                main_loop()
            assert "Invalid selection" in fake_out.getvalue()
        assert e.type == SystemExit
        assert e.value.code == 0  # A SystemExit with code 0 is a successful exit