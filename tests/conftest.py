import pytest
from pathlib import Path

@pytest.fixture
def temp_dataset(tmp_path):
    file = tmp_path / "data.csv"
    file.write_text("x,y\n1,2\n3,4\n")
    return file
