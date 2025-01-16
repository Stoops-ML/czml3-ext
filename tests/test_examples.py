import pathlib
import subprocess
import sys


def test_examples():
    dir_examples = pathlib.Path(__file__).parent.parent / "examples"
    assert dir_examples.exists()
    for f in dir_examples.iterdir():
        if f.suffix != ".py":
            continue
        result = subprocess.run([sys.executable, f], capture_output=True, text=True)
        assert result.returncode == 0
