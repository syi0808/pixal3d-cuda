import ast
from pathlib import Path
import subprocess
import sys

import nbformat
import pytest


def test_native_notebook_schema_python_and_explicit_hardware_gate():
    root = Path(__file__).resolve().parents[1] / "notebooks"
    assert [path.name for path in root.glob("*.ipynb")] == ["colab_native_baseline.ipynb"]
    notebook = nbformat.read(root / "colab_native_baseline.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert cell.execution_count is None and cell.outputs == []
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert 'userdata.get("HF_TOKEN")' in code
    assert "torch==2.6.0" in code and "cu124" in code
    assert 'CUDA_ARCH not in ("8.0", "8.6", "8.9")' in code
    assert "hf_" not in code


def test_notebook_streams_child_stdout_and_stderr(capsys):
    path = Path(__file__).resolve().parents[1] / "notebooks" / "colab_native_baseline.ipynb"
    notebook = nbformat.read(path, as_version=4)
    source = next(cell.source for cell in notebook.cells if cell.cell_type == "code" and "def run_command" in cell.source)
    helper = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "run_command")
    namespace = {"subprocess": subprocess}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), "exec"), namespace)
    namespace["run_command"]([sys.executable, "-c", "import sys; print('child stdout'); print('child stderr', file=sys.stderr)"])
    output = capsys.readouterr().out
    assert "child stdout" in output and "child stderr" in output
    with pytest.raises(subprocess.CalledProcessError) as error:
        namespace["run_command"]([sys.executable, "-c", "raise SystemExit(3)"])
    assert error.value.returncode == 3
