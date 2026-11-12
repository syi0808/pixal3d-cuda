import ast
from pathlib import Path
import subprocess
import sys

import nbformat
import pytest


@pytest.mark.parametrize("name", ["colab_t4_512.ipynb", "colab_t4_multiview_512.ipynb", "colab_t4_performance.ipynb"])
def test_colab_notebook_schema_and_python_cells(name):
    path = Path(__file__).resolve().parents[1] / "notebooks" / name
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert cell.execution_count is None
            assert cell.outputs == []


@pytest.mark.parametrize("name", ["colab_t4_512.ipynb", "colab_t4_multiview_512.ipynb"])
def test_notebook_uses_official_weights_and_no_embedded_credentials(name):
    path = Path(__file__).resolve().parents[1] / "notebooks" / name
    notebook = nbformat.read(path, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert 'userdata.get("HF_TOKEN")' in source
    assert "torch==2.6.0" in source
    assert "cu124" in source
    assert "hf_" not in source


def test_notebook_streams_child_stdout_and_stderr(capsys):
    path = Path(__file__).resolve().parents[1] / "notebooks" / "colab_t4_512.ipynb"
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


def test_operator_notebook_needs_no_model_downloads_or_authentication():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "colab_t4_performance.ipynb"
    notebook = nbformat.read(path, as_version=4)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert "benchmark_cuda_ops.py" in code and '"--device", "cuda"' in code
    assert "get_device_capability() == (7, 5)" in code
    assert "HF_TOKEN" not in code and "userdata" not in code
    assert "--device cpu" not in code
