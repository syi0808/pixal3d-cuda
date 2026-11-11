import ast
from pathlib import Path

import nbformat


def test_colab_notebook_schema_and_python_cells():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "colab_t4_512.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert cell.execution_count is None
            assert cell.outputs == []


def test_notebook_uses_official_weights_and_no_embedded_credentials():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "colab_t4_512.ipynb"
    notebook = nbformat.read(path, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert 'userdata.get("HF_TOKEN")' in source
    assert "torch==2.6.0" in source
    assert "cu124" in source
    assert "hf_" not in source
