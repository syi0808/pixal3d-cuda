import io
import json
import struct

import pytest
import torch

from scripts.verify_checkpoints import remote_shapes, verify_shapes


def test_remote_safetensors_metadata_is_bounded_and_has_no_weight_read(monkeypatch):
    header = json.dumps({"weight": {"shape": [2, 3]}, "bias": {"shape": [2]}, "__metadata__": {}}).encode()
    requests = []
    def open_header(request, timeout):
        requests.append(request)
        return io.BytesIO(struct.pack("<Q", len(header)) + header)
    monkeypatch.setattr("urllib.request.urlopen", open_header)
    actual = remote_shapes("pinned-revision", "model")
    assert requests[0].get_header("Range") == "bytes=0-1048575"
    assert "pinned-revision" in requests[0].full_url
    result = verify_shapes(lambda: torch.nn.Linear(3, 2), actual, "test")
    assert result["matched_tensors"] == 2
    with pytest.raises(ValueError, match="shape_mismatch"):
        verify_shapes(lambda: torch.nn.Linear(4, 2), actual, "test")
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(struct.pack("<Q", 1048576)))
    with pytest.raises(ValueError, match="1 MiB"):
        remote_shapes("pinned-revision", "model")
