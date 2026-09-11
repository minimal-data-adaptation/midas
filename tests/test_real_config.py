import json

import pytest

from midas.real.config import RealRunSpec


def _spec(**overrides):
    values = {
        "resize_image": 8,
        "chunk_len": 4,
        "query_freq": 2,
        "use_vlm_embedding": False,
        "midas_use_trust_region": False,
        "actor_kwargs": {"hidden_dims": [8, 8]},
    }
    values.update(overrides)
    return RealRunSpec(**values)


def test_real_run_spec_round_trip_and_tamper_detection(tmp_path):
    path = tmp_path / "real_run_spec.json"
    original = _spec()
    original.write(path)
    restored = RealRunSpec.read(path)
    assert restored == original
    assert restored.actor_signature == original.actor_signature

    contents = json.loads(path.read_text())
    contents["query_freq"] = 1
    path.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="hash"):
        RealRunSpec.read(path)


def test_real_run_spec_requires_explicit_trust_cap():
    with pytest.raises(ValueError, match="one normalized radius"):
        _spec(midas_use_trust_region=True)
