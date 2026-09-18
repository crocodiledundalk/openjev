import math
import sys
from types import SimpleNamespace

import pytest

from openjev_phase1.core import direct_messages, load_causal_model, softmax, validate_row


ROW = {
    "id": "x",
    "state": "owned evidence",
    "question": "Which answer follows?",
    "options": [
        {"id": "yes", "description": "Yes."},
        {"id": "no", "description": "No."},
    ],
}


def test_direct_prompt_excludes_extra_fields():
    row = dict(ROW, label="yes", provenance={"secret": "do not leak"})
    rendered = str(direct_messages(row))
    assert "owned evidence" in rendered
    assert "secret" not in rendered
    assert "label" not in rendered


def test_softmax_is_finite_and_normalized():
    values = softmax([1000.0, 999.0, -1000.0])
    assert all(math.isfinite(value) for value in values)
    assert sum(values) == pytest.approx(1.0)
    assert values[0] > values[1] > values[2]


def test_duplicate_options_rejected():
    row = dict(ROW, options=[ROW["options"][0], ROW["options"][0]])
    with pytest.raises(ValueError, match="unique"):
        validate_row(row)


def test_structured_json_state_is_supported():
    row = dict(ROW, state={"policy": "Never request passwords", "candidate": ["invoice id"]})
    validate_row(row)
    assert '"policy"' in direct_messages(row)[1]["content"]


def test_nonfinite_structured_state_is_rejected():
    with pytest.raises(ValueError, match="finite JSON-compatible"):
        validate_row(dict(ROW, state={"score": float("nan")}))


class FakeLoadedModel:
    def __init__(self):
        self.moves = []
        self.evaluated = False

    def to(self, device):
        self.moves.append(device)
        return self

    def eval(self):
        self.evaluated = True
        return self


def test_load_causal_model_moves_fp16_model_to_mps(monkeypatch):
    model = FakeLoadedModel()
    calls = {}

    class FakeModelClass:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.update(source=source, kwargs=kwargs)
            return model, {"missing_keys": [], "mismatched_keys": [], "error_msgs": []}

    fake_torch = SimpleNamespace(
        __version__="2.10.0",
        bfloat16="bfloat16",
        float16="float16",
        float32="float32",
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    fake_transformers = SimpleNamespace(
        __version__="5.17.0",
        AutoConfig=SimpleNamespace(from_pretrained=lambda *args, **kwargs: SimpleNamespace(model_type="qwen3")),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: "tokenizer"),
        AutoModelForCausalLM=FakeModelClass,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    loaded, tokenizer, metadata = load_causal_model("Qwen/Qwen3-0.6B", "a" * 40, "mps")

    assert loaded is model
    assert tokenizer == "tokenizer"
    assert calls["kwargs"]["dtype"] == "float16"
    assert "device_map" not in calls["kwargs"]
    assert model.moves == ["mps"]
    assert model.evaluated
    assert metadata["requested_device"] == "mps"
    assert metadata["device"] == "mps"
    assert metadata["dtype"] == "float16"
