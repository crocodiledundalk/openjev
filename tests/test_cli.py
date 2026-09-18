import sys

import pytest

from openjev_phase1 import cli
from openjev_phase1.runtime import RuntimeSpec


def test_help_lists_device_option(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["openjev-score", "--help"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 0
    assert "--device {auto,cuda,mps,cpu}" in capsys.readouterr().out


def test_unsupported_mps_mode_fails_before_model_loading(monkeypatch, tmp_path, capsys):
    source = tmp_path / "input.jsonl"
    source.write_text(
        '{"id":"x","state":"s","question":"q","options":'
        '[{"id":"a","description":"A"},{"id":"b","description":"B"}]}\n'
    )
    output = tmp_path / "output.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "openjev-score",
            "--mode",
            "shared",
            "--device",
            "mps",
            "--model",
            "model",
            "--revision",
            "a" * 40,
            "--input",
            str(source),
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(cli, "resolve_runtime", lambda requested: RuntimeSpec(requested, "mps", "float16"))
    monkeypatch.setattr(cli, "load_causal_model", lambda *args, **kwargs: pytest.fail("model loaded"))

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == 2
    assert "supports only direct mode" in capsys.readouterr().err
    assert not output.exists()
