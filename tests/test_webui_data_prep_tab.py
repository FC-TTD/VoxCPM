import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from extend.webui_data_prep_tab import generate_manifest_jsonl, iter_wav_files


def _write_wav(path: Path, duration_sec: float = 1.0, sr: int = 16000) -> None:
    samples = int(duration_sec * sr)
    audio = np.zeros(samples, dtype=np.float32)
    sf.write(str(path), audio, sr)


def test_iter_wav_files_skips_hidden(tmp_path: Path) -> None:
    visible_dir = tmp_path / "data"
    hidden_dir = tmp_path / ".hidden"
    at_dir = tmp_path / "@private"
    visible_dir.mkdir()
    hidden_dir.mkdir()
    at_dir.mkdir()

    _write_wav(visible_dir / "a.wav")
    _write_wav(hidden_dir / "b.wav")
    _write_wav(at_dir / "c.wav")
    _write_wav(visible_dir / ".d.wav")
    _write_wav(visible_dir / "@e.wav")

    wavs = sorted([p.relative_to(tmp_path) for p in iter_wav_files(str(tmp_path))])

    assert wavs == [Path("data/a.wav")]


def test_generate_manifest_jsonl(tmp_path: Path) -> None:
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()
    wav1 = data_dir / "sample1.wav"
    wav2 = data_dir / "sample2.wav"
    _write_wav(wav1, duration_sec=1.0)
    _write_wav(wav2, duration_sec=2.0)

    def fake_asr(path: str) -> str:
        return "hello" if path.endswith("sample1.wav") else "world"

    output_path = tmp_path / "manifest.jsonl"
    message, count = generate_manifest_jsonl(
        data_dir=str(data_dir),
        output_jsonl=str(output_path),
        asr_func=fake_asr,
        use_relative_paths=True,
    )

    assert count == 2
    assert "生成 2 条" in message

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    second = json.loads(lines[1])

    assert first["audio"] in {"sample1.wav", "sample2.wav"}
    assert second["audio"] in {"sample1.wav", "sample2.wav"}
    assert {first["text"], second["text"]} == {"hello", "world"}
    assert first["duration"] > 0
    assert second["duration"] > 0


def test_generate_manifest_default_output_and_overwrite(tmp_path: Path) -> None:
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()
    wav1 = data_dir / "sample1.wav"
    _write_wav(wav1, duration_sec=1.0)

    def fake_asr(_: str) -> str:
        return "hello"

    message, count = generate_manifest_jsonl(
        data_dir=str(data_dir),
        output_jsonl="",
        asr_func=fake_asr,
        use_relative_paths=False,
    )

    expected_output = data_dir.with_suffix(".jsonl")
    assert count == 1
    assert expected_output.exists()
    assert "生成 1 条" in message

    message_again, count_again = generate_manifest_jsonl(
        data_dir=str(data_dir),
        output_jsonl="",
        asr_func=fake_asr,
    )
    assert count_again == 0
    assert "输出文件已存在" in message_again

    message_overwrite, count_overwrite = generate_manifest_jsonl(
        data_dir=str(data_dir),
        output_jsonl="",
        asr_func=fake_asr,
        overwrite=True,
    )
    assert count_overwrite == 1
    assert "生成 1 条" in message_overwrite


def test_filename_pattern_skips_asr(tmp_path: Path) -> None:
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()
    wav1 = data_dir / "1_阿福_对白_95_输了都怪你拉我干嘛呀？.wav"
    _write_wav(wav1, duration_sec=1.0)

    calls = {"count": 0}

    def fake_asr(_: str) -> str:
        calls["count"] += 1
        return "should_not_use"

    output_path = tmp_path / "manifest.jsonl"
    message, count = generate_manifest_jsonl(
        data_dir=str(data_dir),
        output_jsonl=str(output_path),
        asr_func=fake_asr,
        overwrite=True,
    )

    assert count == 1
    assert calls["count"] == 0
    assert "生成 1 条" in message
    entry = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert entry["text"] == "输了都怪你拉我干嘛呀？"
