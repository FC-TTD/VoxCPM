import json
import logging
import os
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

import gradio as gr
import soundfile as sf

from lib.asr_client import auto_asr

logger = logging.getLogger(__name__)


def _is_hidden_name(name: str) -> bool:
    return name.startswith(".") or name.startswith("@")


def iter_wav_files(root_dir: str) -> Iterable[Path]:
    if not root_dir:
        return []
    root_path = Path(root_dir)
    if not root_path.exists():
        return []

    for current_root, dirs, files in os.walk(root_path, topdown=True):
        dirs[:] = [d for d in dirs if not _is_hidden_name(d)]
        for filename in files:
            if _is_hidden_name(filename):
                continue
            if filename.lower().endswith(".wav"):
                yield Path(current_root) / filename


def get_audio_duration(audio_path: Path) -> float:
    with sf.SoundFile(str(audio_path)) as f:
        if f.samplerate <= 0:
            return 0.0
        return float(len(f)) / float(f.samplerate)


def extract_text_from_filename(wav_path: Path) -> str | None:
    stem = wav_path.stem
    parts = stem.split("_")
    if len(parts) >= 5:
        return parts[-1]
    return None


def generate_manifest_jsonl(
    data_dir: str,
    output_jsonl: str,
    asr_func: Callable[[str], str] | None = None,
    overwrite: bool = False,
    use_relative_paths: bool = False,
) -> Tuple[str, int]:
    if not data_dir or not os.path.isdir(data_dir):
        return "错误：数据目录不存在或不可用", 0
    if not output_jsonl or not output_jsonl.strip():
        output_path = Path(data_dir).resolve().with_suffix(".jsonl")
    else:
        output_path = Path(output_jsonl).expanduser()
    if output_path.exists() and not overwrite:
        return f"错误：输出文件已存在，请勾选覆盖或更换路径：{output_path}", 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries: List[dict] = []
    skipped = 0

    asr_runner = asr_func or auto_asr

    for wav_path in iter_wav_files(data_dir):
        try:
            filename_text = extract_text_from_filename(wav_path)
            if filename_text:
                text = filename_text.strip()
            else:
                text = asr_runner(str(wav_path)).strip()
        except Exception:
            logger.exception("ASR 失败：%s", wav_path)
            skipped += 1
            continue

        if not text:
            skipped += 1
            continue

        try:
            duration = get_audio_duration(wav_path)
        except Exception:
            logger.exception("读取时长失败：%s", wav_path)
            skipped += 1
            continue

        if use_relative_paths:
            audio_path = str(wav_path.relative_to(Path(data_dir)))
        else:
            audio_path = str(wav_path.resolve())

        entries.append({"audio": audio_path, "text": text, "duration": duration})

    with output_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return (
        f"完成：生成 {len(entries)} 条，跳过 {skipped} 条。输出：{output_path}",
        len(entries),
    )


def build_data_prep_tab(
    asr_func: Callable[[str], str] | None = None,
) -> Tuple[gr.Textbox, gr.Textbox, gr.TextArea]:
    gr.Markdown(
        """
        ### 🧰 数据准备 (Data Prep)
        扫描指定目录下的 wav 文件，自动 ASR + 时长提取，生成可训练的 jsonl。
        """
    )

    data_dir = gr.Textbox(
        label="📂 数据目录 (包含 wav)",
        placeholder="/app/data/afu",
        elem_classes="input-field",
    )
    output_jsonl = gr.Textbox(
        label="📄 输出 jsonl 路径（可选）",
        value=None,
        elem_classes="input-field",
    )
    overwrite = gr.Checkbox(
        label="覆盖已存在的 jsonl",
        value=False,
    )
    use_relative = gr.Checkbox(
        label="使用相对路径 (相对数据目录)",
        value=False,
    )

    run_btn = gr.Button("🧾 生成 jsonl", variant="primary", elem_classes="button-primary")
    status = gr.TextArea(
        label="",
        lines=4,
        interactive=False,
        elem_classes="input-field",
        show_label=False,
        placeholder="等待生成...",
    )

    def _run(
        data_dir_val: str,
        output_jsonl_val: str,
        overwrite_val: bool,
        use_relative_val: bool,
    ) -> str:
        message, _ = generate_manifest_jsonl(
            data_dir=data_dir_val,
            output_jsonl=output_jsonl_val,
            asr_func=asr_func,
            overwrite=overwrite_val,
            use_relative_paths=use_relative_val,
        )
        return message

    run_btn.click(
        _run,
        inputs=[data_dir, output_jsonl, overwrite, use_relative],
        outputs=[status],
    )

    return data_dir, output_jsonl, status
