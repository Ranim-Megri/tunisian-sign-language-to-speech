import argparse
import sys
from pathlib import Path
from typing import Optional, Union, Tuple, List
import numpy as np

try:
    from .engine import TunisianTTSEngine, PROJECT_ROOT
except ImportError:
    from engine import TunisianTTSEngine, PROJECT_ROOT

_DEFAULT_ENGINE: Optional[TunisianTTSEngine] = None


def get_engine() -> TunisianTTSEngine:
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = TunisianTTSEngine()
    return _DEFAULT_ENGINE


def synthesize(
    text_or_tokens: Union[str, List[str]],
    output_path: Optional[Union[str, Path]] = None
) -> Tuple[Path, np.ndarray, int]:
    engine = get_engine()
    return engine.synthesize(text_or_tokens, output_path=output_path)


def speak(text_or_tokens: Union[str, List[str]], async_playback: bool = False) -> bool:
    engine = get_engine()
    return engine.speak(text_or_tokens, async_playback=async_playback)


def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Tunisian Derja TTS Synthesis CLI.")
    parser.add_argument("--text", type=str, default=None, help="Text phrase to speak")
    parser.add_argument("--words", nargs="+", default=None, help="Sequence of recognized sign words")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "tts_output.wav", help="Target audio output WAV path")
    parser.add_argument("--play", action="store_true", help="Play audio after synthesis")
    parser.add_argument("--list-phrases", action="store_true", help="List zero-latency cached phrases")
    args = parser.parse_args()

    engine = get_engine()

    if args.list_phrases:
        phrases = engine.get_registered_phrases()
        for p in phrases[:40]:
            print(f"- {p}")
        return

    if args.words:
        target_input = args.words
    elif args.text:
        target_input = args.text
    else:
        target_input = "3aslema chnowa 7alek lyoum ?"

    out_file, audio_data, fs = engine.synthesize(target_input, args.out)
    duration = len(audio_data) / fs
    print(f"Synthesized: {target_input} -> {out_file} ({duration:.2f}s, {fs}Hz)")

    if args.play:
        engine.player.play_wav(out_file)


if __name__ == "__main__":
    main()
