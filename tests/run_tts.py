"""run_tts.py: Test script for the Tunisian Arabic (Derja) TTS Engine.

Uses the TunisianTTSEngine from models/tts (cached audio + edge-tts online
fallback + word-concatenation composite fallback).  No Tacotron fine-tuning.

Usage:
    # Synthesize from free-form text:
    python tests/run_tts.py --text "عسلامة، كيفاش حالك؟"

    # Synthesize from a sequence of recognised sign-language tokens:
    python tests/run_tts.py --words 3aslema labes

    # List all cached phrases the engine can serve instantly:
    python tests/run_tts.py --list-phrases

    # Synthesize and play the result:
    python tests/run_tts.py --text "lyoum sbitar" --play
"""
import argparse
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── path setup ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_models_dir = PROJECT_ROOT / "models"
for p in [str(PROJECT_ROOT), str(_models_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from tts.engine import TunisianTTSEngine          # models/tts/engine.py
from tts.inference import synthesize, get_engine   # models/tts/inference.py


def main():
    parser = argparse.ArgumentParser(
        description="Tunisian Derja TTS – synthesis only (uses models/tts engine)."
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="Free-form text to synthesize (Arabizi or Arabic script)."
    )
    parser.add_argument(
        "--words", nargs="+", default=None,
        help="Ordered list of recognised sign-language word tokens."
    )
    parser.add_argument(
        "--out", type=Path,
        default=PROJECT_ROOT / "results" / "tts_output.wav",
        help="Output WAV file path."
    )
    parser.add_argument(
        "--play", action="store_true",
        help="Play the audio after synthesis."
    )
    parser.add_argument(
        "--list-phrases", action="store_true",
        help="Print the cached phrases the engine can serve with zero latency."
    )
    args = parser.parse_args()

    engine = get_engine()

    # ── list cached phrases ──────────────────────────────────────────────
    if args.list_phrases:
        phrases = engine.get_registered_phrases()
        print(f"Cached phrases ({len(phrases)} total):")
        for p in phrases:
            print(f"  • {p}")
        return

    # ── determine input ──────────────────────────────────────────────────
    if args.words:
        target_input = args.words
    elif args.text:
        target_input = args.text
    else:
        target_input = "عسلامة، كيفاش حالك اليوم؟"

    # ── synthesize ───────────────────────────────────────────────────────
    out_file, audio_data, fs = engine.synthesize(target_input, args.out)
    duration = len(audio_data) / fs
    print(f"✓ Synthesized → {out_file}  ({duration:.2f}s, {fs} Hz)")

    # ── optional playback ────────────────────────────────────────────────
    if args.play:
        engine.player.play_wav(out_file)


if __name__ == "__main__":
    main()
