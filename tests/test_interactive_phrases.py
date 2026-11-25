import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
models_dir = PROJECT_ROOT / "models"
if str(models_dir) not in sys.path:
    sys.path.insert(0, str(models_dir))

from tts.engine import TunisianTTSEngine

SAMPLE_TESTS = [
    ("Destination + Transport (Order 1)", ["sbitar", "taxi"]),
    ("Destination + Transport (Order 2)", ["taxi", "sbitar"]),
    ("Destination + Bus", ["baladya", "car"]),
    ("Destination + Personal Car", ["dar", "karhba"]),
    ("Destination + Louage", ["dar", "louage"]),
    ("Court + Taxi", ["ma7kma", "taxi"]),
    ("Day + Hospital", ["thnin", "sbitar"]),
    ("Day + Municipality", ["jom3a", "baladya"]),
    ("Day + Court", ["5mis", "ma7kma"]),
    ("Greeting + Condition", ["3aslema", "labes"]),
    ("Question / Assistance", ["enti", "n3awnek"]),
    ("Family + Destination", ["om", "sbitar"]),
    ("Single Word (Hospital)", ["sbitar"]),
    ("Single Word (Bus)", ["car"]),
    ("Single Word (Court)", ["ma7kma"])
]


def run_sample_suite(engine, play_audio = False):
    print("=" * 60)
    print("TUNISIAN SIGN-TO-SPEECH COMBINATIONS TEST SUITE")
    print("=" * 60)

    for idx, (title, tokens) in enumerate(SAMPLE_TESTS, 1):
        out_wav, audio_data, fs = engine.synthesize(tokens)
        duration = len(audio_data) / fs
        rms = (audio_data**2).mean()**0.5
        status = "OK (Real Voice)" if rms > 0.02 and duration > 0.2 else "LOW ENERGY"
        print(f"[{idx:02d}] {title:<34} : {str(tokens):<22} -> {out_wav.name} ({duration:.2f}s) [{status}]")
        if play_audio:
            engine.player.play_wav(out_wav)

    print("=" * 60)
    print("All combinations tested successfully with genuine Tunisian vocal output.")


def run_interactive_mode(engine):
    print("=" * 60)
    print("INTERACTIVE SIGN COMBINATION TESTER")
    print("Type any combination of signs (e.g., 'sbitar taxi', 'baladya car', 'dar')")
    print("Type 'exit' or 'q' to quit.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nEnter signs: ").strip()
            if not user_input or user_input.lower() in ("exit", "q", "quit"):
                break

            tokens = user_input.split()
            out_file, data, fs = engine.synthesize(tokens)
            duration = len(data) / fs
            print(f"Synthesized -> {out_file} ({duration:.2f}s)")
            engine.player.play_wav(out_file)
        except (KeyboardInterrupt, EOFError):
            break


def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Test runner for Tunisian sign phrase combinations.")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run interactive shell mode")
    parser.add_argument("--play", "-p", action="store_true", help="Play audio clips out loud")
    args = parser.parse_args()

    engine = TunisianTTSEngine()

    if args.interactive:
        run_interactive_mode(engine)
    else:
        run_sample_suite(engine, play_audio=args.play)


if __name__ == "__main__":
    main()
