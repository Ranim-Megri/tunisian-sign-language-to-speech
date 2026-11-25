import argparse
from pathlib import Path
try:
    from .dataset import TunisianTTSDataset
except ImportError:
    from dataset import TunisianTTSDataset

_candidate_root = Path(__file__).resolve().parent.parent
if (_candidate_root / "assets").exists():
    PROJECT_ROOT = _candidate_root
elif (_candidate_root.parent / "assets").exists():
    PROJECT_ROOT = _candidate_root.parent
else:
    PROJECT_ROOT = _candidate_root
TTS_ASSETS_DIR = PROJECT_ROOT / "assets" / "tts_data"


def main():
    parser = argparse.ArgumentParser(description="Dataset Validation & Model Training Pipeline")
    parser.add_argument(
        "--filelist",
        type=Path,
        default=TTS_ASSETS_DIR / "filelists" / "all.txt",
        help="Path to training filelist"
    )
    args = parser.parse_args()

    ds = TunisianTTSDataset(args.filelist)
    stats = ds.validate()

    print("Dataset Validation:")
    print(f"- Total Entries: {stats['total_samples']}")
    print(f"- Audio Found:   {stats['valid_samples']}")
    print(f"- Missing:       {stats['missing_samples']}")
    print(f"- Audio Duration:{stats['total_duration_seconds']}s")

    if stats["valid_samples"] > 0:
        first_sample = ds[0]
        print(f"- Sample 0 Text: '{first_sample['text']}'")
        print(f"- Sample 0 Tokens ({len(first_sample['tokens'])}): {first_sample['tokens'][:10]}...")
        print(f"- Audio Shape: {first_sample['audio'].shape} @ {first_sample['sample_rate']}Hz")


if __name__ == "__main__":
    main()
