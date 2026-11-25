import os
from pathlib import Path
from typing import List, Dict, Tuple
import soundfile as sf
import numpy as np

try:
    from .text_normalizer import normalize_derja, text_to_token_ids
except ImportError:
    from text_normalizer import normalize_derja, text_to_token_ids

_candidate_root = Path(__file__).resolve().parent.parent
if (_candidate_root / "assets").exists():
    PROJECT_ROOT = _candidate_root
elif (_candidate_root.parent / "assets").exists():
    PROJECT_ROOT = _candidate_root.parent
else:
    PROJECT_ROOT = _candidate_root
TTS_ASSETS_DIR = PROJECT_ROOT / "assets" / "tts_data"


class TunisianTTSDataset:
    def __init__(self, filelist_path: Path):
        self.filelist_path = Path(filelist_path)
        self.samples: List[Dict[str, str]] = []
        self._load()

    def _load(self):
        if not self.filelist_path.exists():
            return

        with open(self.filelist_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "|" not in line:
                    continue
                audio_rel, text = line.split("|", 1)
                audio_path = PROJECT_ROOT / audio_rel
                if not audio_path.exists():
                    audio_path = TTS_ASSETS_DIR / "wavs" / Path(audio_rel).name
                if not audio_path.exists():
                    audio_path = PROJECT_ROOT / "assets" / "audio" / Path(audio_rel).name

                self.samples.append({
                    "audio_path": str(audio_path),
                    "text": text,
                    "normalized_text": normalize_derja(text),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        item = self.samples[idx]
        tokens = text_to_token_ids(item["normalized_text"])
        data, fs = sf.read(item["audio_path"])
        if data.ndim > 1:
            data = data.mean(axis=1)

        return {
            "tokens": np.array(tokens, dtype=np.int64),
            "audio": data.astype(np.float32),
            "sample_rate": fs,
            "text": item["text"],
            "path": item["audio_path"]
        }

    def validate(self) -> Dict[str, int]:
        valid_count = 0
        missing_count = 0
        total_duration = 0.0

        for item in self.samples:
            p = Path(item["audio_path"])
            if p.exists():
                valid_count += 1
                info = sf.info(str(p))
                total_duration += info.duration
            else:
                missing_count += 1

        return {
            "total_samples": len(self.samples),
            "valid_samples": valid_count,
            "missing_samples": missing_count,
            "total_duration_seconds": int(total_duration),
        }
