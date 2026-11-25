import json
import shutil
import tempfile
import asyncio
import re
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict, Set
import numpy as np
import soundfile as sf

try:
    from .text_normalizer import (
        normalize_derja,
        clean_text,
        arabizi_to_derja_arabic,
        is_arabic_script
    )
    from .audio_player import AudioPlayer
except ImportError:
    from text_normalizer import (
        normalize_derja,
        clean_text,
        arabizi_to_derja_arabic,
        is_arabic_script
    )
    from audio_player import AudioPlayer

_candidate_root = Path(__file__).resolve().parent.parent
if (_candidate_root / "assets").exists():
    PROJECT_ROOT = _candidate_root
elif (_candidate_root.parent / "assets").exists():
    PROJECT_ROOT = _candidate_root.parent
else:
    PROJECT_ROOT = _candidate_root
TTS_ASSETS_DIR = PROJECT_ROOT / "assets" / "tts_data"
SIGN_ASSETS_DIR = PROJECT_ROOT / "assets" / "sign_data"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "tts" / "models" if (PROJECT_ROOT / "models").exists() else PROJECT_ROOT / "tts" / "models"

DESTINATIONS = {"sbitar", "baladya", "banka", "dar", "ma7kma"}
TRANSPORTS = {"taxi", "louage", "car", "karhba"}
DAYS = {"thnin", "thleth", "erb3a", "5mis", "jom3a", "sebt", "a7ad", "lyoum"}
GREETINGS = {"3aslema", "labes", "oui", "non", "enti", "n3awnek", "om"}


class TunisianTTSEngine:
    def __init__(
        self,
        voice: str = "ar-TN-HediNeural",
        sample_rate: int = 22050,
        enable_online: bool = True
    ):
        self.voice = voice
        self.sample_rate = sample_rate
        self.enable_online = enable_online
        self.player = AudioPlayer()
        self.fast_cache: Dict[str, Path] = {}
        self.word_cache: Dict[str, Path] = {}
        self.combo_cache: Dict[frozenset, Path] = {}
        self._build_fast_cache()

    def _build_fast_cache(self):
        wavs_dir = TTS_ASSETS_DIR / "wavs"
        all_txt = TTS_ASSETS_DIR / "filelists" / "all.txt"

        if all_txt.exists():
            try:
                with open(all_txt, "r", encoding="utf-8") as f:
                    for line in f:
                        if "|" in line:
                            rel_path, transcript = line.strip().split("|", 1)
                            audio_name = Path(rel_path).name
                            audio_file = wavs_dir / audio_name
                            if audio_file.exists():
                                self._register_key(transcript, audio_file)
                                ar_text = arabizi_to_derja_arabic(transcript)
                                self._register_key(ar_text, audio_file)
            except Exception:
                pass

        sentences_json = SIGN_ASSETS_DIR / "sentences.json"
        if not sentences_json.exists():
            sentences_json = PROJECT_ROOT / "assets" / "audio" / "sentences.json"
        if not sentences_json.exists():
            sentences_json = PROJECT_ROOT / "assets" / "sentences.json"

        if sentences_json.exists():
            try:
                with open(sentences_json, "r", encoding="utf-8") as f:
                    for item in json.load(f):
                        sid = item.get("id", "")
                        audio_field = item.get("audio", "")
                        words_list = item.get("words", [])

                        candidate = wavs_dir / audio_field
                        if not candidate.exists():
                            candidate = wavs_dir / f"{sid}.wav"
                        if not candidate.exists():
                            candidate = PROJECT_ROOT / "assets" / "audio" / audio_field
                        if not candidate.exists():
                            candidate = PROJECT_ROOT / "assets" / "audio" / f"{sid}.wav"

                        if candidate.exists():
                            if words_list:
                                self.combo_cache[frozenset(words_list)] = candidate
                                self._register_key(" ".join(words_list), candidate)
                                self._register_key(" ".join(reversed(words_list)), candidate)

                            for k in [item.get("derja"), item.get("french")]:
                                if k:
                                    self._register_key(k, candidate)
                                    self._register_key(arabizi_to_derja_arabic(k), candidate)
            except Exception:
                pass

        words_json = SIGN_ASSETS_DIR / "words.json"
        if not words_json.exists():
            words_json = PROJECT_ROOT / "assets" / "audio" / "words.json"
        if not words_json.exists():
            words_json = PROJECT_ROOT / "assets" / "words.json"

        if words_json.exists():
            try:
                with open(words_json, "r", encoding="utf-8") as f:
                    for item in json.load(f):
                        audio_name = item.get("audio", "")
                        folder = item.get("folder", "")
                        candidate = wavs_dir / audio_name
                        if not candidate.exists():
                            candidate = wavs_dir / f"{folder}.wav"
                        if not candidate.exists():
                            candidate = PROJECT_ROOT / "assets" / "audio" / audio_name
                        if not candidate.exists():
                            candidate = PROJECT_ROOT / "assets" / "audio" / f"{folder}.wav"

                        if candidate.exists():
                            self.word_cache[folder.lower()] = candidate
                            self.combo_cache[frozenset([folder.lower()])] = candidate
                            for k in [folder, item.get("derja"), item.get("french")]:
                                if k:
                                    self._register_key(k, candidate)
                                    self.word_cache[normalize_derja(k)] = candidate
                                    self.word_cache[clean_text(k)] = candidate
            except Exception:
                pass

        if wavs_dir.exists():
            for p in wavs_dir.glob("*.wav"):
                stem = p.stem.lower()
                self._register_key(stem, p)
                self.word_cache[stem] = p
                parts = stem.split("_")
                if len(parts) >= 2:
                    self.combo_cache[frozenset(parts)] = p
                    self._register_key(" ".join(parts), p)
                    self._register_key(" ".join(reversed(parts)), p)

    def _register_key(self, text: str, audio_path: Path):
        norm = normalize_derja(text)
        if norm:
            self.fast_cache[norm] = audio_path
        raw = clean_text(text)
        if raw:
            self.fast_cache[raw] = audio_path

    def _extract_tokens(self, input_data: Union[str, List[str]]) -> List[str]:
        if isinstance(input_data, list):
            tokens = [str(x).strip().lower() for x in input_data if str(x).strip()]
        else:
            cleaned = clean_text(str(input_data))
            tokens = re.findall(r"[\w\d]+", cleaned.lower())
        return tokens

    def find_cached_audio(self, text_or_tokens: Union[str, List[str]]) -> Optional[Path]:
        tokens = self._extract_tokens(text_or_tokens)
        if tokens:
            token_set = frozenset(tokens)
            if token_set in self.combo_cache and self.combo_cache[token_set].exists():
                return self.combo_cache[token_set]

            joined_str = " ".join(tokens)
            norm = normalize_derja(joined_str)
            if norm in self.fast_cache and self.fast_cache[norm].exists():
                return self.fast_cache[norm]

        if isinstance(text_or_tokens, str):
            text = text_or_tokens
            norm = normalize_derja(text)
            if norm in self.fast_cache and self.fast_cache[norm].exists():
                return self.fast_cache[norm]

            raw = clean_text(text)
            if raw in self.fast_cache and self.fast_cache[raw].exists():
                return self.fast_cache[raw]

            ar = normalize_derja(arabizi_to_derja_arabic(text))
            if ar in self.fast_cache and self.fast_cache[ar].exists():
                return self.fast_cache[ar]

            for k, p in self.fast_cache.items():
                if len(k) > 3 and (k == norm or norm == k or k in norm or norm in k):
                    if p.exists():
                        return p

        return None

    def synthesize(
        self,
        text_or_tokens: Union[str, List[str]],
        output_path: Optional[Union[str, Path]] = None
    ) -> Tuple[Path, np.ndarray, int]:
        if output_path is None:
            out_file = PROJECT_ROOT / "results" / "tts_output.wav"
        else:
            out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        cached_path = self.find_cached_audio(text_or_tokens)
        if cached_path and cached_path.exists():
            shutil.copy(cached_path, out_file)
            data, fs = sf.read(str(out_file))
            return out_file, data, fs

        tokens = self._extract_tokens(text_or_tokens)
        full_text = " ".join(tokens) if isinstance(text_or_tokens, list) else str(text_or_tokens)

        grammar_text = self._build_natural_derja(tokens)
        if grammar_text:
            grammar_match = self.find_cached_audio(grammar_text)
            if grammar_match and grammar_match.exists():
                shutil.copy(grammar_match, out_file)
                data, fs = sf.read(str(out_file))
                return out_file, data, fs
            full_text = grammar_text

        if self.enable_online:
            online_audio = self._generate_online_speech(full_text, out_file)
            if online_audio is not None:
                data, fs = sf.read(str(out_file))
                if tokens:
                    self.combo_cache[frozenset(tokens)] = out_file
                self._register_key(full_text, out_file)
                return out_file, data, fs

        audio_array = self._synthesize_composite(tokens, full_text)
        sf.write(str(out_file), audio_array, self.sample_rate)
        return out_file, audio_array, self.sample_rate

    def _build_natural_derja(self, tokens: List[str]) -> str:
        s = set(tokens)
        dest = list(s.intersection(DESTINATIONS))
        trans = list(s.intersection(TRANSPORTS))
        day = list(s.intersection(DAYS))

        dest_names = {
            "sbitar": "للسبيطار",
            "baladya": "للبلدية",
            "banka": "للبنكة",
            "ma7kma": "للمحكمة",
            "dar": "للدار"
        }
        trans_names = {
            "taxi": "بالتاكسي",
            "louage": "باللواج",
            "car": "بالكار",
            "karhba": "بالكرهبة"
        }
        day_names = {
            "lyoum": "اليوم",
            "thnin": "نهار الإثنين",
            "thleth": "نهار الثلاثاء",
            "erb3a": "نهار الإربعاء",
            "5mis": "نهار الخميس",
            "jom3a": "نهار الجمعة",
            "sebt": "نهار السبت",
            "a7ad": "نهار الأحد"
        }

        if dest and trans:
            d_name = dest_names.get(dest[0], dest[0])
            t_name = trans_names.get(trans[0], trans[0])
            if dest[0] == "dar":
                return f"نحب نرجع {d_name} {t_name}"
            return f"نحب نمشي {d_name} {t_name}"

        if day and dest:
            d_name = dest_names.get(dest[0], dest[0])
            dy_name = day_names.get(day[0], day[0])
            if day[0] == "lyoum":
                if dest[0] == "dar":
                    return "اليوم مروح للدار"
                return f"اليوم نمشي {d_name}"
            return f"نمشي {d_name} {dy_name}"

        if "om" in s and dest:
            d_loc = "في السبيطار" if dest[0] == "sbitar" else ("في الدار" if dest[0] == "dar" else f"في {dest[0]}")
            return f"ماما {d_loc}"

        if "3aslema" in s and "labes" in s:
            return "عسلامة، لاباس؟"

        if "enti" in s and "labes" in s:
            return "إنت لاباس؟"

        if "enti" in s and "n3awnek" in s:
            return "تحب نعاونك؟"

        return arabizi_to_derja_arabic(" ".join(tokens))

    def _generate_online_speech(self, text: str, target_path: Path) -> Optional[np.ndarray]:
        try:
            import edge_tts
            arabic_text = arabizi_to_derja_arabic(text)
            if not arabic_text:
                arabic_text = text

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_mp3 = tmp.name

            async def _run():
                comm = edge_tts.Communicate(arabic_text, self.voice)
                await asyncio.wait_for(comm.save(tmp_mp3), timeout=3.5)

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        pool.submit(lambda: asyncio.run(_run())).result(timeout=4.0)
                else:
                    asyncio.run(_run())
            except Exception:
                asyncio.run(_run())

            if not Path(tmp_mp3).exists() or Path(tmp_mp3).stat().st_size == 0:
                Path(tmp_mp3).unlink(missing_ok=True)
                return None

            data, fs = sf.read(tmp_mp3)
            Path(tmp_mp3).unlink(missing_ok=True)

            if data.ndim > 1:
                data = data.mean(axis=1)

            duration = len(data) / fs
            new_indices = np.linspace(0, len(data) - 1, int(duration * self.sample_rate))
            resampled = np.interp(new_indices, np.arange(len(data)), data)

            abs_sig = np.abs(resampled)
            nonzero = np.where(abs_sig > 0.01)[0]
            if len(nonzero) > 0:
                s = max(0, nonzero[0] - int(self.sample_rate * 0.05))
                e = min(len(resampled), nonzero[-1] + int(self.sample_rate * 0.05))
                resampled = resampled[s:e]

            sf.write(str(target_path), resampled, self.sample_rate, format="WAV", subtype="PCM_16")
            return resampled
        except Exception:
            return None

    def _synthesize_composite(self, tokens: List[str], raw_text: str) -> np.ndarray:
        words = tokens if tokens else normalize_derja(raw_text).split()
        collected_clips = []
        silence_pad = np.zeros(int(self.sample_rate * 0.06), dtype=np.float32)

        for w in words:
            clip = self.word_cache.get(w) or self.find_cached_audio(w)
            if clip and clip.exists():
                data, fs = sf.read(str(clip))
                if data.ndim > 1:
                    data = data.mean(axis=1)
                fade_len = min(len(data), int(self.sample_rate * 0.005))
                if fade_len > 0:
                    data[:fade_len] *= np.linspace(0, 1, fade_len)
                    data[-fade_len:] *= np.linspace(1, 0, fade_len)
                collected_clips.append(data.astype(np.float32))
                collected_clips.append(silence_pad)

        if collected_clips:
            return np.concatenate(collected_clips)

        fallback_clip = TTS_ASSETS_DIR / "wavs" / "tn_0001.wav"
        if fallback_clip.exists():
            data, fs = sf.read(str(fallback_clip))
            if data.ndim > 1:
                data = data.mean(axis=1)
            return data.astype(np.float32)

        return np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)

    def speak(self, text_or_tokens: Union[str, List[str]], async_playback: bool = False) -> bool:
        out_path, _, _ = self.synthesize(text_or_tokens)
        return self.player.play_wav(out_path, async_play=async_playback)

    def get_registered_phrases(self) -> List[str]:
        return sorted(list(set(self.fast_cache.keys())))
