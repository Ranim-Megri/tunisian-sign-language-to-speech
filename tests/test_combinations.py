import sys
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
models_dir = PROJECT_ROOT / "models"
if str(models_dir) not in sys.path:
    sys.path.insert(0, str(models_dir))

from tts.engine import TunisianTTSEngine

RESULTS_DIR = PROJECT_ROOT / "results" / "test_outputs"


class TestTunisianTTSCombinations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = TunisianTTSEngine()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def _verify_audio(self, path: Path, audio: np.ndarray, fs: int, min_duration: float = 0.25):
        self.assertTrue(path.exists())
        self.assertEqual(fs, 22050)
        duration = len(audio) / fs
        self.assertGreater(duration, min_duration)

        rms = np.sqrt(np.mean(audio**2))
        self.assertGreater(rms, 0.015)

        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / fs)
        centroid = np.sum(freqs * fft) / (np.sum(fft) + 1e-9)
        self.assertGreater(centroid, 500.0)
        self.assertLess(centroid, 4500.0)

    def test_single_signs(self):
        signs = [
            "3aslema", "labes", "oui", "non", "sbitar", "baladya",
            "banka", "dar", "ma7kma", "taxi", "louage", "car",
            "karhba", "thnin", "thleth", "erb3a", "5mis", "jom3a",
            "sebt", "a7ad", "lyoum", "om", "enti", "n3awnek"
        ]
        for sign in signs:
            with self.subTest(sign=sign):
                out_path = RESULTS_DIR / f"test_single_{sign}.wav"
                p, data, fs = self.engine.synthesize(sign, out_path)
                self._verify_audio(p, data, fs, min_duration=0.2)

    def test_destination_and_transport_combinations(self):
        pairs = [
            ("sbitar", "taxi"),
            ("baladya", "taxi"),
            ("banka", "taxi"),
            ("ma7kma", "taxi"),
            ("dar", "taxi"),
            ("sbitar", "louage"),
            ("baladya", "louage"),
            ("banka", "louage"),
            ("ma7kma", "louage"),
            ("dar", "louage"),
            ("sbitar", "car"),
            ("baladya", "car"),
            ("banka", "car"),
            ("ma7kma", "car"),
            ("dar", "car"),
            ("sbitar", "karhba"),
            ("dar", "karhba"),
        ]
        for w1, w2 in pairs:
            with self.subTest(pair=f"{w1}_{w2}"):
                out_path = RESULTS_DIR / f"test_combo_{w1}_{w2}.wav"
                p1, d1, fs1 = self.engine.synthesize([w1, w2], out_path)
                self._verify_audio(p1, d1, fs1, min_duration=0.8)

    def test_order_invariance(self):
        pairs = [
            (["sbitar", "taxi"], ["taxi", "sbitar"]),
            (["dar", "louage"], ["louage", "dar"]),
            (["baladya", "car"], ["car", "baladya"]),
            (["ma7kma", "taxi"], ["taxi", "ma7kma"]),
            (["dar", "karhba"], ["karhba", "dar"]),
            (["jom3a", "baladya"], ["baladya", "jom3a"]),
            (["thnin", "sbitar"], ["sbitar", "thnin"]),
            (["3aslema", "labes"], ["labes", "3aslema"]),
            (["enti", "labes"], ["labes", "enti"]),
        ]
        for order1, order2 in pairs:
            with self.subTest(order1=order1, order2=order2):
                p1 = self.engine.find_cached_audio(order1)
                p2 = self.engine.find_cached_audio(order2)
                self.assertIsNotNone(p1)
                self.assertIsNotNone(p2)
                self.assertEqual(p1.resolve(), p2.resolve())

    def test_days_and_destination_combinations(self):
        schedules = [
            ["lyoum", "banka"],
            ["lyoum", "sbitar"],
            ["lyoum", "baladya"],
            ["lyoum", "dar"],
            ["thnin", "sbitar"],
            ["thleth", "sbitar"],
            ["erb3a", "sbitar"],
            ["5mis", "sbitar"],
            ["jom3a", "sbitar"],
            ["sebt", "sbitar"],
            ["thnin", "baladya"],
            ["5mis", "baladya"],
            ["jom3a", "baladya"],
            ["thnin", "ma7kma"],
            ["thleth", "ma7kma"],
            ["5mis", "ma7kma"],
        ]
        for combo in schedules:
            with self.subTest(combo="_".join(combo)):
                out_path = RESULTS_DIR / f"test_sched_{'_'.join(combo)}.wav"
                p, data, fs = self.engine.synthesize(combo, out_path)
                self._verify_audio(p, data, fs, min_duration=0.7)

    def test_dialogues_and_family(self):
        dialogues = [
            ["3aslema", "labes"],
            ["enti", "labes"],
            ["enti", "n3awnek"],
            ["oui", "labes"],
            ["non", "labes"],
            ["om", "sbitar"],
            ["om", "dar"],
        ]
        for d in dialogues:
            with self.subTest(dialogue="_".join(d)):
                out_path = RESULTS_DIR / f"test_dial_{'_'.join(d)}.wav"
                p, data, fs = self.engine.synthesize(d, out_path)
                self._verify_audio(p, data, fs, min_duration=0.6)

    def test_input_representations(self):
        inputs = [
            ["sbitar", "taxi"],
            "sbitar taxi",
            "taxi sbitar",
            "n7eb nemchi lel sbitar bel taxi",
            "نحب نمشي للسبيطار بالتاكسي"
        ]
        for inp in inputs:
            with self.subTest(inp=str(inp)):
                p, data, fs = self.engine.synthesize(inp)
                self._verify_audio(p, data, fs, min_duration=0.8)


if __name__ == "__main__":
    unittest.main()
