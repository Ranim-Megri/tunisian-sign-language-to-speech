import os
import sys
import subprocess
import threading
from pathlib import Path
from typing import Union, Optional
import numpy as np


class AudioPlayer:
    def __init__(self):
        self.platform = sys.platform

    def play_wav(self, wav_path: Union[str, Path], async_play: bool = False) -> bool:
        wav_path = Path(wav_path).resolve()
        if not wav_path.exists():
            return False

        if async_play:
            thread = threading.Thread(target=self._play_blocking, args=(wav_path,), daemon=True)
            thread.start()
            return True
        else:
            return self._play_blocking(wav_path)

    def _play_blocking(self, wav_path: Path) -> bool:
        path_str = str(wav_path)

        if self.platform.startswith("linux"):
            for cmd in ["aplay", "paplay", "play"]:
                try:
                    res = subprocess.run([cmd, path_str], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if res.returncode == 0:
                        return True
                except FileNotFoundError:
                    continue

        elif self.platform == "win32":
            try:
                import winsound
                winsound.PlaySound(path_str, winsound.SND_FILENAME)
                return True
            except Exception:
                pass

            try:
                ps_cmd = f'(New-Object Media.SoundPlayer "{path_str}").PlaySync()'
                res = subprocess.run(["powershell", "-c", ps_cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if res.returncode == 0:
                    return True
            except Exception:
                pass

        elif self.platform == "darwin":
            try:
                res = subprocess.run(["afplay", path_str], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if res.returncode == 0:
                    return True
            except Exception:
                pass

        try:
            import soundfile as sf
            import sounddevice as sd
            data, fs = sf.read(path_str)
            sd.play(data, fs)
            sd.wait()
            return True
        except Exception:
            pass

        return False

    def play_array(self, audio: np.ndarray, sample_rate: int = 22050, async_play: bool = False) -> bool:
        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            sf.write(str(tmp_path), audio, sample_rate, format="WAV")
            success = self.play_wav(tmp_path, async_play=async_play)
        finally:
            if not async_play and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
        return success
