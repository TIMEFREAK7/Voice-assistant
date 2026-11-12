"""Record the 50 phase 0 commands on your laptop: press Enter to start, Enter to stop (push-to-talk).

usage: python record.py <out_dir> [--from N]
Writes NN.wav as 16 kHz mono 16-bit, the format the app will capture. Speak the way you would to the app;
reword nothing. Re-run with --from N to redo command N onwards.
"""
import argparse, csv, queue, sys
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf

HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("--from", dest="start", type=int, default=1)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

for r in csv.DictReader(open(HERE / "commands.csv")):
    n = int(r["id"])
    if n < a.start:
        continue
    print(f"\n[{n}/50]  {r['text']}")
    input("  Enter to start... ")
    q: "queue.Queue[np.ndarray]" = queue.Queue()
    with sd.InputStream(samplerate=16000, channels=1, dtype="float32",
                        callback=lambda d, *_: q.put(d.copy())):
        input("  recording, Enter to stop ")
    audio = np.concatenate(list(q.queue)) if not q.empty() else np.zeros((0, 1), "float32")
    if len(audio) < 16000 * 0.5:
        sys.exit(f"  under half a second recorded for {n}; check the microphone, then re-run with --from {n}")
    sf.write(a.out / f"{n:02d}.wav", audio, 16000, subtype="PCM_16")
    print(f"  saved {len(audio) / 16000:.1f}s, peak {np.abs(audio).max():.2f}"
          + ("  (quiet: move closer)" if np.abs(audio).max() < 0.05 else ""))
print(f"\nDone. Score with: python score.py {a.out}")
