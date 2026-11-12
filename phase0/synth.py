"""Synthesise commands.csv with Kokoro (sherpa-onnx): a pipeline smoke test and the phase 0 voice sample.

usage: python synth.py <out_dir> [--speaker N]
Synthetic speech says nothing about your accent; it only proves the scorer and Kokoro run.
Uses kokoro-en-v0_19 (espeak-ng phonemes): the multi-lang v1.0 lexicon path silently drops
every word not in its lexicon, including digits, "contractor" and "delayed".
"""
import argparse, csv, sys, time
from pathlib import Path
import sherpa_onnx, soundfile as sf

HERE = Path(__file__).parent
M = HERE / "models" / "kokoro-en-v0_19"

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("--speaker", type=int, default=0)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
    kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
        model=str(M / "model.onnx"), voices=str(M / "voices.bin"), tokens=str(M / "tokens.txt"),
        data_dir=str(M / "espeak-ng-data")),
    num_threads=4)))


def say(text: str, path: Path):
    t0 = time.time()
    audio = tts.generate(text, sid=a.speaker, speed=1.0)
    if len(audio.samples) == 0:
        sys.exit(f"Kokoro produced no audio for: {text}")
    sf.write(path, audio.samples, audio.sample_rate)
    print(f"{len(audio.samples) / audio.sample_rate:.1f}s audio in {time.time() - t0:.2f}s  {text}")


for r in csv.DictReader(open(HERE / "commands.csv")):
    say(r["text"], a.out / f"{int(r['id']):02d}.wav")
say("Reminder set for today at 5 PM, to call the contractor. Say undo to cancel it.",
    a.out / "voice_sample_reply.wav")
