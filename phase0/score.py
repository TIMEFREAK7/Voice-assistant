"""Phase 0 scorer: transcribe NN.wav recordings with Whistle and count usable commands.

usage: python score.py <wav_dir> [--no-bias] [--out results.csv]
A command is usable when every slot in commands.csv appears in the normalised transcript.
Pass mark from the scope doc: 45 of 50.
"""
import argparse, csv, os, re, sys, time
from pathlib import Path

os.environ.setdefault("CACTUS_TELEMETRY", "0")  # the scope doc requires no outbound traffic
os.environ.setdefault("DO_NOT_TRACK", "1")

import numpy as np
import soundfile as sf
import jiwer
import cactus

HERE = Path(__file__).parent
# Keyword biasing: names a stock speech model is likely to split or misspell.
KEYWORDS = ["Primavera", "WhatsApp", "AutoCAD", "PowerPoint", "Outlook", "Spotify", "Teams",
            "PepsiCo", "Ashoka", "Aadhaar", "Larsen and Toubro", "Suresh"]
NUMS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}
NUMS.update({"thirty": "30", "forty": "40", "fifty": "50"})
NUMS.update({w: f"{i}th" for i, w in enumerate("zeroth first second third fourth fifth sixth seventh "
                                                "eighth ninth tenth".split())})


def norm(s: str) -> str:
    s = s.lower().replace(":", " ").replace("-", " ").replace("’", "'")
    s = re.sub(r"\b(\d+) 00\b", r"\1", s)            # 5:00 -> 5
    s = re.sub(r"(\d)\s*([ap])\.? ?m\b\.?", r"\1 \2m", s)  # 10am, 10 p.m. -> 10 am, 10 pm
    s = re.sub(r"'s\b|s'(?=\s|$)", "s", s)              # contractor's / contractors' -> contractors
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    # shortcut: word->digit only for 0-20, tens and ordinals to tenth; "twenty five" stays partly words.
    words = [NUMS.get(w, w) for w in s.split() if w != "of"]  # "first of december" == "1 december"
    out = " ".join(words)
    return re.sub(r"\b(\d+)(st|nd|rd|th)?\b", lambda m: m[1] + ("th" if m[2] else ""), out)


def same_slot(slot: str, hyp: str) -> bool:
    # an ordinal date slot ("1th december") also matches the bare day ("1 december")
    # a trailing plural "s" still names the right thing ("contractors" for "contractor")
    return any(re.search(rf"(^| ){re.escape(s)}s?( |$)", hyp)
               for s in (slot, re.sub(r"(\d+)th", r"\1", slot)))


def load16k(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != 16000:  # shortcut: linear resample is fine for speech scoring, not for training data
        t = np.arange(0, len(audio) / sr, 1 / 16000)
        audio = np.interp(t, np.arange(len(audio)) / sr, audio).astype("float32")
    return audio


def slot_ok(slots: str, hyp: str) -> bool:
    return all(same_slot(norm(s), hyp) for s in slots.split("|"))


assert norm("Remind me at 10am.") == "remind me at 10 am"
assert norm("the contractor's letter") == norm("the contractors' letter")
assert same_slot(norm("first of december"), norm("Remind me on 1 December about"))
assert same_slot(norm("15th"), norm("on the 15th")) and not same_slot(norm("odd"), norm("aadhaar"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav_dir", type=Path)
    ap.add_argument("--no-bias", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(HERE / "commands.csv")))
    results, refs, hyps = [], [], []
    for r in rows:
        wav = a.wav_dir / f"{int(r['id']):02d}.wav"
        if not wav.exists():
            print(f"missing {wav.name}", file=sys.stderr)
            continue
        t0 = time.time()
        res = cactus.transcribe(load16k(wav), language="en",
                                keywords=None if a.no_bias else KEYWORDS)
        ms = int((time.time() - t0) * 1000)
        hyp = norm(res["text"])
        ok = slot_ok(r["slots"], hyp)
        refs.append(norm(r["text"])); hyps.append(hyp or "<empty>")
        results.append({"id": r["id"], "kind": r["kind"], "expected": r["text"],
                        "heard": res["text"].strip(), "usable": ok, "ms": ms})
        print(f"{'OK ' if ok else 'BAD'} {r['id']:>2} {ms:>5}ms  {res['text'].strip()}")

    if not results:
        sys.exit("no recordings found")
    n_ok = sum(x["usable"] for x in results)
    print(f"\nusable {n_ok}/{len(results)}  WER {jiwer.wer(refs, hyps):.1%}  "
          f"median {sorted(x['ms'] for x in results)[len(results)//2]}ms  bias={'off' if a.no_bias else 'on'}")
    for kind in ("open", "remind", "find", "dictate"):
        k = [x for x in results if x["kind"] == kind]
        if k:
            print(f"  {kind:8} {sum(x['usable'] for x in k)}/{len(k)}")
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)
    print("PASS" if n_ok >= 45 and len(results) == 50 else "FAIL (pass = 45 of 50, all 50 recorded)")


if __name__ == "__main__":
    main()
