# Phase 0: try before building

Exit test from the scope doc: **45 of 50 commands usable** on your voice, and Kokoro's voice and
reply speed acceptable to you. A command is *usable* when every slot in `commands.csv` (the app
name, the time, the file words) appears in Whistle's transcript.

## On your Windows laptop (about 20 minutes)

```powershell
cd phase0
py -m venv .venv
.venv\Scripts\pip install cactus-compute "sherpa-onnx==1.13.8" sounddevice soundfile numpy jiwer
$env:CACTUS_TELEMETRY="0"; $env:DO_NOT_TRACK="1"
.venv\Scripts\python record.py mine          # Enter to start, Enter to stop, 50 times
.venv\Scripts\python score.py mine --out results_mine.csv
.venv\Scripts\python score.py mine --no-bias # how much keyword biasing helps you
```

The first `score.py` run downloads the engine and Whistle (about 20 MB) from Hugging Face.
Record in your normal room, at your normal distance and speed. Don't re-record a command because
Whistle got it wrong: that would bias the score.

Then commit `mine/` and `results_mine.csv`. The recordings become the regression set for every
later build and model pack.

## Also part of phase 0, with no code

- Play `synth_af/voice_sample_reply.wav` (Kokoro). Is the voice acceptable?
- Install Google's AI Edge Gallery on the OnePlus 13 and try Gemma 4 E2B and E4B: speed and answers.
- Note your laptop's RAM and GPU (Task Manager → Performance). They decide between E2B and E4B on the PC.

## Smoke test result (synthetic voice, not yours)

Kokoro's American voice through Whistle on a 4-core Linux CPU:

| Biasing | Usable | WER | Median latency |
|---|---|---|---|
| on  | 49/50 | 2.3% | 70 ms |
| off | 45/50 | 4.5% | 79 ms |

This proves the pipeline runs. It says nothing about your accent: a clean synthetic voice is the
easy case. "Aadhaar" failed even with biasing ("odd hard card"), so Indian proper nouns are the
first thing to watch in your own run.

## Findings for the build

- **Needle and Whistle on Windows and Android:** release 4.x ships `windows-x86_64`,
  `windows-arm64` and `android-arm64`, but only as a static `libcactus.a` plus a C header
  (`cactus.h`). There are no Kotlin bindings, so the app needs its own JNI glue.
- **Telemetry:** only the Python wrapper (`cactus/_telemetry.py`) sends usage counts, and it stops
  when `CACTUS_TELEMETRY=0` or `DO_NOT_TRACK` is set. The Linux native engine contains no URLs and
  imports no socket calls. The Windows and Android binaries are unchecked.
- **Kokoro:** use sherpa-onnx 1.13.8. 2.0.x requires a lexicon for Kokoro, and the multi-language
  v1.0 model then silently drops every word not in it, including digits, "contractor" and
  "delayed". Spoken replies would lose times and names.
