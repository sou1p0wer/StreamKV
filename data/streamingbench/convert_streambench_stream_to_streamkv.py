#!/usr/bin/env python3
"""Convert StreamingBench *stream* annotations into the StreamKV JSON format.

Source (official stream schema), per record:
    {
        "video_path": "./videos/sample_348_real.mp4",
        "questions": [
            {
                "task_type": "...",
                "question": "...",
                "time_stamp": "00:00:31",   # HH:MM:SS — single point in time
                "answer": "B",              # option LETTER
                "options": ["A. ...", "B. ...", ...],   # may or may not carry "A. " prefix
                "required_ability": "...",
                "audio_path": "...",
                "time": ...
            },
            ...
        ]
    }

Output (StreamKV schema), per record:
    {
        "video_id": "sample_348_real",
        "video_path": "<video_prefix>/sample_348_real.mp4",
        "duration": <last question's time_stamp in seconds>,
        "conversations": [
            {
                "question": "...",
                "choices": ["...", "...", ...],          # "A. " prefix stripped if present
                "answer": "<choice TEXT>",               # letter -> text
                "question_type": "<task_type>",
                "start_time": 0,                         # StreamKV ignores this (always encodes from 0)
                "end_time": <time_stamp in seconds>       # StreamKV slices video[0 : end_time*sample_fps]
            },
            ...
        ]
    }

Notes
-----
- StreamKV's stream solver (video_qa/streamkv_stream_vqa.py) reads only `end_time`;
  `start_time` and `duration` are passthrough (filled 0 / last-time_stamp respectively).
- `omni`/`real`/`sqa` are multiple-choice: source is ``questions_<task>_stream.json``
  (``answer`` = option letter; ``options`` carry an optional "A. " prefix).
- `proactive` has no ``_stream`` variant — source is ``questions_proactive.json`` and
  each record carries a top-level ``time`` window ``[start - end]`` plus per-question
  ``ground_truth_time_stamp`` / ``ground_truth_output`` (no options / answer letter).
  Mapping: ``start_time``/``end_time`` <- window; ``ground_truth_time`` <-
  ``ground_truth_time_stamp``; ``answer``/``ground_truth_output`` <- ``ground_truth_output``.
- `video_path` defaults to a repo-relative path under the videos dir
  (``--video_prefix`` default ``data/streamingbench/StreamingBench/src/data/videos``), so the
  generated JSONs carry no absolute paths and work after cloning anywhere. Override
  ``--video_prefix`` to repoint elsewhere, or pass ``--video_prefix ""`` to keep
  the source path verbatim.
"""
import argparse
import json
import os
import re
import sys

PREFIX_RE = re.compile(r"^[A-H]\.\s")
WIN_RE = re.compile(r"\[\s*(.+?)\s*-\s*(.+?)\s*\]")


def ts_to_seconds(time_stamp: str) -> int:
    """'00:00:31' -> 31 (int seconds). Supports HH:MM:SS and MM:SS."""
    parts = time_stamp.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        raise ValueError(f"unexpected time_stamp format: {time_stamp!r}")
    return int(h) * 3600 + int(m) * 60 + int(s)


def strip_option_prefix(option: str) -> str:
    """'A. foo' -> 'foo'; 'foo' -> 'foo' (sqa ships unprefixed)."""
    return PREFIX_RE.sub("", option)


def letter_to_index(letter: str) -> int:
    """'A' -> 0, 'B' -> 1, ..."""
    letter = letter.strip().upper()
    if len(letter) != 1 or not letter.isalpha():
        raise ValueError(f"answer is not a single letter: {letter!r}")
    return ord(letter) - ord("A")


def convert_record(stream_rec: dict, video_prefix: str | None) -> dict:
    video_path = stream_rec["video_path"]
    if video_prefix:
        video_path = os.path.join(video_prefix, os.path.basename(video_path))

    video_id = os.path.splitext(os.path.basename(video_path))[0]

    questions = stream_rec["questions"]
    # Questions are time-ascending by construction; keep order.
    duration = ts_to_seconds(questions[-1]["time_stamp"]) if questions else 0

    conversations = []
    for q in questions:
        choices = [strip_option_prefix(o) for o in q["options"]]
        answer_letter = q["answer"]
        answer_idx = letter_to_index(answer_letter)
        if answer_idx >= len(choices):
            raise ValueError(
                f"answer '{answer_letter}' (idx {answer_idx}) out of range for "
                f"{len(choices)} choices in video {video_id}"
            )
        answer_text = choices[answer_idx]
        conversations.append({
            "question": q["question"],
            "choices": choices,
            "answer": answer_text,
            "question_type": q.get("task_type"),
            "start_time": 0,
            "end_time": ts_to_seconds(q["time_stamp"]),
        })

    return {
        "video_id": video_id,
        "video_path": video_path,
        "duration": duration,
        "conversations": conversations,
    }


def parse_window(time_field: str) -> tuple[int, int]:
    """'[0:00:00 - 0:00:30]' -> (0, 30) as (start_seconds, end_seconds).

    Proactive source records carry the question's temporal window as a top-level
    ``time`` field of the form ``[start - end]`` (both H:MM:SS).
    """
    m = WIN_RE.search(time_field)
    if not m:
        raise ValueError(f"unexpected time window format: {time_field!r}")
    return ts_to_seconds(m.group(1)), ts_to_seconds(m.group(2))


def convert_proactive_record(stream_rec: dict, video_prefix: str | None) -> dict:
    """Convert a proactive-output source record into the StreamKV schema.

    Source (questions_proactive.json), per record:
        {
            "time": "[0:00:00 - 0:00:30]",            # temporal window of the question
            "video_path": ".../sample_17_proactive.mp4",
            "questions": [
                {
                    "task_type": "Proactive Output",
                    "question": "When the scoreboard shows 3 ...",
                    "time_stamp": "00:00:20",                 # unused (ask time)
                    "ground_truth_time_stamp": "00:00:29",    # -> ground_truth_time
                    "ground_truth_output": "3",                # -> answer + ground_truth_output
                    "required_ability": "..."                 # unused
                }, ...
            ]
        }

    Output conversation: {question, answer, start_time, end_time,
    ground_truth_time, ground_truth_output, task_type} — start/end come from the
    top-level ``time`` window, ground_truth_time from ``ground_truth_time_stamp``
    (both H:MM:SS -> int seconds). ``answer`` mirrors ``ground_truth_output``
    (the gold output), matching the meaning of ``answer`` in the MC subsets.
    """
    video_path = stream_rec["video_path"]
    if video_prefix:
        video_path = os.path.join(video_prefix, os.path.basename(video_path))
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    start_time, end_time = parse_window(stream_rec["time"])

    conversations = []
    for q in stream_rec["questions"]:
        gt_output = q["ground_truth_output"]
        conversations.append({
            "question": q["question"],
            "answer": gt_output,
            "start_time": start_time,
            "end_time": end_time,
            "ground_truth_time": ts_to_seconds(q["ground_truth_time_stamp"]),
            "ground_truth_output": gt_output,
            "task_type": q.get("task_type"),
        })

    return {
        "video_id": video_id,
        "video_path": video_path,
        "conversations": conversations,
    }


def convert_task(src_dir: str, out_dir: str, task: str, video_prefix: str | None,
                 *, src_filename: str | None = None,
                 convert_fn=convert_record) -> str | None:
    src_filename = src_filename or f"questions_{task}_stream.json"
    src_path = os.path.join(src_dir, src_filename)
    if not os.path.isfile(src_path):
        print(f"[skip] {task}: no source file at {src_path}", file=sys.stderr)
        return None

    with open(src_path, encoding="utf-8") as f:
        stream_data = json.load(f)

    out_records = [convert_fn(rec, video_prefix) for rec in stream_data]
    total_qa = sum(len(r["conversations"]) for r in out_records)

    out_path = os.path.join(out_dir, f"question_{task}_streamkv_online.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_records, f, ensure_ascii=False, indent=2)

    print(f"[ok]   {task}: {len(out_records)} videos x ~{total_qa // max(len(out_records), 1)} "
          f"QA ({total_qa} total) -> {out_path}")
    return out_path


# Per-task source filename + record converter. omni/real/sqa are multiple-choice
# (read questions_<task>_stream.json); proactive has no _stream variant (reads
# questions_proactive.json) and uses a different record schema.
TASK_SPECS = {
    "omni":      {},
    "real":      {},
    "sqa":       {},
    "proactive": {"src_filename": "questions_proactive.json",
                  "convert_fn": convert_proactive_record},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src_dir", default="data/streamingbench/StreamingBench/src/data",
                    help="dir containing questions_<task>_stream.json / questions_proactive.json")
    ap.add_argument("--out_dir", default="data/streamingbench",
                    help="dir to write question_<task>_streamkv_online.json")
    ap.add_argument("--video_prefix", default="data/streamingbench/StreamingBench/src/data/videos",
                    help="dir prefixed to each video_path basename so the output "
                         "uses repo-relative paths (default: data/streamingbench/StreamingBench/src/data/videos). "
                         "Pass '' to keep the source path verbatim, or another dir to repoint.")
    ap.add_argument("--tasks", nargs="*", default=["omni", "real", "sqa", "proactive"],
                    help="which tasks to convert")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for task in args.tasks:
        spec = TASK_SPECS.get(task, {})
        convert_task(args.src_dir, args.out_dir, task, args.video_prefix, **spec)


if __name__ == "__main__":
    main()
