#!/usr/bin/env python3
import sys
import os
import json
from typing import Any, Dict, List
import whisper

# convert the timestamp received natively from whisper to srt format
def fmt_timestamp(seconds: float) -> str:
    if seconds is None:
        raise ValueError("Timestamp value is None")
    try:
        sec_float = float(seconds)
    except Exception as exc:
        raise ValueError(f"Invalid timestamp '{seconds}': {exc}")

    hrs = int(sec_float // 3600)
    mins = int((sec_float % 3600) // 60)
    secs = int(sec_float % 60)
    millis = int(round((sec_float - int(sec_float)) * 1000))

    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"

# extract the start end and text info from the whisper output
def extract_segment_info(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    result = []
    for seg in data.get("segments", []):
        result.append({
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": seg.get("text", "").strip(),
            }
        )
    return result

# create the srt file
def write_srt(segments: List[Dict[str, Any]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, start=1):
            start_ts = fmt_timestamp(seg["start"])
            end_ts = fmt_timestamp(seg["end"])
            text = seg["text"]
            f.write(f"{idx}\n")
            f.write(f"{start_ts} --> {end_ts}\n")
            f.write(f"{text}\n\n")


# transcribe using whisper
def main(input_path: str) -> None:
    model = whisper.load_model("base")
    print(f"Transcribing '{input_path}' …")
    result = model.transcribe(input_path, task="transcribe")
    segments = extract_segment_info(result)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    srt_path = f"{base_name}.srt"
    write_srt(segments, srt_path)

    print(f"SRT file successfully written to: {srt_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        media_file = sys.argv[1]
    else:
        media_file = "short_weather_news.webm"

    if not os.path.isfile(media_file):
        sys.exit(f"File not found: {media_file}")

    main(media_file)
