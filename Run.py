
import json
import os
from pathlib import Path

# استيراد الدوال من الملفات الأخرى
from transcription import transcribe_audio

def main():

    audio_file = Path("audio/test.wav")
    output_dir = Path("output")

    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_file.is_file():
        print("Audio file not found!")
        return

    print("1 - Transcription")
    print("2 - Speaker Diarization")

    choice = input("Choose an option: ")

    if choice == "1":

        text = transcribe_audio(audio_file)

        output_file = output_dir / "transcription.txt"

        output_file.write_text(text, encoding="utf-8")

        print("Transcription completed!")
        print(text)

    elif choice == "2":
        from diarization import diarize_audio

        token = os.getenv("HF_TOKEN")

        if not token:
            print("HuggingFace token is missing!")
            return

        segments = diarize_audio(
            audio_file,
            token,
            min_speakers=2,
            max_speakers=2
        )

        json_file = output_dir / "diarization.json"

        with open(json_file, "w", encoding="utf-8") as file:
            json.dump(
                segments,
                file,
                ensure_ascii=False,
                indent=2
            )

        lines = []

        for segment in segments:
            start = float(segment.get("start", 0))
            end = float(segment.get("end", 0))
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "")

            lines.append(
                f"[{start:.2f} - {end:.2f}] {speaker}: {text}"
            )

        transcript_file = output_dir / "diarized_transcript.txt"

        transcript_file.write_text(
            "\n".join(lines),
            encoding="utf-8"
        )

        print("Diarization completed!")

    else:
        print("Invalid choice!")


# نقطة بداية البرنامج
if __name__ == "__main__":
    main()
