
from pathlib import Path
import torch
import whisperx


def diarize_audio(
    file_path,
    hf_token,
    min_speakers=2,
    max_speakers=2
):

    audio_path = Path(file_path)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    if not hf_token:
        raise ValueError("HuggingFace token is required")

    # تحديد جهاز التشغيل
    device = "cuda" if torch.cuda.is_available() else "cpu"

    compute_type = "float16" if device == "cuda" else "float32"

    # تحميل نموذج WhisperX
    model = whisperx.load_model(
        "small",
        device,
        compute_type=compute_type
    )

    # قراءة الصوت
    audio = whisperx.load_audio(str(audio_path))

    # تحويل الصوت إلى نص
    result = model.transcribe(
        audio,
        batch_size=16 if device == "cuda" else 8
    )

    language = result.get("language", "en")

    # محاذاة النص مع توقيت الكلمات
    align_model, metadata = whisperx.load_align_model(
        language_code=language,
        device=device
    )

    result = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False
    )

    # تحميل نموذج تحديد المتحدثين
    diarizer = whisperx.DiarizationPipeline(
        model_name="pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
        device=device
    )

    # تحديد عدد المتحدثين
    options = {
        "min_speakers": min_speakers,
        "max_speakers": max_speakers
    }

    diarization_segments = diarizer(audio, **options)

    # ربط كل كلمة بالمتحدث
    result = whisperx.assign_word_speakers(
        diarization_segments,
        result
    )

    return result["segments"]
