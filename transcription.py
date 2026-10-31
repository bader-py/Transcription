
from pathlib import Path  # التعامل مع مسارات الملفات.
import torch     # 	تشغيل النماذج والتع مع GPU أو CPU.امل
import librosa
from transformers import pipeline  # نموذج  تحويل الصوت الى نص

# انشاء داله لنستخدمها من اي ملف ثاني
def transcribe_audio(file_path, language=None):

    audio_path = Path(file_path)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    # تحديد جهاز التشغيل
    device = 0 if torch.cuda.is_available() else -1

    # قراءة الصوت وتحويله إلى 16000 Hz
    audio, sample_rate = librosa.load(
        str(audio_path),
        sr=16000,
        mono=True
    )

    # تحميل نموذج Whisper
    model = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device=device
    )

    # تحديد مهمة التفريغ
    generate_kwargs = {"task": "transcribe"}

    if language:
        generate_kwargs["language"] = language

    # استخراج النص
    result = model(
        {"raw": audio, "sampling_rate": sample_rate},
        generate_kwargs=generate_kwargs
    )

    return result["text"].strip()
