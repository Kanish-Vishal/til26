"""Manages the ASR model."""

import os
from io import BytesIO

from faster_whisper import WhisperModel


class ASRManager:
    """Thin wrapper around a multilingual Whisper ASR model."""

    def __init__(self):
        model_size = os.getenv("ASR_MODEL_SIZE", "large-v3")
        device = os.getenv("ASR_DEVICE", self._default_device())
        compute_type = os.getenv(
            "ASR_COMPUTE_TYPE",
            "float16" if device == "cuda" else "int8",
        )
        cpu_threads = int(os.getenv("ASR_CPU_THREADS", "4"))
        num_workers = int(os.getenv("ASR_NUM_WORKERS", "1"))

        self.beam_size = int(os.getenv("ASR_BEAM_SIZE", "5"))
        self.vad_filter = os.getenv("ASR_VAD_FILTER", "false").lower() == "true"
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )

    def asr(self, audio_bytes: bytes) -> str:
        """Performs ASR transcription on an audio file.

        Args:
            audio_bytes: The audio file in bytes.

        Returns:
            A string containing the transcription of the audio.
        """

        audio_file = BytesIO(audio_bytes)
        audio_file.name = "audio.wav"

        segments, _ = self.model.transcribe(
            audio_file,
            task="transcribe",
            beam_size=self.beam_size,
            best_of=5,
            temperature=[0.0, 0.2, 0.4],
            condition_on_previous_text=False,
            vad_filter=self.vad_filter,
        )

        return " ".join(segment.text.strip() for segment in segments).strip()

    @staticmethod
    def _default_device() -> str:
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda"
        except Exception:
            pass
        return "cpu"
