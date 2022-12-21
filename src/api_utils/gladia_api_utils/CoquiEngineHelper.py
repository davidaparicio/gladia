import os
import pathlib
import sys
import wave
from io import BytesIO
from logging import getLogger

import ffmpeg
import numpy as np
from gladia_api_utils.model_management import download_model
from stt import Model

logger = getLogger(__name__)


class SpeechToTextEngine:
    """
    Speech to text engine from Coqui

    Args:
        model_uri (str): model uri
        model (str): model file name
        scorer (str): scorer file name

    Returns:
        SpeechToTextEngine: speech to text engine instance
    """

    def __init__(
        self,
        model_uri: str = "english/coqui/v1.0.0-huge-vocab",
        model: str = "model.tflite",
        scorer: str = "huge-vocabulary.scorer",
    ):
        """
        Initialize the engine

        Args:
            model_uri (str): model uri
            model (str): model file name
            scorer (str): scorer file name

        Returns:
            SpeechToTextEngine: speech to text engine instance
        """

        # the main path of the application calling this Helper by default /app/apis/"
        app_prefix = str(
            pathlib.Path(sys._getframe(1).f_globals["__file__"]).parents[0].absolute()
        )
        app_prefix = "/".join(app_prefix[1:].split("/")[:3])

        model_path_prefix = os.path.join(
            os.getenv("GLADIA_MODEL_PATH", "/gladia/models"),
            app_prefix,
            "audio",
            "text",
            model_uri,
        )

        logger.debug(f"model_path_prefix: {model_path_prefix}")

        model_url = f"https://coqui.gateway.scarf.sh/{model_uri}"

        logger.debug(f"Downloading model from {model_url} to {model_path_prefix}")

        model_path = download_model(
            url=f"{model_url}/{model}",
            output_path=os.path.join(model_path_prefix, "model"),
            uncompress_after_download=False,
        )

        scorer_path = download_model(
            url=f"{model_url}/{scorer}",
            output_path=os.path.join(model_path_prefix, "scorer"),
            uncompress_after_download=False,
        )

        self.model = Model(model_path)
        self.model.enableExternalScorer(scorer_path)

    def normalize_audio(self, audio: bytes) -> bytes:
        """
        Normalize audio to [-1, 1]

        Args:
            audio (bytes): audio to normalize

        Returns:
            bytes: normalized audio

        Raises:
            Exception: if audio normalization fails
        """

        out, err = (
            ffmpeg.input("pipe:0")
            .output(
                "pipe:1",
                f="WAV",
                acodec="pcm_s16le",
                ac=1,
                ar="16k",
                loglevel="error",
                hide_banner=None,
            )
            .run(input=audio, capture_stdout=True, capture_stderr=True)
        )
        if err:
            raise SystemError(err)
        return out

    def run(self, audio: bytes) -> str:
        """
        Run the model on audio

        Args:
            audio (bytes): audio to run the model on

        Returns:
            str: transcription
        """

        audio = self.normalize_audio(audio)
        audio = BytesIO(audio)

        with wave.Wave_read(audio) as wav:
            audio = np.frombuffer(wav.readframes(wav.getnframes()), np.int16)
        result = self.model.stt(audio)

        return result
