from fastapi import APIRouter
from gladia_api_utils.submodules import TaskRouter
from gladia_api_utils.task_management import get_task_metadata

task_metadata = get_task_metadata(__file__)

inputs = [
    {
        "type": task_metadata["inputs"]["audio_url"]["type"],
        "name": "audio",
        "example": task_metadata["inputs"]["audio_url"]["examples"][0],
        "examples": task_metadata["inputs"]["audio_url"]["examples"],
        "placeholder": "Audio to transcribe",
    },
    {
        "type": task_metadata["inputs"]["language"]["type"],
        "name": "language",
        "default": task_metadata["inputs"]["language"]["examples"][0],
        "example": task_metadata["inputs"]["language"]["examples"][0],
        "examples": task_metadata["inputs"]["language"]["examples"],
        "placeholder": "Language of the audio",
    },
]

output = {
    "name": "transcription",
    "type": "string",
    "example": "I'm telling you that this is the tools i've seen so far.",
}

router = APIRouter()

TaskRouter(
    router=router, input=inputs, output=output, default_model="openai-whisper-tiny"
)
