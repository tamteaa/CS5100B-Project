from enum import Enum
from typing import Protocol, TypeVar


class GroqModels(Enum):
    LLAMA_70B = "llama-3.1-70b-versatile"
    LLAMA_90B = "llama-3.2-90b-vision-preview"
    LLAMA_8B = "llama-3.1-8b-instant"
    GEMMA_7B = "gemma-7b-it"


class TogetherModels(Enum):
    MIXTRAL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    LLAMA_405B = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
    LLAMA31_8B = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    LLAMA3_70B = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    LLAMA31_8B_FINE_TUNED = "aarontamte/Meta-Llama-3.1-8B-Instruct-Reference-stabilized-finetune-v2-a2879a73-d860eb59"


class LocalModels(Enum):
    LLAMA_7B = "llama-2-7b-chat"
    MISTRAL_7B = "mistral-7b-instruct"
    NEURAL_7B = "neural-chat-7b"


class Provider(Enum):
    GROQ = ("groq", GroqModels)
    TOGETHER = ("together", TogetherModels)
    LOCAL = ("local", LocalModels)

    def __init__(self, value: str, models):
        self._value_ = value
        self.models = models

