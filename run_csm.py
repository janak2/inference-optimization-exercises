import os

# Disable Triton compilation - must be set before importing moshi
os.environ["NO_TORCH_COMPILE"] = "1"


from pathlib import Path

import torch
import torchaudio
from data.SPEAKER_PROMPTS import SPEAKER_PROMPTS
from generator import Generator, Model
from huggingface_hub import snapshot_download


def main():
    # Select the best available device, skipping MPS due to float64 limitations
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load model
    model_path = snapshot_download(
        "sesame/csm-1b",
        local_dir=Path(__file__).parent / "models",
        allow_patterns=["config.json", "model.safetensors"],
    )
    model = Model.from_pretrained(
        model_path,
    )
    model.to(device=device, dtype=torch.bfloat16)

    generator = Generator(model)

    # Generate conversation
    conversation = [
        {"text": "Hey how are you doing?", "speaker_id": 0},
        {"text": "Pretty good, pretty good. How about you?", "speaker_id": 1},
        {"text": "I'm great! So happy to be speaking with you today.", "speaker_id": 0},
        {"text": "Me too! This is some cool stuff, isn't it?", "speaker_id": 1},
    ]

    # Generate each utterance
    generated_segments = []

    for utterance in conversation:
        print(f"Generating: {utterance['text']}")
        audio_tensor = generator.generate(
            text=utterance["text"],
            speaker_audio_path=SPEAKER_PROMPTS["conversational_a"]["audio"],
            speaker_text=SPEAKER_PROMPTS["conversational_a"]["text"],
            max_audio_length_ms=10_000,
        )
        generated_segments.append(audio_tensor)

    # Concatenate all generations
    all_audio = torch.cat(generated_segments, dim=0)
    torchaudio.save(
        "full_conversation.wav", all_audio.unsqueeze(0).cpu(), generator.sample_rate
    )
    print("Successfully generated full_conversation.wav")


if __name__ == "__main__":
    main()
