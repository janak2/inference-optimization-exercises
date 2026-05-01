from typing import Tuple

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from models import Model
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer


def load_llama3_tokenizer():
    """
    https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
    """
    tokenizer_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[
            (f"{bos}", tokenizer.bos_token_id),
            (f"{eos}", tokenizer.eos_token_id),
        ],
    )

    return tokenizer


class Generator:
    def __init__(
        self,
        model: Model,
    ):
        self._model = model
        self._model.setup_caches(1)

        self._text_tokenizer = load_llama3_tokenizer()

        device = next(model.parameters()).device
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        mimi.set_num_codebooks(32)
        self._audio_tokenizer = mimi

        self.sample_rate = mimi.sample_rate
        self.device = device

    def _tokenize_text_segment(
        self, text: str, speaker: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()  # [L, 33]
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()  # [L, 33]
        text_frame[:, -1] = torch.tensor(text_tokens)  # [L, 33]
        text_frame_mask[:, -1] = True  # [L, 33]

        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))

        return torch.cat(frame_tokens, dim=0), torch.cat(
            frame_masks, dim=0
        )  # [L, 33], [L, 33]

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert audio.ndim == 1, "Audio must be single channel"

        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio = audio.to(self.device)  # [T]
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[
            0
        ]  # [32, T*12.5/24000]
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)  # [32, L']

        audio_frame = (
            torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
        )  # [L', 33]
        audio_frame_mask = (
            torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
        )  # [L', 33]
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)  # [L', 33]
        audio_frame_mask[:, :-1] = True  # [L', 33]

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker_audio_path: str,
        speaker_text: str,
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
    ) -> torch.Tensor:
        self._model.reset_caches()

        audio_tensor, sample_rate = torchaudio.load(speaker_audio_path)
        audio_tensor = audio_tensor.squeeze(0)
        # Resample is lazy so we can always call it
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, orig_freq=sample_rate, new_freq=self.sample_rate
        )

        max_generation_len = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []

        text_tokens, text_masks = self._tokenize_text_segment(
            speaker_text, 0
        )  # [L, 33], [L, 33]
        audio_tokens, audio_masks = self._tokenize_audio(
            audio_tensor
        )  # [L', 33], [L', 33]

        speaker_tokens, speaker_tokens_mask = (
            torch.cat([text_tokens, audio_tokens], dim=0),
            torch.cat([text_masks, audio_masks], dim=0),
        )  # [L+L', 33], [L+L', 33]
        tokens.append(speaker_tokens)
        tokens_mask.append(speaker_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(
            text, 0
        )  # [L', 33], [L', 33]
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)  # [L, 33]
        prompt_tokens_mask = (
            torch.cat(tokens_mask, dim=0).bool().to(self.device)
        )  # [L, 33]

        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)  # [1, L, 33]
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)  # [1, L, 33]
        curr_pos = (
            torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)
        )  # [1, L]

        max_seq_len = 2048
        max_context_len = max_seq_len - max_generation_len
        if curr_tokens.size(1) >= max_context_len:
            raise ValueError(
                f"Inputs too long, must be below max_seq_len - max_generation_len: {max_context_len}"
            )

        for _ in range(max_generation_len):
            sample = self._model.generate_frame(
                curr_tokens, curr_tokens_mask, curr_pos, temperature, topk
            )  # [1, 32]
            if torch.all(sample == 0):
                break  # eos

            samples.append(sample)

            curr_tokens = torch.cat(
                [sample, torch.zeros(1, 1).long().to(self.device)], dim=1
            ).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [
                    torch.ones_like(sample).bool(),
                    torch.zeros(1, 1).bool().to(self.device),
                ],
                dim=1,
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        audio = (
            self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0))
            .squeeze(0)
            .squeeze(0)
        )

        return audio
