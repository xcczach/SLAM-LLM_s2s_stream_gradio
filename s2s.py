import random
import torch
from slam_llm.utils.model_utils import get_custom_model_factory
from utils.snac_utils import reconscruct_snac, reconstruct_tensors, layershift
import whisper
import numpy as np
from s2s_config import InferenceConfig, CKPT_PATH, CKPT_REPO, CKPT_LOCAL_DIR, CKPT_NAME
import os
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from typing import Callable


def update_progress(progress_callback: Callable[[str], None] | None, message: str):
    if progress_callback:
        progress_callback(message)


def pull_model_ckpt():
    if not os.path.exists(CKPT_LOCAL_DIR):
        os.makedirs(CKPT_LOCAL_DIR)
    if os.path.exists(CKPT_PATH):
        return
    hf_hub_download(
        repo_id=CKPT_REPO,
        filename=CKPT_NAME,
        local_dir=CKPT_LOCAL_DIR,
        token=os.getenv("HF_TOKEN"),
    )


pull_model_ckpt()


def extract_audio_feature(audio_path, mel_size):
    print("Extracting audio features from", audio_path)
    audio_raw = whisper.load_audio(audio_path)
    audio_raw = whisper.pad_or_trim(audio_raw)
    audio_mel = whisper.log_mel_spectrogram(audio_raw, n_mels=mel_size).permute(1, 0)
    audio_length = (audio_mel.shape[0] + 1) // 2
    audio_length = audio_length // 5
    audio_res = audio_mel

    return audio_res, audio_length


def get_input_ids(length, special_token_a, special_token_t, vocab_config):
    input_ids = []
    for i in range(vocab_config.code_layer):
        input_ids_item = []
        input_ids_item.append(layershift(vocab_config.input_a, i))
        input_ids_item += [layershift(vocab_config.pad_a, i)] * length
        input_ids_item += [
            (layershift(vocab_config.eoa, i)),
            layershift(special_token_a, i),
        ]
        input_ids.append(torch.tensor(input_ids_item).unsqueeze(0))
    input_id_T = torch.tensor(
        [vocab_config.input_t]
        + [vocab_config.pad_t] * length
        + [vocab_config.eot, special_token_t]
    )
    input_ids.append(input_id_T.unsqueeze(0))
    return input_ids


def generate_from_wav(
    wav_path, model, codec_decoder, dataset_config, decode_config, device
):
    mel_size = dataset_config.mel_size
    prompt = dataset_config.prompt
    prompt_template = "USER: {}\n ASSISTANT: "
    vocab_config = dataset_config.vocab_config
    special_token_a = vocab_config.answer_a
    special_token_t = vocab_config.answer_t
    code_layer = vocab_config.code_layer
    task_type = dataset_config.task_type

    audio_mel, audio_length = extract_audio_feature(wav_path, mel_size)

    prompt = prompt_template.format(prompt)
    prompt_ids = model.tokenizer.encode(prompt)
    prompt_length = len(prompt_ids)
    prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64)

    example_ids = get_input_ids(
        audio_length + prompt_length, special_token_a, special_token_t, vocab_config
    )
    text_layer = example_ids[code_layer]
    text_layer = torch.cat(
        (
            text_layer[:, : audio_length + 1],
            prompt_ids.unsqueeze(0),
            text_layer[:, -2:],
        ),
        dim=1,
    )  # <bos> <audio> <prompt> <eos> <task>
    example_ids[code_layer] = text_layer

    input_length = audio_length
    example_mask = example_ids[0][0].ge(-1)
    example_ids = torch.stack(example_ids).squeeze()

    input_ids = example_ids.unsqueeze(0).to(device)
    attention_mask = example_mask.unsqueeze(0).to(device)
    audio_mel = audio_mel.unsqueeze(0).to(device)
    input_length = torch.tensor([input_length]).to(device)
    audio_length = torch.tensor([audio_length]).to(device)
    task_type = [task_type]

    modality_mask = torch.zeros_like(attention_mask)
    padding_left = 1  # +1 for <bos>
    modality_mask[0, padding_left : padding_left + audio_length] = True

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "audio_mel": audio_mel,
        "input_length": input_length,
        "audio_length": audio_length,
        "modality_mask": modality_mask,
        "task_types": task_type,
    }

    model_outputs = model.generate(**batch, **decode_config)
    text_outputs = model_outputs[7]
    audio_outputs = model_outputs[:7]
    output_text = model.tokenizer.decode(
        text_outputs, add_special_tokens=False, skip_special_tokens=True
    )

    if decode_config.decode_text_only:
        return None, output_text

    audio_tokens = [audio_outputs[layer] for layer in range(7)]
    audiolist = reconscruct_snac(audio_tokens)
    audio = reconstruct_tensors(audiolist)
    with torch.inference_mode():
        audio_hat = codec_decoder.decode(audio)

    return audio_hat, output_text


def generate(
    wav_path: str, progress_callback: Callable[[str], None] | None = None
) -> tuple[np.ndarray, int | float]:
    config = OmegaConf.structured(InferenceConfig())
    train_config, model_config, dataset_config, decode_config = (
        config.train_config,
        config.model_config,
        config.dataset_config,
        config.decode_config,
    )

    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    update_progress(progress_callback, "Loading model")

    model_factory = get_custom_model_factory(model_config)
    model, _ = model_factory(train_config, model_config, CKPT_PATH)
    codec_decoder = model.codec_decoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    update_progress(progress_callback, "Generating")
    output_wav, output_text = generate_from_wav(
        wav_path, model, codec_decoder, dataset_config, decode_config, device
    )

    return output_wav.squeeze().cpu().numpy(), 24000


if __name__ == "__main__":
    wav_path = "sample.wav"
    generate(wav_path)
