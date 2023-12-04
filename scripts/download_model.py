from os import getenv

from huggingface_hub import hf_hub_download

files_list = {
    "liuhaotian/llava-v1.5-7b": [
        "config.json",
        "generation_config.json",
        "mm_projector.bin",
        "pytorch_model-00001-of-00002.bin",
        "pytorch_model-00002-of-00002.bin",
        "pytorch_model.bin.index.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "tokenizer_config.json",
    ]
}

if __name__ == "__main__":
    model_path = getenv("MODEL_PATH")
    if model_path not in files_list:
        raise ValueError(f"Unknown model path: {model_path}")
    for file in files_list[model_path]:
        hf_hub_download(
            repo_id=model_path,
            filename=file,
            subfolder=None,
        )
