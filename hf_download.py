from huggingface_hub import snapshot_download
import os

models = [
    'minchul/cvlface_adaface_vit_base_kprpe_webface12m',
    'minchul/cvlface_adaface_vit_base_kprpe_webface4m',
    'minchul/cvlface_adaface_vit_base_webface4m',
    'minchul/cvlface_adaface_ir101_webface12m',
    'minchul/cvlface_adaface_ir101_webface4m',
    'minchul/cvlface_adaface_ir101_ms1mv3',
    'minchul/cvlface_adaface_ir101_ms1mv2'
]

for model in models:
    model_name = model.split('/')[-1]
    local_dir = f'/mnt/data/CVLface/cvlface/pretrained_models/recognition/{model_name}'
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(repo_id=model, local_dir=local_dir)
    print(f'Downloaded {model} to {local_dir}')