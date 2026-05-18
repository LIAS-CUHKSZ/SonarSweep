# pip install -U huggingface_hub
# you can also download from https://huggingface.co/datasets/Lingpenghaha/Sonarsweep_dataset/tree/main

# download test.txt and vfov12hfov60_test/ under data folder

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Lingpenghaha/Sonarsweep_dataset",
    repo_type="dataset",
    allow_patterns=[
        "vfov12hfov60_test/**",
        "test.txt",
    ],
    local_dir="data"
)
