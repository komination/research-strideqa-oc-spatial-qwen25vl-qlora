# save as scripts/download_stride_mini.py
from huggingface_hub import snapshot_download
import os

local_dir = "./data/stride-qa-mini"
os.makedirs(local_dir, exist_ok=True)

snapshot_download(
    repo_id="turing-motors/STRIDE-QA-Mini",
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False,   # Docker/WSLでのシンボリックリンク問題を回避
    resume_download=True,           # 途中から再開
    allow_patterns=["*.json", "images/*"]  # 必要ファイルだけ
)
print("done:", local_dir)
