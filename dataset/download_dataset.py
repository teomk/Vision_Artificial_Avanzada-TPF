from huggingface_hub import snapshot_download

#run --> python dataset/download_data.py

snapshot_download(
    repo_id="LucioLuque/sen12mscr-south-america",
    repo_type="dataset",
    local_dir="data",
    allow_patterns=["*.tif", "**/*.tif"],
    max_workers=8,
)