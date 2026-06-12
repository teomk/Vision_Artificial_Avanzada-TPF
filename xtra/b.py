from huggingface_hub import HfApi

api = HfApi()

api.upload_large_folder(
    repo_id="LucioLuque/sen12mscr-south-america",
    repo_type="dataset",
    folder_path="data",
    num_workers=4,
    allow_patterns="**/*.tif",
)