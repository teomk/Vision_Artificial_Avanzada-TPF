from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="saved_models/lama_no_sar_finetuned.pth",
    path_in_repo="lama_no_sar_finetuned_v1.pth",
    repo_id="LucioLuque/lama_no_sar",
    repo_type="model",
)