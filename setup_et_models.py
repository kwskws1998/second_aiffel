import argparse
import os
import shutil
from urllib.request import Request, urlopen


DEFAULT_ET2_REPO = "skboy/et_prediction_2"
DEFAULT_ET2_FILENAME = "et_predictor2_seed123.safetensors"


def _resolve_local_checkpoint(checkpoint_path):
    for ext in ["", ".safetensors", ".pt", ".bin"]:
        candidate = checkpoint_path if checkpoint_path.endswith(ext) else checkpoint_path + ext
        if os.path.isfile(candidate):
            return candidate
    return None


def _download_file(url, destination_path):
    os.makedirs(os.path.dirname(destination_path) or ".", exist_ok=True)
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
    )
    with urlopen(req, timeout=120) as response, open(destination_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)
    return destination_path


def resolve_or_download_et2_checkpoint(
    checkpoint_path,
    hf_repo_id=DEFAULT_ET2_REPO,
    hf_filename=DEFAULT_ET2_FILENAME,
    auto_download=True,
):
    resolved = _resolve_local_checkpoint(checkpoint_path)
    if resolved:
        return os.path.abspath(resolved)

    if not auto_download:
        raise FileNotFoundError(f"Missing ET2 checkpoint: {checkpoint_path}[.safetensors/.pt/.bin]")

    destination = checkpoint_path if checkpoint_path.endswith((".safetensors", ".pt", ".bin")) else checkpoint_path + ".safetensors"
    url = f"https://huggingface.co/{hf_repo_id}/resolve/main/{hf_filename}"
    print(f"[setup_et_models] downloading ET2 checkpoint: {url}")
    _download_file(url, destination)

    resolved = _resolve_local_checkpoint(checkpoint_path)
    if not resolved:
        raise FileNotFoundError(f"Downloaded ET2 checkpoint but could not resolve: {checkpoint_path}")
    return os.path.abspath(resolved)


def write_env_file(resolved_path, env_path=".env_et"):
    line = f"ET2_CHECKPOINT_PATH={resolved_path}"
    with open(env_path, "w") as output_file:
        output_file.write(line + "\n")
    return line


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--et2-checkpoint", default="./checkpoints/et_predictor2_seed123")
    parser.add_argument("--et2-hf-repo", default=DEFAULT_ET2_REPO)
    parser.add_argument("--et2-hf-filename", default=DEFAULT_ET2_FILENAME)
    parser.add_argument("--no-et2-auto-download", action="store_true")
    args = parser.parse_args()

    resolved = resolve_or_download_et2_checkpoint(
        args.et2_checkpoint,
        hf_repo_id=args.et2_hf_repo,
        hf_filename=args.et2_hf_filename,
        auto_download=not args.no_et2_auto_download,
    )
    env_line = write_env_file(resolved)
    size_mb = os.path.getsize(resolved) / 1e6
    print(f"[setup_et_models] ET2 checkpoint ready: {resolved} ({size_mb:.1f} MB)")
    print(f"[setup_et_models] wrote .env_et: {env_line}")
    print("[setup_et_models] optional: source .env_et")


if __name__ == "__main__":
    main()
