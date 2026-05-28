import os
import sys
from pathlib import Path


def _log_tree(path: Path, prefix: str = "", max_depth: int = 2, _depth: int = 0) -> None:
    if _depth > max_depth:
        return
    if not path.exists():
        print(f"{prefix}(does not exist)")
        return
    if path.is_file():
        print(f"{prefix}{path.name} ({path.stat().st_size})")
        return
    try:
        children = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        print(f"{prefix}(permission denied)")
        return
    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        branch = "└── " if is_last else "├── "
        print(f"{prefix}{branch}{child.name}")
        if child.is_dir() and _depth < max_depth:
            next_prefix = prefix + ("    " if is_last else "│   ")
            _log_tree(child, next_prefix, max_depth, _depth + 1)


def _find_hf_cache_snapshot(repo_id: str, hf_home: Path | None = None) -> Path | None:
    """Find the on-disk snapshot directory for a cached HF repo."""
    if hf_home is None:
        hf_home = Path(os.environ.get("HF_HOME", "/runpod-volume/huggingface-cache/hub"))
    sanitized = repo_id.replace("/", "--")
    repo_cache = hf_home / f"models--{sanitized}"
    if not repo_cache.exists():
        target_name = f"models--{sanitized}"
        for child in hf_home.iterdir():
            if child.is_dir() and child.name.lower() == target_name.lower():
                repo_cache = child
                break
        else:
            return None

    refs_dir = repo_cache / "refs"
    snapshots_dir = repo_cache / "snapshots"
    if not refs_dir.exists() or not snapshots_dir.exists():
        return None

    for ref_file in refs_dir.iterdir():
        commit_hash = ref_file.read_text().strip()
        snapshot = snapshots_dir / commit_hash
        if snapshot.exists():
            return snapshot

    for child in snapshots_dir.iterdir():
        if child.is_dir():
            return child
    return None


def setup_lance_models() -> None:
    """Bridge RunPod's HF Model Cache to Lance's expected model layout."""
    print("=" * 60)
    print("[Setup] Starting Lance model setup from HF cache")
    print("=" * 60)

    hf_home = Path(os.environ.get("HF_HOME", "/runpod-volume/huggingface-cache/hub"))
    checkpoint_dir = Path(os.environ.get("LANCE_MODEL_BASE_DIR", "/runpod-volume/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Symlink /app/downloads -> checkpoint_dir so the Lance library's hardcoded
    # "downloads/" relative paths resolve to the model cache
    downloads_link = Path("/app/downloads")
    if not downloads_link.exists():
        downloads_link.symlink_to(checkpoint_dir)
        print(f"[Setup] Created symlink: /app/downloads -> {checkpoint_dir}")

    repo_id = os.environ.get("LANCE_MODEL_REPO", "bytedance-research/Lance")

    print(f"[Setup] LANCE_MODEL_BASE_DIR = {checkpoint_dir}")
    print(f"[Setup] HF_HOME = {hf_home}")
    print(f"[Setup] LANCE_MODEL_REPO = {repo_id}")

    # Log /runpod-volume
    runpod_vol = Path("/runpod-volume")
    if runpod_vol.exists():
        print("\n[Setup] /runpod-volume contents:")
        _log_tree(runpod_vol, max_depth=2)
    else:
        print("\n[Setup] /runpod-volume does NOT exist")

    # Log HF cache tree
    if hf_home.exists():
        print(f"\n[Setup] HF cache tree ({hf_home}):")
        _log_tree(hf_home, max_depth=2)
    else:
        print(f"\n[Setup] HF cache dir does NOT exist: {hf_home}")

    # Find cached snapshot
    snapshot = _find_hf_cache_snapshot(repo_id, hf_home)
    if snapshot is None:
        print(f"\n[Setup] WARNING: No HF cache snapshot found for {repo_id}")
        print(f"[Setup] Checked: {hf_home / ('models--' + repo_id.replace('/', '--'))}")
        print("[Setup] Lance models must be available via RunPod Model Cache.")
        return

    print(f"\n[Setup] Found HF cache snapshot: {snapshot}")
    print("[Setup] Snapshot contents:")
    _log_tree(snapshot, max_depth=1)

    # Symlink each subdirectory/file from snapshot into checkpoint_dir
    print(f"\n[Setup] Linking snapshot contents to {checkpoint_dir}:")
    for child in snapshot.iterdir():
        dst = checkpoint_dir / child.name
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink():
                dst.unlink()
            elif dst.is_dir():
                print(f"[Setup] SKIP: directory {child.name} already exists")
                continue
            else:
                dst.unlink()
        try:
            dst.symlink_to(child)
            print(f"[Setup] LINKED {dst.name} -> {child}")
        except OSError as exc:
            print(f"[Setup] ERROR linking {child.name}: {exc}")

    # Validate expected model assets
    expected = [
        checkpoint_dir / "Lance_3B",
        checkpoint_dir / "Lance_3B_Video",
        checkpoint_dir / "Qwen2.5-VL-ViT",
        checkpoint_dir / "Wan2.2_VAE.pth",
    ]
    print("\n[Setup] Validating expected model assets:")
    all_ok = True
    for path in expected:
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {path.name}")
        if not exists:
            all_ok = False

    if not all_ok:
        print("\n[Setup] WARNING: Some model assets are missing!")
    else:
        print("\n[Setup] All model assets present.")

    # Lance_3B_Video in the HF repo lacks llm_config.json — borrow from Lance_3B.
    video_dir = checkpoint_dir / "Lance_3B_Video"
    image_dir = checkpoint_dir / "Lance_3B"
    if video_dir.exists() and image_dir.exists():
        for cfg_name in ("llm_config.json", "generation_config.json",
                         "vocab.json", "merges.txt", "tokenizer.json"):
            video_cfg = video_dir / cfg_name
            image_cfg = image_dir / cfg_name
            if not video_cfg.exists() and image_cfg.exists():
                video_cfg.symlink_to(image_cfg)
                print(f"[Setup] LINKED {cfg_name} -> {image_cfg} (video model borrowed from image)")

    print("=" * 60)


if __name__ == "__main__":
    setup_lance_models()
