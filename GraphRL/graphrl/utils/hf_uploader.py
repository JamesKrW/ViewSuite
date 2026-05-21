"""Per-iteration HuggingFace upload utility.

Configured via ``upload_to_hf`` in pipeline.yaml::

    upload_to_hf: [rl_model, sft_model, sft_data]

After each iteration, the controller calls :func:`upload_iter_resources`, which
walks ``iter_XXX/<name>`` for every entry and uploads it:

  * If the directory looks like an HF model (``config.json`` at top level),
    it goes to a **model** repo via ``HfApi.upload_folder``.
  * Otherwise it is packed into ``.tar.gz`` (pigz when available, stdlib
    fallback) and uploaded to a **dataset** repo (mirrors ViewSuite's
    ``upload_folder_to_hf_as_targz.py`` flow).

Defaults
--------
* Repo id = ``<project_name>`` (shared across experiments — the HF user that
  owns ``HF_TOKEN`` is the implicit owner). Override with
  ``upload_to_hf_repo_owner`` / ``upload_to_hf_model_repo`` /
  ``upload_to_hf_data_repo``.
* ``upload_to_hf_unified_repo: true`` (DEFAULT) bundles tarballs into the
  SAME model repo as the model dirs (``data_repo`` is ignored) so a single
  browsable HF page holds checkpoints + rollouts side by side. Set to
  ``false`` to restore the legacy split (model dirs → model repo,
  tarballs → dataset repo).
* path_in_repo = ``<experiment_name>/iter_XXX/<name>`` for models;
  ``<experiment_name>/iter_XXX/<name>.tar.gz`` for tarballs.
* ``HF_HUB_ENABLE_HF_TRANSFER`` is enabled if not already set.
* ``HF_TOKEN`` is read from the environment. If unset, a warning is logged and
  the call returns — pipeline continues without raising.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


def _looks_like_hf_model(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def _make_tarball(src: Path, dst: Path, *, level: int = 1, threads: int = 16) -> None:
    """Pack ``src`` (a directory) into ``dst`` (.tar.gz). Prefer pigz; fall back to stdlib."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    pigz = shutil.which("pigz")
    if pigz and shutil.which("tar"):
        with dst.open("wb") as out:
            tar = subprocess.Popen(
                ["tar", "-cf", "-", "-C", str(src.parent), src.name],
                stdout=subprocess.PIPE,
            )
            assert tar.stdout is not None
            gz = subprocess.Popen(
                [pigz, f"-{level}", "-p", str(threads)],
                stdin=tar.stdout, stdout=out,
            )
            tar.stdout.close()
            gz.communicate()
            tar.wait()
            if tar.returncode != 0 or gz.returncode != 0:
                raise RuntimeError(
                    f"pigz pack failed for {src} (tar={tar.returncode} pigz={gz.returncode})"
                )
        return
    import tarfile
    with tarfile.open(dst, "w:gz", compresslevel=level) as tf:
        tf.add(src, arcname=src.name)


def _resolve_repo_id(api, repo_id: str, repo_owner: Optional[str]) -> str:
    """If ``repo_id`` is unqualified, prepend an owner (explicit or token user)."""
    if "/" in repo_id:
        return repo_id
    if repo_owner:
        return f"{repo_owner}/{repo_id}"
    try:
        whoami = api.whoami()
    except Exception as exc:
        logger.warning("[upload_to_hf] whoami() failed: %s — using bare repo id %r", exc, repo_id)
        return repo_id
    owner = whoami.get("name")
    if owner:
        return f"{owner}/{repo_id}"
    orgs = whoami.get("orgs") or []
    if orgs and isinstance(orgs[0], dict) and orgs[0].get("name"):
        return f"{orgs[0]['name']}/{repo_id}"
    return repo_id


def upload_iter_resources(
    iter_dir: Path,
    iter_num: int,
    *,
    resources: Sequence[str],
    project_name: str,
    experiment_name: str,
    repo_owner: Optional[str] = None,
    model_repo: Optional[str] = None,
    data_repo: Optional[str] = None,
    visibility: str = "public",
    unified_repo: bool = True,
) -> None:
    """Upload per-iteration resources to HuggingFace. No-op if HF_TOKEN is unset.

    ``unified_repo`` (default True). When True, every resource — model
    dirs AND tarballed non-model dirs — lands in the same **model** repo
    (``model_repo`` / ``project_name``); the ``data_repo`` parameter is
    ignored. Tarballs are uploaded with ``repo_type="model"`` (HF model
    repos accept arbitrary blob files just fine; only the repo's public
    surface differs from a dataset repo). Set False to restore the legacy
    split (model dirs → model repo, tarballs → dataset repo).
    """
    resources = [r for r in (resources or []) if r]
    if not resources:
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.warning(
            "[upload_to_hf] HF_TOKEN not set — skipping upload of %s",
            list(resources),
        )
        return

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("[upload_to_hf] huggingface_hub not importable — skipping")
        return

    api = HfApi(token=token)
    private = str(visibility).strip().lower() in {"private", "true", "1", "yes", "y"}
    iter_tag = f"iter_{iter_num:03d}"

    model_repo_id = _resolve_repo_id(api, model_repo or project_name, repo_owner)
    if unified_repo:
        # Tarballs land in the SAME model repo as the model dirs. Useful when
        # you want a single HF page with checkpoints + rollouts side by side.
        data_repo_id = model_repo_id
        data_repo_type = "model"
    else:
        data_repo_id = _resolve_repo_id(api, data_repo or project_name, repo_owner)
        data_repo_type = "dataset"

    created_repos: Set[Tuple[str, str]] = set()

    def _ensure_repo(repo_id: str, repo_type: str) -> None:
        key = (repo_id, repo_type)
        if key in created_repos:
            return
        try:
            api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
        except Exception as exc:
            logger.warning("[upload_to_hf] create_repo %s (%s) failed: %s", repo_id, repo_type, exc)
        created_repos.add(key)

    for name in resources:
        local_path = iter_dir / name
        if not local_path.exists():
            logger.warning(
                "[upload_to_hf] %s not found at %s — skipping",
                name, local_path,
            )
            continue

        try:
            if _looks_like_hf_model(local_path):
                path_in_repo = f"{experiment_name}/{iter_tag}/{name}"
                logger.info(
                    "[upload_to_hf] %s → %s:%s (model)",
                    local_path, model_repo_id, path_in_repo,
                )
                _ensure_repo(model_repo_id, "model")
                api.upload_folder(
                    folder_path=str(local_path),
                    repo_id=model_repo_id,
                    repo_type="model",
                    path_in_repo=path_in_repo,
                    commit_message=f"upload {experiment_name}/{iter_tag}/{name}",
                )
            else:
                with tempfile.TemporaryDirectory() as td:
                    # Flatten any slashes in ``name`` so a literal nested path
                    # like ``rl/rollout_data`` writes to ``td/rl_rollout_data.tar.gz``
                    # instead of failing on the nonexistent ``td/rl/`` subdir.
                    safe_name = name.replace("/", "_").replace("\\", "_")
                    tar_path = Path(td) / f"{safe_name}.tar.gz"
                    logger.info("[upload_to_hf] packing %s → %s", local_path, tar_path)
                    _make_tarball(local_path, tar_path)
                    path_in_repo = f"{experiment_name}/{iter_tag}/{tar_path.name}"
                    logger.info(
                        "[upload_to_hf] %s → %s:%s (%s)",
                        tar_path, data_repo_id, path_in_repo, data_repo_type,
                    )
                    _ensure_repo(data_repo_id, data_repo_type)
                    api.upload_file(
                        path_or_fileobj=str(tar_path),
                        path_in_repo=path_in_repo,
                        repo_id=data_repo_id,
                        repo_type=data_repo_type,
                        commit_message=f"upload {experiment_name}/{iter_tag}/{name}",
                    )
            logger.info("[upload_to_hf] done: %s", name)
        except Exception as exc:
            logger.warning("[upload_to_hf] failed for %s: %s", name, exc, exc_info=True)
