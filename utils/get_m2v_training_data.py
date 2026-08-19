"""
get_m2v_training_data.py - download model2vec classifier training data

Downloads training data from GitHub release if not present.

Institutional Books - Enriched Text - 2026
"""

import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import requests
from loguru import logger

JSON = dict[str, Any]

# Default paths
DEFAULT_TRAINING_DATA_DIR = Path("DATA/release_assets/endmatter_training_data")
TOPCLASSIFIER_FILE = "m2v_training_data_topclassifier.jsonl"
SUBCLASSIFIER_FILE = "m2v_training_data_subclassifier.jsonl"
REQUIRED_FILES = (TOPCLASSIFIER_FILE, SUBCLASSIFIER_FILE)

# GitHub release source
GITHUB_REPO = "institutional/institutional-books-enriched-text-pipeline"
# None => resolve the most recent release (including prereleases).
# We may optionally pin to a tag in the future.
DEFAULT_RELEASE_TAG: str | None = None
# Asset extensions treated as the training-data tarball.
_TARBALL_SUFFIXES = (".tar", ".tar.gz", ".tgz")

_GITHUB_API = "https://api.github.com"
_DOWNLOAD_TIMEOUT = 100  # seconds, generous per request


def _github_headers(accept: str = "application/vnd.github+json") -> dict[str, str]:
    """Build request headers, adding bearer auth if a GitHub token is present."""
    headers = {"Accept": accept, "X-GitHub-Api-Version": "2022-11-28"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _resolve_release(repo: str, tag: str | None) -> JSON:
    """
    Fetch the release JSON for ``tag``, or the most recent release when ``tag``
    is None. "Most recent" includes prereleases (which GitHub's ``/latest``
    endpoint skips) but excludes drafts.
    """
    if tag:
        url = f"{_GITHUB_API}/repos/{repo}/releases/tags/{tag}"
        resp = requests.get(url, headers=_github_headers(), timeout=_DOWNLOAD_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    # No tag pinned: list releases and pick the newest published one.
    url = f"{_GITHUB_API}/repos/{repo}/releases"
    resp = requests.get(
        url,
        headers=_github_headers(),
        params={"per_page": 100},
        timeout=_DOWNLOAD_TIMEOUT,
    )
    resp.raise_for_status()
    releases = [r for r in resp.json() if not r.get("draft")]
    if not releases:
        raise RuntimeError(f"No published releases found for {repo}.")
    releases.sort(
        key=lambda r: r.get("published_at") or r.get("created_at") or "",
        reverse=True,
    )
    return releases[0]


def _pick_tarball_asset(release: JSON) -> JSON:
    """Select the training-data tarball asset from a release."""
    assets = release.get("assets", [])
    tarballs = [a for a in assets if a.get("name", "").lower().endswith(_TARBALL_SUFFIXES)]
    if not tarballs:
        names = [a.get("name") for a in assets]
        raise RuntimeError(
            f"No tarball asset ({', '.join(_TARBALL_SUFFIXES)}) found in release "
            f"{release.get('tag_name')!r}. Available assets: {names}"
        )
    if len(tarballs) > 1:
        logger.warning(
            f"Multiple tarball assets in release {release.get('tag_name')!r}; "
            f"using first: {[a['name'] for a in tarballs]}"
        )
    return tarballs[0]


def _download_asset(asset: JSON, dest: Path) -> None:
    """Stream a release asset to ``dest``.

    Uses the API asset URL with an octet-stream Accept header so downloads work
    for private repos when a token is set; anonymous public downloads work too.
    """
    # asset["url"] is the API endpoint; requesting octet-stream returns the
    # binary (or a redirect to the CDN, which requests follows automatically).
    url = asset["url"]
    headers = _github_headers(accept="application/octet-stream")
    size = asset.get("size")
    size_str = f" ({size / 1e6:.1f} MB)" if size else ""
    logger.info(f"Downloading {asset.get('name')}{size_str}...")
    with requests.get(url, headers=headers, stream=True, timeout=_DOWNLOAD_TIMEOUT) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)


def _extract_training_files(tar_path: Path, output_dir: Path) -> list[str]:
    """
    Extract the required ``.jsonl`` files from ``tar_path`` into ``output_dir``,
    flattening any leading directory in the archive. Returns the basenames
    written. Guards against path traversal by only writing known basenames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(REQUIRED_FILES)
    written: list[str] = []
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            basename = os.path.basename(member.name)
            if basename not in wanted:
                continue
            src = tar.extractfile(member)
            if src is None:
                continue
            target = output_dir / basename
            with src, open(target, "wb") as out:
                out.write(src.read())
            written.append(basename)
            logger.info(f"Extracted {basename} -> {target}")
    missing = wanted - set(written)
    if missing:
        raise RuntimeError(
            f"Tarball {tar_path.name} did not contain expected files: {sorted(missing)}"
        )
    return written


def download_from_github_release(
    output_dir: Path = DEFAULT_TRAINING_DATA_DIR,
    repo: str = GITHUB_REPO,
    tag: str | None = DEFAULT_RELEASE_TAG,
) -> Path:
    """
    Download and extract the training-data tarball from a GitHub release.

    Args:
        output_dir: Directory to write the extracted ``.jsonl`` files into.
        repo: ``owner/name`` of the releases repo.
        tag: Release tag to pull; None resolves the most recent release
            (including prereleases).

    Returns:
        Path to ``output_dir``.
    """
    output_dir = Path(output_dir)
    release = _resolve_release(repo, tag)
    logger.info(
        f"Using release {release.get('tag_name')!r} from {repo} "
        + f"(prerelease={release.get('prerelease')})"
    )
    asset = _pick_tarball_asset(release)
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = Path(tmpdir) / asset["name"]
        _download_asset(asset, tar_path)
        _extract_training_files(tar_path, output_dir)
    return output_dir


def get_training_data(
    output_dir: Path = DEFAULT_TRAINING_DATA_DIR,
    force: bool = False,
    repo: str = GITHUB_REPO,
    tag: str | None = DEFAULT_RELEASE_TAG,
) -> Path:
    """
    Ensure training data is available locally, downloading it from the GitHub
    release when missing (or when ``force`` is set).

    Args:
        output_dir: Directory to store/check for training data.
        force: If True, re-download even if the files already exist.
        repo: ``owner/name`` of the releases repo.
        tag: Release tag to pull; None resolves the most recent release
            (including prereleases).

    Returns:
        Path to the training data directory.

    Raises:
        FileNotFoundError: If training data is missing and the download fails.
    """
    output_dir = Path(output_dir)

    missing = [f for f in REQUIRED_FILES if not (output_dir / f).exists()]

    if missing and not force:
        logger.info(
            f"Training data missing {missing} in {output_dir}; "
            + f"downloading from GitHub release ({repo})."
        )
    elif force:
        logger.info(f"force=True; re-downloading training data into {output_dir}.")

    if missing or force:
        try:
            download_from_github_release(output_dir=output_dir, repo=repo, tag=tag)
        except Exception as e:
            still_missing = [f for f in REQUIRED_FILES if not (output_dir / f).exists()]
            if still_missing:
                raise FileNotFoundError(
                    f"Training data not found: {still_missing}. "
                    + f"Expected in {output_dir}. Automatic download from "
                    + f"{repo} failed: {e}. Set GITHUB_TOKEN/GH_TOKEN if the repo "
                    + "is private, or place the files manually."
                ) from e
            logger.warning(f"Download reported an error but files are present: {e}")

    logger.info(f"Training data available at: {output_dir}")
    return output_dir


def get_topclassifier_training_path(
    training_dir: Path = DEFAULT_TRAINING_DATA_DIR,
) -> Path:
    """Get path to top classifier (mmem) training data."""
    return training_dir / TOPCLASSIFIER_FILE


def get_subclassifier_training_path(
    training_dir: Path = DEFAULT_TRAINING_DATA_DIR,
) -> Path:
    """Get path to subclassifier (em) training data."""
    return training_dir / SUBCLASSIFIER_FILE
