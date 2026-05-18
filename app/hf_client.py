"""Thin wrapper around huggingface_hub."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.errors import HfHubHTTPError

from . import DownloadError


@dataclass
class ModelSummary:
    repo_id: str
    author: str
    last_modified: datetime | None
    downloads: int
    tags: list[str]
    description: str | None = None
    pipeline_tag: str | None = None
    base_model: str | None = None
    license: str | None = None
    min_size_gb: float | None = None


@dataclass
class RepoFile:
    filename: str
    size: int
    quant: str | None
    is_mmproj: bool


@dataclass
class FileGroup:
    """One install unit. For split GGUFs, groups all parts under the primary (00001)."""
    primary_filename: str
    all_filenames: list[str] = field(default_factory=list)
    total_bytes: int = 0
    quant: str | None = None
    is_mmproj: bool = False
    part_count: int = 1


_SPLIT_RE = re.compile(r"(?P<base>.+?)-(?P<idx>\d{5})-of-(?P<tot>\d{5})\.gguf$", re.IGNORECASE)

_QUANT_RE = re.compile(
    r"(?:^|[.\-_/])"
    r"((?:UD-)?(?:IQ|Q|F|BF|MXFP|NF)\d+(?:_[A-Z0-9]+)*(?:-[A-Z0-9]+)?)",
    re.IGNORECASE,
)

_QUANT_PREF = [
    "UD-Q4_K_XL", "Q4_K_M", "UD-Q4_K_M", "Q4_K_S",
    "UD-Q5_K_XL", "Q5_K_M", "Q5_K_S",
    "UD-Q6_K_XL", "Q6_K",
    "UD-Q8_K_XL", "Q8_0", "Q8_K_M",
    "IQ4_NL", "IQ4_XS",
    "Q3_K_M", "Q3_K_S", "IQ3_XS", "IQ3_M", "IQ3_S",
    "Q2_K", "IQ2_XS", "IQ2_S", "IQ2_XXS",
    "MXFP4_MOE", "NF4", "F16", "BF16", "F32",
]

_SORT_MAP = {
    "downloads": "downloads",
    "updated": "lastModified",
    "created": "createdAt",
    "likes": "likes",
}


def _api() -> HfApi:
    return HfApi()


def _parse_quant(filename: str) -> str | None:
    """Parse a canonical quant tag. Handles split suffixes and subfolder prefixes."""
    basename = Path(filename).name
    stem = basename
    split_m = _SPLIT_RE.search(stem)
    if split_m:
        stem = split_m.group("base")
    if "/" in filename:
        parent = filename.rsplit("/", 1)[0].split("/")[-1]
        m_parent = _QUANT_RE.search(parent)
        if m_parent:
            return m_parent.group(1).upper()
    m = _QUANT_RE.search(stem)
    if not m:
        return None
    return m.group(1).upper()


def _is_mmproj(filename: str) -> bool:
    return "mmproj" in filename.lower()


def _split_key(filename: str) -> tuple[str, int, int] | None:
    """Return (base, idx, total) if filename is a split part, else None."""
    m = _SPLIT_RE.search(filename)
    if not m:
        return None
    return (m.group("base"), int(m.group("idx")), int(m.group("tot")))


def _quant_sort_rank(q: str | None) -> int:
    if not q:
        return 9999
    try:
        return _QUANT_PREF.index(q.upper())
    except ValueError:
        return 500


def _to_dt(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _card_to_dict(card_data) -> dict:
    if card_data is None:
        return {}
    if isinstance(card_data, dict):
        return card_data
    to_dict = getattr(card_data, "to_dict", None)
    if callable(to_dict):
        try:
            d = to_dict()
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    return dict(getattr(card_data, "__dict__", {}) or {})


def _extract_description(card_data: dict) -> str | None:
    for key in ("description", "summary", "short_description", "tagline"):
        val = card_data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()[:240]
    return None


def _extract_base_model(card_data: dict) -> str | None:
    bm = card_data.get("base_model")
    if isinstance(bm, str) and bm.strip():
        return bm.strip()
    if isinstance(bm, list) and bm and isinstance(bm[0], str):
        return bm[0].strip()
    return None


import time as _time
from concurrent.futures import ThreadPoolExecutor

_SIZE_CACHE: dict[str, tuple[float, int]] = {}
_SIZE_CACHE_TTL = 3600.0

_README_CACHE: dict[str, tuple[float, str]] = {}
_README_CACHE_TTL = 3600.0
_README_MAX_CHARS = 20_000


def get_readme(repo_id: str) -> str | None:
    """Fetch raw README.md from HF, strip YAML frontmatter, cache 1h. Returns markdown source."""
    import httpx

    now = _time.time()
    cached = _README_CACHE.get(repo_id)
    if cached and cached[0] > now:
        return cached[1] or None
    url = f"https://huggingface.co/{repo_id}/raw/main/README.md"
    try:
        resp = httpx.get(url, timeout=10.0, follow_redirects=True)
    except httpx.HTTPError:
        _README_CACHE[repo_id] = (now + _README_CACHE_TTL, "")
        return None
    if resp.status_code != 200:
        _README_CACHE[repo_id] = (now + _README_CACHE_TTL, "")
        return None
    text = resp.text
    if text.startswith("---"):
        end = text.find("\n---\n", 3)
        if end == -1:
            end = text.find("\n---\r\n", 3)
        if end > 0:
            text = text[end + 5:]
    text = text.lstrip()
    if len(text) > _README_MAX_CHARS:
        text = text[:_README_MAX_CHARS] + "\n\n*(...truncated, see full on Hugging Face)*"
    _README_CACHE[repo_id] = (now + _README_CACHE_TTL, text)
    return text or None


def render_markdown(text: str) -> str:
    """Render markdown to HTML with raw HTML disabled (safe for untrusted input)."""
    from markdown_it import MarkdownIt

    md = MarkdownIt("commonmark", {"html": False, "linkify": True, "typographer": False})
    return md.render(text)


_MIN_WEIGHT_BYTES = 500 * 1024 * 1024


def _min_group_bytes(repo_id: str) -> int | None:
    """Return smallest weight-GGUF group size. Filters out helper/tokenizer .gguf files
    (those usually lack a quant tag and are under 500 MB)."""
    now = _time.time()
    cached = _SIZE_CACHE.get(repo_id)
    if cached and cached[0] > now:
        return cached[1] if cached[1] > 0 else None
    try:
        main, _ = list_repo_groups(repo_id)
    except DownloadError:
        _SIZE_CACHE[repo_id] = (now + _SIZE_CACHE_TTL, 0)
        return None
    candidates = [
        g.total_bytes
        for g in main
        if g.total_bytes >= _MIN_WEIGHT_BYTES and g.quant is not None
    ]
    if not candidates:
        candidates = [g.total_bytes for g in main if g.total_bytes >= _MIN_WEIGHT_BYTES]
    if not candidates:
        _SIZE_CACHE[repo_id] = (now + _SIZE_CACHE_TTL, 0)
        return None
    mb = min(candidates)
    _SIZE_CACHE[repo_id] = (now + _SIZE_CACHE_TTL, mb)
    return mb


def search_models(
    query: str | None,
    author: str | None = None,
    library: str = "gguf",
    sort: str = "downloads",
    limit: int = 20,
    offset: int = 0,
    enrich_sizes: bool = False,
) -> list[ModelSummary]:
    """Search HF models and return a page of summaries with best-effort description."""
    api = _api()
    sort_key = _SORT_MAP.get(sort, "downloads")
    kwargs: dict = {"sort": sort_key, "limit": limit + offset, "cardData": True, "full": True}
    if query:
        kwargs["search"] = query
    if author:
        kwargs["author"] = author
    if library and library != "all":
        kwargs["filter"] = [library]

    try:
        items = list(api.list_models(**kwargs))
    except (HfHubHTTPError, TypeError) as exc:
        raise DownloadError(f"HF search failed: {exc}") from exc

    out: list[ModelSummary] = []
    for m in items[offset : offset + limit]:
        repo_id = getattr(m, "id", None) or getattr(m, "modelId", "")
        author_part = repo_id.split("/", 1)[0] if "/" in repo_id else ""
        card = _card_to_dict(getattr(m, "cardData", None) or getattr(m, "card_data", None))
        pipeline = (
            getattr(m, "pipeline_tag", None)
            or getattr(m, "pipelineTag", None)
            or card.get("pipeline_tag")
        )
        license_val = card.get("license") if isinstance(card.get("license"), str) else None
        out.append(
            ModelSummary(
                repo_id=repo_id,
                author=author_part,
                last_modified=_to_dt(getattr(m, "last_modified", None) or getattr(m, "lastModified", None)),
                downloads=int(getattr(m, "downloads", 0) or 0),
                tags=list(getattr(m, "tags", []) or []),
                description=_extract_description(card),
                pipeline_tag=pipeline,
                base_model=_extract_base_model(card),
                license=license_val,
            )
        )
    if enrich_sizes and out:
        with ThreadPoolExecutor(max_workers=10) as pool:
            sizes = list(pool.map(_min_group_bytes, [m.repo_id for m in out]))
        for m, sz in zip(out, sizes):
            if sz is not None and sz > 0:
                m.min_size_gb = round(sz / 1024 / 1024 / 1024, 1)
    return out


def list_repo_files(repo_id: str) -> list[RepoFile]:
    """Return .gguf files in a repo with size, quant label, and mmproj flag."""
    api = _api()
    try:
        info = api.repo_info(repo_id=repo_id, files_metadata=True)
    except HfHubHTTPError as exc:
        raise DownloadError(f"HF repo_info failed: {exc}") from exc

    results: list[RepoFile] = []
    for sibling in getattr(info, "siblings", []) or []:
        filename = getattr(sibling, "rfilename", None) or getattr(sibling, "filename", "")
        if not filename or not filename.lower().endswith(".gguf"):
            continue
        size = int(getattr(sibling, "size", 0) or 0)
        results.append(
            RepoFile(
                filename=filename,
                size=size,
                quant=_parse_quant(filename),
                is_mmproj=_is_mmproj(filename),
            )
        )
    return results


def list_repo_groups(repo_id: str) -> tuple[list[FileGroup], list[FileGroup]]:
    """Return (main_groups, mmproj_groups) for a repo.

    Split GGUFs (NNNNN-of-MMMMM) collapse into a single group keyed on part 00001.
    Groups are sorted by a curated quant-quality preference, then by size.
    """
    files = list_repo_files(repo_id)
    splits: dict[str, FileGroup] = {}
    singles: list[FileGroup] = []
    mmproj: list[FileGroup] = []
    for f in files:
        if f.is_mmproj:
            mmproj.append(
                FileGroup(
                    primary_filename=f.filename,
                    all_filenames=[f.filename],
                    total_bytes=f.size,
                    quant=f.quant,
                    is_mmproj=True,
                    part_count=1,
                )
            )
            continue
        sk = _split_key(f.filename)
        if sk is None:
            singles.append(
                FileGroup(
                    primary_filename=f.filename,
                    all_filenames=[f.filename],
                    total_bytes=f.size,
                    quant=f.quant,
                    part_count=1,
                )
            )
            continue
        base, idx, tot = sk
        group = splits.setdefault(
            base,
            FileGroup(primary_filename="", all_filenames=[], total_bytes=0, quant=f.quant, part_count=tot),
        )
        group.all_filenames.append(f.filename)
        group.total_bytes += f.size
        group.quant = group.quant or f.quant
        if idx == 1:
            group.primary_filename = f.filename

    for g in splits.values():
        g.all_filenames.sort()
        if not g.primary_filename and g.all_filenames:
            g.primary_filename = g.all_filenames[0]

    main = singles + [g for g in splits.values() if g.primary_filename]

    def key(g: FileGroup) -> tuple[int, int]:
        return (_quant_sort_rank(g.quant), g.total_bytes)

    main.sort(key=key)
    mmproj.sort(key=lambda g: _quant_sort_rank(g.quant))
    return main, mmproj


class DownloadCancelled(Exception):
    pass


def download_file(
    repo_id: str,
    filename: str,
    dest_dir: Path,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], str | None] | None = None,
) -> Path:
    """Download a single file from a repo into dest_dir. Returns the local path.

    Respects the runtime speed limit from config.get_dl_speed_limit().
    When limit > 0, uses chunked requests download with throttling.
    When limit == 0, uses hf_hub_download for maximum speed.

    cancel_check: callable returning "cancel"/"pause" to stop, or None to continue.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = _download_throttled(repo_id, filename, dest_dir, cancel_check)
    if progress_cb is not None:
        try:
            size = path.stat().st_size
            progress_cb(size, size)
        except OSError:
            pass
    return path


def _download_throttled(
    repo_id: str,
    filename: str,
    dest_dir: Path,
    cancel_check: Callable[[], str | None] | None = None,
) -> Path:
    """Chunked download with resume, cancel/pause, and optional speed throttling.

    Speed limit is read from config each chunk — 0 means unlimited.
    """
    import time
    import requests
    from .config import get_dl_speed_limit

    url = hf_hub_url(repo_id=repo_id, filename=filename)
    token = HfApi().token
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    dest_path = dest_dir / Path(filename).name
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return dest_path

    tmp_path = dest_path.with_suffix(dest_path.suffix + ".incomplete")

    resume_pos = 0
    if tmp_path.exists():
        resume_pos = tmp_path.stat().st_size
        headers["Range"] = f"bytes={resume_pos}-"

    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=30,
                            allow_redirects=True)
        if resp.status_code == 416:
            if tmp_path.exists():
                tmp_path.rename(dest_path)
                return dest_path
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise DownloadError(f"HF download failed for {repo_id}/{filename}: {exc}") from exc

    chunk_size = 8 * 1024 * 1024

    mode = "ab" if resume_pos > 0 and resp.status_code == 206 else "wb"
    with open(tmp_path, mode) as f:
        window_start = time.monotonic()
        window_bytes = 0
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            if cancel_check is not None:
                signal = cancel_check()
                if signal:
                    raise DownloadCancelled(signal)
            f.write(chunk)
            window_bytes += len(chunk)
            limit_mb = get_dl_speed_limit()
            if limit_mb > 0:
                limit_bytes = limit_mb * 1024 * 1024
                elapsed = time.monotonic() - window_start
                if elapsed > 0 and window_bytes / elapsed > limit_bytes:
                    sleep_time = (window_bytes / limit_bytes) - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                if window_bytes >= limit_bytes:
                    window_start = time.monotonic()
                    window_bytes = 0
            else:
                window_start = time.monotonic()
                window_bytes = 0
    tmp_path.rename(dest_path)
    return dest_path


def expand_split(repo_id: str, primary_filename: str) -> list[str]:
    """If primary_filename is the first part of a split, return all sibling parts (sorted).
    Otherwise return [primary_filename]."""
    sk = _split_key(primary_filename)
    if sk is None:
        return [primary_filename]
    base = sk[0]
    all_files = list_repo_files(repo_id)
    siblings: list[str] = []
    for f in all_files:
        if f.is_mmproj:
            continue
        s = _split_key(f.filename)
        if s is None:
            continue
        if Path(f.filename).name.startswith(Path(base).name) and Path(f.filename).parent == Path(primary_filename).parent:
            siblings.append(f.filename)
    siblings.sort()
    if primary_filename not in siblings:
        siblings.append(primary_filename)
        siblings.sort()
    return siblings


def repo_last_modified(repo_id: str) -> datetime:
    """Return the repo's last_modified timestamp (UTC)."""
    api = _api()
    try:
        info = api.repo_info(repo_id=repo_id)
    except HfHubHTTPError as exc:
        raise DownloadError(f"HF repo_info failed: {exc}") from exc
    lm = _to_dt(getattr(info, "last_modified", None) or getattr(info, "lastModified", None))
    if lm is None:
        raise DownloadError(f"no last_modified available for {repo_id}")
    return lm


_LM_CACHE: dict[str, tuple[float, datetime | None]] = {}
_LM_CACHE_TTL = 1800.0  # 30 min


def repo_last_modified_cached(repo_id: str, force: bool = False) -> datetime | None:
    """Like repo_last_modified, but cached + None on failure (no raise)."""
    now = _time.time()
    if not force:
        c = _LM_CACHE.get(repo_id)
        if c and c[0] > now:
            return c[1]
    try:
        lm = repo_last_modified(repo_id)
    except DownloadError:
        lm = None
    _LM_CACHE[repo_id] = (now + _LM_CACHE_TTL, lm)
    return lm


def bulk_repo_last_modified(repo_ids: list[str], force: bool = False) -> dict[str, datetime | None]:
    """Parallel-fetch latest_modified for multiple repos. Cache-aware."""
    out: dict[str, datetime | None] = {}
    if not repo_ids:
        return out
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda r: repo_last_modified_cached(r, force=force), repo_ids))
    for r, lm in zip(repo_ids, results):
        out[r] = lm
    return out


def invalidate_lm_cache(repo_id: str | None = None) -> None:
    if repo_id is None:
        _LM_CACHE.clear()
    else:
        _LM_CACHE.pop(repo_id, None)


__all__ = [
    "ModelSummary",
    "RepoFile",
    "FileGroup",
    "search_models",
    "list_repo_files",
    "list_repo_groups",
    "download_file",
    "expand_split",
    "repo_last_modified",
    "repo_last_modified_cached",
    "bulk_repo_last_modified",
    "invalidate_lm_cache",
    "get_readme",
    "render_markdown",
]
