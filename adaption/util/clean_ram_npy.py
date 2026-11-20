# memory_cache_utils.py

import os
import ctypes
from pathlib import Path
from typing import Iterable, List, Tuple, Union, Optional


# ============================================================
# libc / posix_fadvise setup
# ============================================================

POSIX_FADV_DONTNEED = 4


def _load_libc():
    """Load the Linux C library."""
    libc = ctypes.CDLL("libc.so.6", use_errno=True)

    libc.posix_fadvise.argtypes = [
        ctypes.c_int,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_int
    ]
    libc.posix_fadvise.restype = ctypes.c_int

    return libc


# ============================================================
# Memory status
# ============================================================

def read_cgroup_memory_stat() -> dict:
    """Read memory statistics from cgroup v2."""
    stat_path = Path("/sys/fs/cgroup/memory.stat")

    if not stat_path.exists():
        raise FileNotFoundError(
            "Cannot find /sys/fs/cgroup/memory.stat. "
            "This function is intended for Linux cgroup environments."
        )

    stat = {}

    with open(stat_path, "r") as f:
        for line in f:
            k, v = line.split()
            stat[k] = int(v)

    return stat


def read_cgroup_memory_current() -> int:
    """Read current cgroup memory usage."""
    path = Path("/sys/fs/cgroup/memory.current")

    if not path.exists():
        raise FileNotFoundError(
            "Cannot find /sys/fs/cgroup/memory.current."
        )

    return int(path.read_text().strip())


def read_cgroup_memory_limit() -> Optional[int]:
    """Read the cgroup memory limit, or return None if unlimited."""
    path = Path("/sys/fs/cgroup/memory.max")

    if not path.exists():
        raise FileNotFoundError(
            "Cannot find /sys/fs/cgroup/memory.max."
        )

    value = path.read_text().strip()

    if value == "max":
        return None

    return int(value)


def get_cgroup_memory_info() -> dict:
    """Collect cgroup memory usage and limits."""
    stat = read_cgroup_memory_stat()
    current = read_cgroup_memory_current()
    limit = read_cgroup_memory_limit()

    available = None if limit is None else limit - current

    return {
        "current": current,
        "limit": limit,
        "available": available,
        "anon": stat.get("anon", 0),
        "file": stat.get("file", 0),
        "slab": stat.get("slab", 0),
        "stat": stat,
    }


def show_mem(title: str = "Memory Status") -> dict:
    """Print and return cgroup memory information."""
    info = get_cgroup_memory_info()

    current = info["current"]
    limit = info["limit"]
    available = info["available"]

    print("\n" + "=" * 60)
    print(title)
    print(f"Used:      {current / 1024**3:.2f} GiB")

    if limit is None:
        print("Limit:     unlimited")
        print("Available: unknown")
    else:
        print(f"Limit:     {limit / 1024**3:.2f} GiB")
        print(f"Available: {available / 1024**3:.2f} GiB")

    print(f"anon:      {info['anon'] / 1024**3:.2f} GiB")
    print(f"file:      {info['file'] / 1024**3:.2f} GiB")
    print(f"slab:      {info['slab'] / 1024**3:.2f} GiB")

    return info


# ============================================================
# File search
# ============================================================

def find_large_files(
    roots: Iterable[Union[str, Path]],
    pattern: str = "*.npy",
    min_size_gb: float = 1.0,
    sort_desc: bool = True,
    verbose: bool = True
) -> List[Tuple[Path, float]]:
    """Find files matching a pattern and minimum size under the given roots."""
    targets = []

    for root in roots:
        root = Path(root)

        if not root.exists():
            if verbose:
                print(f"[Skip] Root does not exist: {root}")
            continue

        for p in root.rglob(pattern):
            try:
                size_gb = p.stat().st_size / 1024**3

                if size_gb >= min_size_gb:
                    targets.append((p, size_gb))

            except Exception as e:
                if verbose:
                    print(f"[Warn] Cannot stat file: {p} | {e}")

    if sort_desc:
        targets.sort(key=lambda x: x[1], reverse=True)

    if verbose:
        print(f"\nFound {len(targets)} files matching {pattern} >= {min_size_gb} GiB")

        for p, size_gb in targets:
            print(f"{size_gb:8.2f} GiB  {p}")

    return targets


# ============================================================
# Drop file cache hint
# ============================================================

def drop_file_cache(
    path: Union[str, Path],
    libc=None,
    verbose: bool = True
) -> bool:
    """Advise Linux that cached pages for this file can be discarded."""
    path = Path(path)

    if not path.exists():
        if verbose:
            print(f"[Skip] Not exists: {path}")
        return False

    if libc is None:
        libc = _load_libc()

    try:
        fd = os.open(str(path), os.O_RDONLY)

        try:
            ret = libc.posix_fadvise(
                fd,
                ctypes.c_longlong(0),
                ctypes.c_longlong(0),
                POSIX_FADV_DONTNEED
            )

            if ret == 0:
                if verbose:
                    print(f"OK: {path}")
                return True

            else:
                if verbose:
                    print(f"FAIL: {path} | ret={ret}")
                return False

        finally:
            os.close(fd)

    except Exception as e:
        if verbose:
            print(f"ERROR: {path} | {e}")
        return False


def drop_file_caches_for_targets(
    targets: Iterable[Tuple[Union[str, Path], float]],
    sync_before: bool = True,
    sync_after: bool = True,
    verbose: bool = True
) -> dict:
    """Apply file cache advice to each path."""
    targets = list(targets)

    if sync_before:
        if verbose:
            print("\nRunning os.sync() before dropping file cache hints...")
        os.sync()

    if verbose:
        print("\nDropping file cache hints...")

    libc = _load_libc()

    success = 0
    failed = 0

    for p, _ in targets:
        ok = drop_file_cache(
            path=p,
            libc=libc,
            verbose=verbose
        )

        if ok:
            success += 1
        else:
            failed += 1

    if sync_after:
        if verbose:
            print("\nRunning os.sync() after dropping file cache hints...")
        os.sync()

    result = {
        "success": success,
        "failed": failed,
        "total": len(targets)
    }

    if verbose:
        print("\nDrop cache summary")
        print("-" * 20)
        print("success:", success)
        print("failed:", failed)
        print("total:", len(targets))

    return result


# ============================================================
# Main reusable wrapper
# ============================================================

def drop_large_npy_file_cache(
    roots: Iterable[Union[str, Path]] = ("/workspace/data",),
    min_size_gb: float = 1.0,
    pattern: str = "*.npy",
    show_before: bool = True,
    show_after: bool = True,
    sync_before: bool = True,
    sync_after: bool = True,
    verbose: bool = True
) -> dict:
    """Find large arrays and advise Linux to discard their cached pages."""
    before_info = None
    after_info = None

    if show_before:
        before_info = show_mem("Before")

    targets = find_large_files(
        roots=roots,
        pattern=pattern,
        min_size_gb=min_size_gb,
        sort_desc=True,
        verbose=verbose
    )

    drop_result = drop_file_caches_for_targets(
        targets=targets,
        sync_before=sync_before,
        sync_after=sync_after,
        verbose=verbose
    )

    if show_after:
        after_info = show_mem("After")

    return {
        "before": before_info,
        "after": after_info,
        "targets": targets,
        "drop_result": drop_result
    }
