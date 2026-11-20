# azure_upload_utils.py

from azure.storage.blob import ContainerClient
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union, List, Dict, Optional


# ============================================================
# ============================================================

def create_container_client(sas_url: str) -> ContainerClient:
    """Create an Azure container client from a SAS URL."""
    return ContainerClient.from_container_url(sas_url)


# ============================================================
# ============================================================

def collect_files(
    local_dir: Union[str, Path],
    pattern: str = "*",
    recursive: bool = True
) -> List[Path]:
    """Find local files that match the upload pattern."""
    local_dir = Path(local_dir)

    if not local_dir.exists():
        raise FileNotFoundError(f"Local directory not found: {local_dir}")

    if not local_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {local_dir}")

    if recursive:
        files = [p for p in local_dir.rglob(pattern) if p.is_file()]
    else:
        files = [p for p in local_dir.glob(pattern) if p.is_file()]

    return files


# ============================================================
# ============================================================

def build_blob_name(
    local_file: Union[str, Path],
    local_dir: Union[str, Path],
    azure_prefix: str
) -> str:
    """Build a blob name from the file's relative path and upload prefix."""
    local_file = Path(local_file)
    local_dir = Path(local_dir)

    relative_path = local_file.relative_to(local_dir)

    azure_prefix = azure_prefix.strip("/")

    blob_name = f"{azure_prefix}/{relative_path.as_posix()}"

    return blob_name


# ============================================================
# ============================================================

def upload_single_file(
    container: ContainerClient,
    local_file: Union[str, Path],
    local_dir: Union[str, Path],
    azure_prefix: str,
    overwrite: bool = True
) -> Dict[str, object]:
    """Upload one file to Azure Blob Storage."""
    local_file = Path(local_file)
    local_dir = Path(local_dir)

    try:
        blob_name = build_blob_name(
            local_file=local_file,
            local_dir=local_dir,
            azure_prefix=azure_prefix
        )

        with open(local_file, "rb") as f:
            container.upload_blob(
                name=blob_name,
                data=f,
                overwrite=overwrite
            )

        return {
            "success": True,
            "local_file": str(local_file),
            "blob_name": blob_name,
            "message": f"Uploaded {local_file} -> {blob_name}"
        }

    except Exception as e:
        return {
            "success": False,
            "local_file": str(local_file),
            "blob_name": None,
            "message": f"Failed: {local_file} | {e}"
        }


# ============================================================
# ============================================================

def upload_directory_to_azure(
    sas_url: str,
    local_dir: Union[str, Path],
    azure_prefix: str,
    max_workers: int = 4,
    overwrite: bool = True,
    pattern: str = "*",
    recursive: bool = True,
    verbose: bool = True
) -> Dict[str, object]:
    """Upload matching files concurrently and report the results."""
    local_dir = Path(local_dir)

    container = create_container_client(sas_url)

    files = collect_files(
        local_dir=local_dir,
        pattern=pattern,
        recursive=recursive
    )

    if verbose:
        print(f"Local directory: {local_dir}")
        print(f"Azure prefix: {azure_prefix}")
        print(f"Files found: {len(files)}")
        print(f"Starting upload, max_workers={max_workers}, overwrite={overwrite}\n")

    results = []

    if len(files) == 0:
        if verbose:
            print("No files found.")

        return {
            "local_dir": str(local_dir),
            "azure_prefix": azure_prefix,
            "total": 0,
            "success": 0,
            "failed": 0,
            "results": []
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                upload_single_file,
                container,
                local_file,
                local_dir,
                azure_prefix,
                overwrite
            )
            for local_file in files
        ]

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)

            if verbose:
                print(f"[{i}/{len(files)}]", result["message"])

    success_count = sum(1 for r in results if r["success"])
    failed_count = len(results) - success_count

    if verbose:
        print("\nUpload complete.")
        print(f"Succeeded: {success_count}")
        print(f"Failed: {failed_count}")
        print(f"Total: {len(results)}")

    return {
        "local_dir": str(local_dir),
        "azure_prefix": azure_prefix,
        "total": len(results),
        "success": success_count,
        "failed": failed_count,
        "results": results
    }


# ============================================================
# ============================================================

def preview_upload_plan(
    local_dir: Union[str, Path],
    azure_prefix: str,
    pattern: str = "*",
    recursive: bool = True,
    max_show: int = 50
) -> List[Dict[str, str]]:
    """Preview local paths and their Azure blob destinations."""
    local_dir = Path(local_dir)

    files = collect_files(
        local_dir=local_dir,
        pattern=pattern,
        recursive=recursive
    )

    plan = []

    for local_file in files:
        blob_name = build_blob_name(
            local_file=local_file,
            local_dir=local_dir,
            azure_prefix=azure_prefix
        )

        plan.append({
            "local_file": str(local_file),
            "blob_name": blob_name
        })

    print(f"Local directory: {local_dir}")
    print(f"Azure prefix: {azure_prefix}")
    print(f"Files found: {len(plan)}")
    print("\nPreview:")

    for item in plan[:max_show]:
        print(f"{item['local_file']} -> {item['blob_name']}")

    if len(plan) > max_show:
        print(f"... {len(plan) - max_show} more files")

    return plan
