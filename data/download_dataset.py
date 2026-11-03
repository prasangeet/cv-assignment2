from pathlib import Path
import subprocess
import shutil
import sys
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DOWNLOAD_DIR = DATA_DIR / "downloads"

DATASETS = {
    "DUTS-TR": "https://saliencydetection.net/duts/download/DUTS-TR.zip",
    "DUTS-TE": "https://saliencydetection.net/duts/download/DUTS-TE.zip",
}


def download_dataset(name: str, url: str) -> None:
    archive = DOWNLOAD_DIR / f"{name}.zip"
    destination = DATA_DIR / name

    # Skip downloading if the expected folders already exist.
    image_dir = destination / f"{name}-Image"
    mask_dir = destination / f"{name}-Mask"

    if image_dir.is_dir() and mask_dir.is_dir():
        print(f"[SKIP] {name} already extracted.")
        return

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)

    print(f"\n[DOWNLOAD] {name}")

    # wget supports resuming interrupted downloads.
    subprocess.run(
        [
            "wget",
            "-c",
            "--show-progress",
            "-O",
            str(archive),
            url,
        ],
        check=True,
    )

    print(f"[VERIFY] Checking {archive.name}")

    if not zipfile.is_zipfile(archive):
        raise RuntimeError(
            f"{archive} is not a valid ZIP archive. "
            "Check the download and try again."
        )

    print(f"[EXTRACT] {name}")

    # Extract into a temporary directory first.
    temp_dir = DATA_DIR / f".{name}_extracting"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    temp_dir.mkdir(parents=True)

    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(temp_dir)

        extracted_root = temp_dir / name

        if not extracted_root.is_dir():
            raise RuntimeError(
                f"Unexpected archive structure in {archive}"
            )

        if not (
            (extracted_root / f"{name}-Image").is_dir()
            and (extracted_root / f"{name}-Mask").is_dir()
        ):
            raise RuntimeError(
                f"Image/mask directories missing in {archive}"
            )

        # Move the extracted dataset into its final location.
        for item in extracted_root.iterdir():
            target = destination / item.name

            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()

            shutil.move(str(item), str(target))

        print(f"[DONE] {name} extracted successfully.")

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def main() -> None:
    if shutil.which("wget") is None:
        sys.exit("Error: wget is not installed. Install it first.")

    try:
        for name, url in DATASETS.items():
            download_dataset(name, url)

    except (subprocess.CalledProcessError, RuntimeError) as exc:
        sys.exit(f"\nDataset setup failed: {exc}")

    print("\nDataset setup complete!")
    print(f"Dataset location: {DATA_DIR}")


if __name__ == "__main__":
    main()
