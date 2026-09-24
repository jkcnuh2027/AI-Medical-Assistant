import shutil
import tempfile
from pathlib import Path

from fastapi import UploadFile


def save_uploaded_files(files: list[UploadFile]) -> list[str]:
    """Save to unique generated paths. The caller must remove returned files."""
    paths = []
    try:
        for file in files:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as output:
                paths.append(output.name)
                shutil.copyfileobj(file.file, output)
        return paths
    except Exception:
        for path in paths:
            Path(path).unlink(missing_ok=True)
        raise
