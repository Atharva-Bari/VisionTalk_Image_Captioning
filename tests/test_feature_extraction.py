"""Stage 3 validation: CNN features on a handful of images, cache reuse, skip corrupt files."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_extraction import extract_features, load_features, vector_cache_path


def _write_test_images(images_dir: Path, count: int = 8) -> list[str]:
    images_dir.mkdir(parents=True)
    names = []
    for i in range(count):
        name = f"sample_{i:02d}.jpg"
        image = Image.new("RGB", (64, 48), color=(20 * i, 80, 160))
        image.save(images_dir / name, format="JPEG")
        names.append(name)
    (images_dir / "zz_broken.jpg").write_bytes(b"not an image")
    return names


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        images_dir = tmp_path / "Images"
        vector_dir = tmp_path / "vectors"
        metadata_path = tmp_path / "extractor_metadata.json"
        names = _write_test_images(images_dir, count=8)

        first = extract_features(
            images_dir=images_dir,
            vector_dir=vector_dir,
            metadata_path=metadata_path,
            limit=8,
            force=False,
            batch_size=4,
        )
        assert first["num_extracted_this_run"] == 8
        assert first["feature_vector_dim"] == 4096
        assert first["cnn_model_name"] == "VGG16"
        assert first["num_reused_from_cache"] == 0

        for name in names:
            path = vector_cache_path(name, vector_dir)
            assert path.is_file(), f"missing {path}"
            vector = np.load(path)
            assert vector.shape == (4096,), vector.shape

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["preprocess"].endswith("vgg16.preprocess_input")
        assert metadata["include_top"] is True

        mtimes = {name: vector_cache_path(name, vector_dir).stat().st_mtime for name in names}

        second = extract_features(
            images_dir=images_dir,
            vector_dir=vector_dir,
            metadata_path=metadata_path,
            limit=8,
            force=False,
            batch_size=4,
        )
        assert second["num_extracted_this_run"] == 0
        assert second["num_reused_from_cache"] == 8
        for name in names:
            assert vector_cache_path(name, vector_dir).stat().st_mtime == mtimes[name]

        loaded = load_features(names, vector_dir=vector_dir)
        assert set(loaded) == set(names)
        assert loaded[names[0]].shape == (4096,)

        third = extract_features(
            images_dir=images_dir,
            vector_dir=vector_dir,
            metadata_path=metadata_path,
            limit=9,
            force=False,
            batch_size=4,
        )
        assert "zz_broken.jpg" in third["failed_images"]
        assert not vector_cache_path("zz_broken.jpg", vector_dir).is_file()

    print()
    print("Stage 3 validation test passed.")
    print("CNN: VGG16 FC2  feature dim: 4096  images tested: 8  cache reuse: yes")


if __name__ == "__main__":
    main()
