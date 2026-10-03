"""Caption cleaning, tokenizer persistence, and image-level splits."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import split_image_ids
from src.text_preprocessing import clean_caption, run_preprocessing
from src.tokenizer_utils import build_vocabulary, load_tokenizer, save_tokenizer, texts_to_sequences
from src.utils import strip_sequence_tokens


def _write_mini_dataset(root: Path) -> tuple[Path, Path]:
    images_dir = root / "Images"
    images_dir.mkdir(parents=True)
    names = [f"img_{ch}.jpg" for ch in "abcdefghij"]
    for name in names:
        (images_dir / name).write_bytes(b"")

    captions_file = root / "captions.txt"
    rows = ["image,caption"]
    samples = [
        ("img_a.jpg", "A child in a pink dress is climbing stairs!"),
        ("img_a.jpg", "A little girl climbing into a wooden playhouse."),
        ("img_b.jpg", "A man in a blue shirt is standing on a sidewalk."),
        ("img_c.jpg", "Two dogs are running through the snow."),
        ("img_d.jpg", "A brown dog is playing with a red ball."),
        ("img_e.jpg", "People sit on benches near a fountain."),
        ("img_f.jpg", "A cyclist rides down a city street."),
        ("img_g.jpg", "A black dog jumps over a fallen tree."),
        ("img_h.jpg", "Children play soccer in a green park."),
        ("img_i.jpg", "A woman holds an umbrella in the rain."),
        ("img_j.jpg", "A boy throws a yellow frisbee on the beach."),
        ("missing.jpg", "This caption has no image file."),
    ]
    for image, caption in samples:
        rows.append(f"{image},{caption}")
    captions_file.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return captions_file, images_dir


def test_clean_caption() -> None:
    assert clean_caption("  Hello, WORLD!!!  ") == "hello world"
    assert "startseq" not in clean_caption("A dog.")
    assert strip_sequence_tokens("startseq a dog endseq") == "a dog"


def test_tokenizer_roundtrip(tmp_path: Path | None = None) -> None:
    tokenizer = build_vocabulary(["startseq a dog runs endseq", "startseq a cat sleeps endseq"])
    assert "dog" in tokenizer["word_to_index"]
    assert "unicorn" not in tokenizer["word_to_index"]
    assert tokenizer["word_to_index"]["startseq"] != 0
    sequences = texts_to_sequences(["startseq a dog endseq"], tokenizer)
    assert sequences[0][0] == tokenizer["word_to_index"]["startseq"]
    target = tmp_path / "tokenizer.pkl" if tmp_path else Path(tempfile.mkdtemp()) / "tokenizer.pkl"
    save_tokenizer(tokenizer, target)
    loaded = load_tokenizer(target)
    assert loaded["vocab_size"] == tokenizer["vocab_size"]
    assert loaded["word_to_index"]["endseq"] == tokenizer["word_to_index"]["endseq"]


def test_preprocessing_pipeline() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        captions_file, images_dir = _write_mini_dataset(tmp_path)
        processed_dir = tmp_path / "processed"
        metadata = run_preprocessing(
            force=True,
            captions_file=captions_file,
            images_dir=images_dir,
            processed_dir=processed_dir,
        )
        tokenizer = load_tokenizer(processed_dir / "tokenizer.pkl")
        splits = json.loads((processed_dir / "splits.json").read_text(encoding="utf-8"))
        train_set, val_set, test_set = set(splits["train"]), set(splits["val"]), set(splits["test"])
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)
        assert metadata["skipped_missing_image"] == 1
        assert metadata["vocab_size"] == tokenizer["vocab_size"]
        assert "startseq" in tokenizer["word_to_index"]
        assert "endseq" in tokenizer["word_to_index"]
        split_again = split_image_ids(list(train_set | val_set | test_set))
        assert split_again == splits


def main() -> None:
    test_clean_caption()
    test_tokenizer_roundtrip()
    test_preprocessing_pipeline()
    print("test_preprocessing passed.")


if __name__ == "__main__":
    main()
