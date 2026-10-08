"""Common Arrow/JSONL data contract, with relocatable audio and sidecar paths."""

import json
import tempfile
from pathlib import Path

from datasets import Dataset, load_from_disk
from datasets.arrow_writer import ArrowWriter

LANGUAGES = {"en": "a", "zh": "z", "ja": "j"}


class Manifest:
    def __init__(self, path, audio_root=None):
        path = Path(path)
        self.root = Path(audio_root) if audio_root else path.parent
        if path.is_dir():
            self.rows = load_from_disk(str(path))
            if not isinstance(self.rows, Dataset):
                raise ValueError("Select a single Dataset split, not a DatasetDict")
        else:
            self.rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not len(self.rows):
            raise ValueError("Empty dataset")
        self.g2p = None

    def __len__(self):
        return len(self.rows)

    def audio_path(self, row):
        path = Path(row["audio_path"])
        return path if path.is_absolute() else self.root / path

    def ids(self, row, mixed=False):
        if "phoneme_ids" in row and row["phoneme_ids"] is not None:
            ids = list(row["phoneme_ids"])
        else:
            from pl_flow.text.frontend import KokoroG2P

            if self.g2p is None:
                self.g2p = KokoroG2P(lang_codes=["a", "z", "j"])
            if mixed:
                ids = self.g2p.mixed_g2p(row["text"], primary_lang=row["language"], threshold=0.95)[
                    1
                ]
            else:
                ids = self.g2p(row["text"], lang_code=LANGUAGES[row["language"]])[1]
        if len(ids) < 3 or min(ids) < 0 or max(ids) >= 178 or ids[0] != 0 or ids[-1] != 0:
            raise ValueError("Invalid phoneme sequence; expected fixed-vocabulary BOS/text/EOS")
        return ids


def write_rows(rows, output):
    """Stream rows to Arrow without pickling generators, models or entire datasets."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pl-flow-pack-", dir=output.parent) as temporary:
        arrow = Path(temporary) / "data.arrow"
        count = 0
        with ArrowWriter(path=str(arrow)) as writer:
            for row in rows:
                writer.write(row)
                count += 1
            if not count:
                raise ValueError("Cannot write an empty dataset")
            writer.finalize()
        Dataset.from_file(str(arrow)).save_to_disk(str(output))


def pack(rows, output, audio_root):
    """Build a portable save_to_disk pack without copying audio or local path prefixes."""
    output, root = Path(output), Path(audio_root).resolve()
    if output.exists():
        raise FileExistsError(output)

    def portable():
        for row in rows:
            row = dict(row)
            audio = Path(row["audio_path"])
            audio = audio.resolve() if audio.is_absolute() else (root / audio).resolve()
            if not audio.is_relative_to(root):
                raise ValueError("Audio path escapes the supplied audio root")
            row["audio_path"] = audio.relative_to(root).as_posix()
            yield row

    write_rows(portable(), output)
