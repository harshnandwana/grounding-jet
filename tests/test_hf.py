import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from visual_jev.hf import REQUIRED_FILES, verify_download


class HuggingFaceDownloadTest(unittest.TestCase):
    def test_verifies_hashes_and_split_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hashes = {}
            for name in REQUIRED_FILES:
                content = b'{}\n'
                (root / name).write_bytes(content)
                hashes[name] = hashlib.sha256(content).hexdigest()
            (root / "manifest.json").write_text(json.dumps({
                "image_files_included": False,
                "image_manifest_records": 1,
                "files_sha256": hashes,
                "splits": {split: {"records": 1} for split in ("train", "validation", "test")},
            }))
            verify_download(root)
            (root / "test.jsonl").write_bytes(b'{"changed":true}\n')
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_download(root)


if __name__ == "__main__":
    unittest.main()
