import json
from pathlib import Path
import unittest

from visual_jev.dataset import make_records, validate_record


class DatasetTest(unittest.TestCase):
    def setUp(self):
        source = json.loads((Path(__file__).resolve().parent / "fixtures" / "coco_instances.json").read_text())
        self.records = make_records(source["images"][0], source["annotations"],
                                    {item["id"]: item["name"] for item in source["categories"]},
                                    "train2017", Path(__file__).resolve().parent / "fixtures")

    def test_fixture_records_are_valid_and_balanced(self):
        self.assertEqual(len(self.records), 9)
        self.assertTrue(all(not validate_record(record) for record in self.records))
        spatial_answers = [record["choices"][record["answer_index"]]
                           for record in self.records if record["task"] == "spatial_boolean"]
        self.assertEqual(spatial_answers.count("YES"), spatial_answers.count("NO"))

    def test_duplicate_category_does_not_create_ambiguous_grounding(self):
        source = json.loads((Path(__file__).resolve().parent / "fixtures" / "coco_instances.json").read_text())
        extra = {**source["annotations"][0], "id": 104, "bbox": [350, 100, 50, 200]}
        records = make_records(source["images"][0], source["annotations"] + [extra],
                               {item["id"]: item["name"] for item in source["categories"]},
                               "train2017", Path(__file__).resolve().parent / "fixtures")
        self.assertFalse(any(record["task"] == "box_choice" and "person" in record["question"] for record in records))

    def test_wrong_candidate_is_rejected(self):
        record = next(item for item in self.records if item["task"] == "box_choice")
        record["answer_index"] = (record["answer_index"] + 1) % 3
        self.assertIn("correct candidate does not match evidence", validate_record(record))


if __name__ == "__main__":
    unittest.main()
