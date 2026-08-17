import json
import re
import unittest
from pathlib import Path


class FishPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = (
            Path(__file__).parents[1]
            / "assets"
            / "resource"
            / "base"
            / "pipeline"
            / "Fish"
            / "FishNew.json"
        )
        text = path.read_text(encoding="utf-8")
        cls.pipeline = json.loads(re.sub(r"//.*$", "", text, flags=re.MULTILINE))

    def test_game_result_is_handled_before_restart(self):
        self.assertEqual(self.pipeline["FishNewGaming"]["next"][0], "FishNewGameResult")
        self.assertEqual(
            self.pipeline["FishNewGameResult"]["next"],
            ["[Anchor]FishNewRestart"],
        )
        self.assertNotIn(
            "[Anchor]FishNewRestart", self.pipeline["FishNewGaming"]["next"]
        )
        self.assertIn("FishNewGaming", self.pipeline["FishNewGaming"]["next"])


if __name__ == "__main__":
    unittest.main()
