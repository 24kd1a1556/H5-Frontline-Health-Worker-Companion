import sys
import unittest
from pathlib import Path

FULL_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FULL_CODE))

from baseline import BM25, tokenize


class TestTokenize(unittest.TestCase):
    def test_converts_text_to_lowercase_tokens(self):
        self.assertEqual(tokenize("Heavy BLEEDING"), ["heavy", "bleeding"])

    def test_handles_numbers_and_punctuation(self):
        self.assertEqual(tokenize("Child: 60/min!"), ["child", "60", "min"])


class TestBM25(unittest.TestCase):
    def setUp(self):
        self.documents = {
            "ANC-1.1": "pregnancy antenatal care checkup",
            "CHD-3.1": "child fast breathing respiratory rate",
            "NUT-2.1": "child malnutrition nutrition assessment",
        }
        self.searcher = BM25(self.documents)

    def test_retrieves_relevant_document(self):
        results = self.searcher.search("fast breathing", k=3)
        self.assertTrue(results)
        self.assertEqual(results[0][1], "CHD-3.1")

    def test_respects_result_limit(self):
        results = self.searcher.search("child", k=2)
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()

from baseline import choose_section


class TestChooseSection(unittest.TestCase):
    def setUp(self):
        self.documents = {
            "CHD-3.1": "child fast breathing respiratory rate",
            "HBNC-1.2": "low birth weight newborn care",
            "ANC-1.1": "pregnancy antenatal care",
        }
        self.searcher = BM25(self.documents)

    def test_fast_breathing_selects_child_protocol(self):
        result = choose_section("What is the fast breathing threshold?", self.searcher)
        self.assertEqual(result, "CHD-3.1")

    def test_low_birth_weight_selects_newborn_protocol(self):
        result = choose_section("How to care for a low birth weight baby?", self.searcher)
        self.assertEqual(result, "HBNC-1.2")


if __name__ == "__main__":
    unittest.main()

