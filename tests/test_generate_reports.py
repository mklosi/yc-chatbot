import unittest

from modules.generate_reports import get_chunks


class TestGetChunks(unittest.TestCase):

    def test_1(self):
        max_length = 6
        overlap_percent = 0.33
        big_text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        expected = [
            "word1 word2 word3 word4 word5 word6",
            "word5 word6 word7 word8 word9 word10",
            "word9 word10",
        ]
        observed = get_chunks(big_text, chunk_size=max_length, overlap_percent=overlap_percent)
        assert observed == expected

    def test_2(self):
        max_length = 6
        overlap_percent = 0.33
        big_text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11"
        expected = [
            "word1 word2 word3 word4 word5 word6",
            "word5 word6 word7 word8 word9 word10",
            "word9 word10 word11",
        ]
        observed = get_chunks(big_text, chunk_size=max_length, overlap_percent=overlap_percent)
        assert observed == expected

    def test_almost_zero_overlap(self):
        max_length = 6
        overlap_percent = 0.000001
        big_text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11"
        expected = [
            "word1 word2 word3 word4 word5 word6",
            "word6 word7 word8 word9 word10 word11",
            "word11",
        ]
        observed = get_chunks(big_text, chunk_size=max_length, overlap_percent=overlap_percent)
        assert observed == expected

    def test_half_overlap(self):
        max_length = 2
        overlap_percent = 0.5
        big_text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11"
        expected = [
            "word1 word2",
            "word2 word3",
            "word3 word4",
            "word4 word5",
            "word5 word6",
            "word6 word7",
            "word7 word8",
            "word8 word9",
            "word9 word10",
            "word10 word11",
            "word11",
        ]
        observed = get_chunks(big_text, chunk_size=max_length, overlap_percent=overlap_percent)
        assert observed == expected
