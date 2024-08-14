from src.fraud_detection.utils.encoder_utils import _pad_sentence


def test_pad_sentence_single_word():
    sentence = ['single']
    max_sentence_length = 5
    expected_result = ['single', '', '', '', '']
    assert _pad_sentence(sentence, max_sentence_length) == expected_result

def test_pad_sentence_long_sentence():
    sentence = ['this', 'is', 'a', 'very', 'long', 'sentence']
    max_sentence_length = 3
    expected_result = ['this', 'is', 'a']
    assert _pad_sentence(sentence, max_sentence_length) == expected_result

def test_pad_sentence_exact_length():
    sentence = ['exact', 'length']
    max_sentence_length = 2
    expected_result = ['exact', 'length']
    assert _pad_sentence(sentence, max_sentence_length) == expected_result

def test_pad_sentence_empty_sentence():
    sentence = []
    max_sentence_length = 5
    expected_result = ['', '', '', '', '']
    assert _pad_sentence(sentence, max_sentence_length) == expected_result

def test_pad_sentence_sentence_with_max_length():
    sentence = ['a', 'b', 'c', 'd', 'e']
    max_sentence_length = 5
    expected_result = ['a', 'b', 'c', 'd', 'e']
    assert _pad_sentence(sentence, max_sentence_length) == expected_result