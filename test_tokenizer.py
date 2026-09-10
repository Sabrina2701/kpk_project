import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tokenizer import VOCAB, generate_move_geometry, SPECIAL_TOKENS


def test_vocab_size_is_small():

    assert VOCAB.size < 2500, f"vocab unexpectedly large: {VOCAB.size}"


def test_promoted_piece_moves_are_covered():
  
    for uci in ["g7c3", "b1a3", "h8a1", "c1h6"]:
        assert uci in VOCAB.stoi, f"{uci} missing from vocab (promoted-piece move)"


def test_no_duplicate_tokens():
    geo = generate_move_geometry()
    assert len(geo) == len(set(geo))


def test_special_tokens_present_and_first():
    for i, tok in enumerate(SPECIAL_TOKENS):
        assert VOCAB.itos[i] == tok


def test_encode_decode_roundtrip():
    for uci in ["e2e4", "e7e5", "e7e8q", "a1h1", "a1a8", "e4d5"]:
        idx = VOCAB.encode(uci)
        assert VOCAB.decode(idx) == uci


def test_unknown_move_raises():
    try:
        VOCAB.encode("z9z9")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_deterministic_across_calls():
    v1 = generate_move_geometry()
    v2 = generate_move_geometry()
    assert v1 == v2, "vocab must be deterministic so KPK/KRK models share ids (needed for M6)"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"{t.__name__}: OK")
    print(f"\n{len(tests)} tests passed. Vocab size = {VOCAB.size}")
