import pytest

from trimbed.remap import IdRemap


def test_from_kept_numbers_sorted_ids_from_zero():
    remap = IdRemap.from_kept([7, 2, 5, 2])
    assert remap.new_to_old == (2, 5, 7)
    assert remap.old_to_new == {2: 0, 5: 1, 7: 2}
    assert len(remap) == 3


def test_from_kept_preserves_relative_order():
    remap = IdRemap.from_kept([100, 3, 50])
    olds = [remap.to_old(i) for i in range(len(remap))]
    assert olds == sorted(olds)


def test_membership_and_iteration():
    remap = IdRemap.from_kept([4, 9])
    assert 4 in remap
    assert 5 not in remap
    assert list(remap) == [4, 9]


def test_map_sequence_returns_none_when_a_token_was_dropped():
    remap = IdRemap.from_kept([1, 2, 3])
    assert remap.map_sequence([3, 1]) == [2, 0]
    assert remap.map_sequence([1, 99]) is None


def test_empty_and_negative_are_rejected():
    with pytest.raises(ValueError, match="empty"):
        IdRemap.from_kept([])
    with pytest.raises(ValueError, match="non-negative"):
        IdRemap.from_kept([-1, 2])


def test_from_vocabularies_matches_on_token_strings():
    old = {"a": 0, "b": 1, "c": 2, "d": 3}
    new = {"a": 0, "c": 1, "d": 2}
    remap = IdRemap.from_vocabularies(old, new)
    assert remap.new_to_old == (0, 2, 3)
    assert remap.to_new(2) == 1
    assert 1 not in remap


def test_from_vocabularies_rejects_tokens_the_original_lacked():
    with pytest.raises(ValueError, match="unknown tokens"):
        IdRemap.from_vocabularies({"a": 0}, {"a": 0, "surprise": 1})


def test_from_vocabularies_rejects_non_contiguous_ids():
    with pytest.raises(ValueError, match="outside"):
        IdRemap.from_vocabularies({"a": 0, "b": 1}, {"a": 0, "b": 7})
