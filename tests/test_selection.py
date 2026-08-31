from collections import Counter

import pytest

from trimbed.config import SelectionConfig
from trimbed.counting import CorpusCounts
from trimbed.selection import Selection, _apply_cap, select_tokens
from trimbed.spec import TokenizerSpec


@pytest.fixture
def spec(wordpiece):
    return TokenizerSpec.from_tokenizer(wordpiece)


@pytest.fixture
def counts(spec):
    ids = spec.vocabulary
    return CorpusCounts(
        counts=Counter({ids["the"]: 100, ids["cat"]: 50, ids["sat"]: 10, ids["dog"]: 1}),
        total_num_tokens=161,
        num_documents=4,
    )


def test_structural_tokens_are_always_kept(spec, counts):
    selection = select_tokens(spec, counts, SelectionConfig(top_k=1))
    assert spec.vocabulary["[UNK]"] in selection.kept_ids
    assert selection.structural_ids <= selection.kept_ids


def test_post_processor_tokens_are_structural(wordlevel_undeclared_post_processor):
    # Read from the document, nothing promotes `<sep>` to an added token, so the
    # post-processor is the only thing that marks it as load-bearing.
    spec = TokenizerSpec.from_json_str(wordlevel_undeclared_post_processor.backend_tokenizer.to_str())

    selection = select_tokens(spec, None, SelectionConfig(keep_tokens=["the"]))

    assert spec.vocabulary["<sep>"] in selection.structural_ids
    assert spec.vocabulary["<sep>"] in selection.kept_ids


def test_top_k_keeps_the_most_frequent(spec, counts):
    selection = select_tokens(spec, counts, SelectionConfig(top_k=2))
    kept = {spec.id_to_token[i] for i in selection.kept_ids}
    assert {"the", "cat"} <= kept
    assert "dog" not in kept


def test_min_count_filters_the_tail(spec, counts):
    kept = {spec.id_to_token[i] for i in select_tokens(spec, counts, SelectionConfig(min_count=10)).kept_ids}
    assert {"the", "cat", "sat"} <= kept
    assert "dog" not in kept


def test_coverage_cuts_the_ranking_at_a_cumulative_fraction(spec, counts):
    # 100/161 is ~62%, so a 0.6 target is met by "the" alone.
    kept = {spec.id_to_token[i] for i in select_tokens(spec, counts, SelectionConfig(coverage=0.6)).kept_ids}
    assert "the" in kept
    assert "cat" not in kept


def test_coverage_keeps_the_whole_ranking_when_the_target_is_unreachable(spec):
    # A cached counts file can carry a total the per-token counts no longer add up to,
    # e.g. when it was produced against a wider corpus slice. The cut must then keep
    # everything rather than silently keeping nothing.
    partial = CorpusCounts(
        counts=Counter({spec.vocabulary["the"]: 1, spec.vocabulary["cat"]: 1}), total_num_tokens=1000
    )
    selection = select_tokens(spec, partial, SelectionConfig(coverage=1.0))

    assert {spec.vocabulary["the"], spec.vocabulary["cat"]} <= selection.kept_ids


def test_combined_criteria_take_the_strictest(spec, counts):
    both = select_tokens(spec, counts, SelectionConfig(coverage=1.0, top_k=1))
    only_k = select_tokens(spec, counts, SelectionConfig(top_k=1))
    assert both.kept_ids == only_k.kept_ids


def test_presets_act_as_a_floor_not_a_filter(spec, counts):
    without = select_tokens(spec, counts, SelectionConfig(top_k=1))
    with_preset = select_tokens(spec, counts, SelectionConfig(top_k=1, keep_presets=["single_characters"]))
    assert without.kept_ids < with_preset.kept_ids
    assert spec.vocabulary["z"] in with_preset.kept_ids


def test_provenance_records_why_each_token_survived(spec, counts):
    selection = select_tokens(spec, counts, SelectionConfig(top_k=1, keep_tokens=["dog"]))
    assert "explicit_token" in selection.provenance[spec.vocabulary["dog"]]
    assert "corpus" in selection.provenance[spec.vocabulary["the"]]
    assert selection.counts_by_reason()["explicit_token"] == 1


def test_requested_tokens_absent_from_the_vocabulary_are_reported_not_fatal(spec, counts):
    selection = select_tokens(spec, counts, SelectionConfig(top_k=1, keep_tokens=["kangaroo"]))
    assert selection.unknown_tokens == ["kangaroo"]


def test_an_out_of_range_token_id_is_fatal(spec, counts):
    with pytest.raises(ValueError, match="does not exist"):
        select_tokens(spec, counts, SelectionConfig(top_k=1, keep_token_ids=[99999]))


def test_keep_token_ids_and_files(spec, counts, tmp_path):
    path = tmp_path / "keep.txt"
    path.write_text("# a comment\n\nrun\ndog\n", encoding="utf-8")
    selection = select_tokens(
        spec,
        counts,
        SelectionConfig(top_k=1, keep_token_files=[path]),
    )
    assert spec.vocabulary["run"] in selection.kept_ids
    assert f"file:{path}" in selection.provenance[spec.vocabulary["run"]]


def test_keep_patterns_match_the_surface_form(spec, counts):
    selection = select_tokens(spec, counts, SelectionConfig(top_k=1, keep_patterns=[r"^ing$"]))
    # "##ing" has surface form "ing", so the anchored pattern must still match it.
    assert spec.vocabulary["##ing"] in selection.kept_ids


def test_selection_without_a_corpus_uses_explicit_sources_only(spec):
    selection = select_tokens(spec, None, SelectionConfig(keep_tokens=["cat"]))
    assert spec.vocabulary["cat"] in selection.kept_ids
    assert selection.kept_ids == selection.structural_ids | {spec.vocabulary["cat"]}


def test_the_cap_shrinks_the_selection(spec, counts):
    uncapped = select_tokens(spec, counts, SelectionConfig(min_count=1))
    target = len(uncapped) - 2
    assert len(select_tokens(spec, counts, SelectionConfig(min_count=1, max_vocab_size=target))) == target


def test_the_cap_reports_requested_tokens_it_had_to_drop(spec, counts):
    structural = len(select_tokens(spec, counts, SelectionConfig(keep_tokens=["dog"])).structural_ids)
    capped = select_tokens(spec, counts, SelectionConfig(keep_tokens=["dog"], max_vocab_size=structural))
    assert spec.vocabulary["dog"] in capped.dropped_requested
    assert capped.dropped_requested[spec.vocabulary["dog"]] == {"explicit_token"}


def test_a_cap_below_the_structural_floor_is_fatal(spec, counts):
    with pytest.raises(ValueError, match="structural"):
        select_tokens(spec, counts, SelectionConfig(min_count=1, max_vocab_size=1))


PARENT, CHILD_A, CHILD_B, FREQUENT = 0, 1, 2, 3
SHARED_PARENT = {CHILD_A: (PARENT,), CHILD_B: (PARENT,)}


def capped(dependencies, max_vocab_size, structural=frozenset()):
    """Cap a four-token selection over `dependencies` and return what survives."""
    selection = Selection(kept_ids={PARENT, CHILD_A, CHILD_B, FREQUENT}, structural_ids=set(structural))
    # The parent is in the selection for reachability only, so it has no count at all.
    counts = CorpusCounts(counts=Counter({CHILD_A: 1, CHILD_B: 2, FREQUENT: 100}))
    _apply_cap(selection, counts, dependencies, max_vocab_size)
    return selection.kept_ids


class TestCapDependencyCounts:
    """The cap frees a parent exactly when its *last* dependent goes, and not before.

    Written against a bare dependency graph rather than a tokenizer so that the shared
    parent, the repeated parent and the structural parent are each visible on their own.
    The counter tracking this used to be built and never updated, which left every parent
    that ever had a dependent pinned in the selection: the cap then met its target by
    taking a frequent token instead, and only a test on identity notices that.
    """

    def test_a_parent_stays_while_another_dependent_needs_it(self):
        assert capped(SHARED_PARENT, max_vocab_size=3) == {PARENT, CHILD_B, FREQUENT}

    def test_a_freed_parent_goes_before_a_frequent_token(self):
        # With both children gone the parent is unreachable dead weight, so it must be
        # taken before the token the corpus is full of.
        assert capped(SHARED_PARENT, max_vocab_size=1) == {FREQUENT}

    def test_a_parent_named_twice_by_one_merge_is_freed_by_that_one_removal(self):
        # A merge can be built from the same token twice ("e" + "e" -> "ee"), so the
        # bookkeeping has to come back down by two when that single child goes.
        assert capped({CHILD_A: (PARENT, PARENT)}, max_vocab_size=2) == {CHILD_B, FREQUENT}

    def test_a_structural_parent_is_never_freed(self):
        # Reaching zero dependents makes a byte-alphabet character removable on paper;
        # removing it makes some byte sequences unencodable.
        assert capped(SHARED_PARENT, max_vocab_size=1, structural={PARENT}) == {PARENT}

    def test_a_cap_that_cannot_be_met_gives_up_rather_than_orphaning(self, package_logs):
        # A merge is always larger than its parts so the real graph cannot cycle, but a
        # backend whose graph does has no leaf to start from and nothing drops to zero
        # dependents. Missing the cap is the safe outcome; an unencodable vocabulary is not.
        cycle = {PARENT: (CHILD_A,), CHILD_A: (CHILD_B,), CHILD_B: (FREQUENT,), FREQUENT: (PARENT,)}

        kept = capped(cycle, max_vocab_size=1)

        assert kept == {PARENT, CHILD_A, CHILD_B, FREQUENT}
        assert "could not shrink below" in package_logs.text


class TestBpeDependencies:
    """Dependency closure is what stops a kept token becoming unreachable."""

    @pytest.fixture
    def spec(self, byte_level_bpe):
        return TokenizerSpec.from_tokenizer(byte_level_bpe)

    @pytest.fixture
    def counts(self, spec, byte_level_bpe):
        totals = CorpusCounts()
        for text in ["the cat", "the dog"]:
            ids = byte_level_bpe(text, add_special_tokens=False)["input_ids"]
            totals.counts.update(ids)
            totals.total_num_tokens += len(ids)
        return totals

    def test_merge_ancestors_are_pulled_in(self, spec, counts):
        selection = select_tokens(spec, counts, SelectionConfig(min_count=1))
        # "the" is only reachable if "he" survives too, even though the corpus never
        # emits "he" as a token in its own right.
        assert spec.vocabulary["the"] in selection.kept_ids
        assert spec.vocabulary["he"] in selection.kept_ids
        assert "dependency" in selection.provenance[spec.vocabulary["he"]]

    def test_the_cap_removes_in_frequency_order_as_parents_come_free(self, spec, byte_level_bpe):
        # Counts: "at" 3, "the" 2, "Ġd" 1, and "he" 0 because it is only in the selection
        # to keep "the" reachable. Dropping "the" therefore frees "he", which has to go
        # ahead of "at": a parent that stops being needed is worth less than a token the
        # corpus actually uses.
        counts = CorpusCounts()
        ids = byte_level_bpe("the cat sat at the dog", add_special_tokens=False)["input_ids"]
        counts.counts.update(ids)
        counts.total_num_tokens = len(ids)
        uncapped = select_tokens(spec, counts, SelectionConfig(min_count=1))

        removal_order = []
        previous = uncapped.kept_ids
        for dropped in range(1, 5):
            capped = select_tokens(spec, counts, SelectionConfig(min_count=1, max_vocab_size=len(uncapped) - dropped))
            (token_id,) = previous - capped.kept_ids
            removal_order.append(spec.id_to_token[token_id])
            previous = capped.kept_ids

        assert removal_order == ["Ġd", "the", "he", "at"]

    def test_the_cap_never_orphans_a_kept_token(self, spec, counts):
        selection = select_tokens(spec, counts, SelectionConfig(min_count=1, max_vocab_size=261))
        dependencies = spec.backend.dependencies(spec)
        for token_id in selection.kept_ids:
            for parent in dependencies.get(token_id, ()):
                assert parent in selection.kept_ids
