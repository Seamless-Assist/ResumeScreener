from sa_candidate_finder.cli import should_expand_global_pool


def test_fast_rerank_never_expands_to_global_pool():
    assert not should_expand_global_pool(
        rerank_fast=True,
        good_match_count=0,
        target_good_matches=20,
    )


def test_full_search_expands_when_shortlist_is_below_target():
    assert should_expand_global_pool(
        rerank_fast=False,
        good_match_count=6,
        target_good_matches=20,
    )


def test_full_search_skips_expansion_when_target_is_met():
    assert not should_expand_global_pool(
        rerank_fast=False,
        good_match_count=20,
        target_good_matches=20,
    )
