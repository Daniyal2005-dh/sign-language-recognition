"""Unit tests for the WordCommitter state machine."""
from signai.predict import WordCommitter


def make_committer():
    return WordCommitter(stable_frames=3, cooldown_s=1.0, gap_frames=2, min_score=0.6)


def test_commits_after_stable_frames():
    c = make_committer()
    assert c.update("Open_Palm", 0.9, now=0.0) is None
    assert c.update("Open_Palm", 0.9, now=0.1) is None
    assert c.update("Open_Palm", 0.9, now=0.2) == "Open_Palm"


def test_same_word_requires_gap_before_recommit():
    c = make_committer()
    for i in range(3):
        c.update("Open_Palm", 0.9, now=0.1 * i)
    for i in range(6):                              # still no gap -> no second commit
        assert c.update("Open_Palm", 0.9, now=1.0 + 0.1 * i) is None
    c.update("None", 0.9, now=2.0)                  # gap frames re-arm the word
    c.update("None", 0.9, now=2.1)
    assert c.update("Open_Palm", 0.9, now=2.2) is None
    assert c.update("Open_Palm", 0.9, now=2.3) is None
    assert c.update("Open_Palm", 0.9, now=3.2) == "Open_Palm"  # cooldown passed too


def test_different_word_commits_without_gap():
    c = make_committer()
    for i in range(3):
        c.update("Open_Palm", 0.9, now=0.1 * i)
    for i in range(3):
        out = c.update("Victory", 0.9, now=1.5 + 0.1 * i)
    assert out == "Victory"


def test_low_score_counts_as_gap():
    c = make_committer()
    assert c.update("Open_Palm", 0.5, now=0.0) is None   # below min_score
    assert c.update("None", 0.9, now=0.1) is None
    c.candidate = "Open_Palm"
    c.update("Open_Palm", 0.2, now=0.2)
    assert c.candidate is None                          # reset by low score


def test_reset_clears_state():
    c = make_committer()
    c.update("Victory", 0.9, now=0.0)
    c.reset()
    assert c.candidate is None and c.count == 0
