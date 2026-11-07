"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

import os
import time
import traceback
from itertools import combinations, permutations

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import get_table, warmup
from game import RAISE_ACTIONS, Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# rename as your team/bot's name
PARTICIPANT = "hack the hacker"
AGENT_NAME = "hack ag"


# ===== YOUR CODE HERE =====

# ----- HANDOFF NOTES (work in progress, 2026-11-07) ---------------------------
# WHY: a full 1000-deal judge match vs baseline_random (original code) won
# +4.29 bb/deal, but our bot emptied its 8 s time bank in BOTH halves
# (342 / 636 calls auto-checked/folded). Our own thinking alone was 6-9 s per
# half, plus the judge charges HTTP time per request (only the first 1 ms is
# free). On this Windows laptop that overhead was ~3 ms per call: even the
# random bot was charged 6.7 s. So the work below is mostly about speed.
#
# DONE (code in this file):
#   1. Preflop is a table lookup: preflop_table.npz (1.3 MB, must be
#      committed), built by tools/build_preflop_table.py (~18 min on 2 cores).
#      19,998 suit-canonical hands x 32 opponent-strength buckets, from
#      900 boards x 1350 hands. Checked: matches the old live sampling (mean
#      equity diff -0.003). If the file is missing, the old live sampling runs.
#      REBUILD the table if you change _strength / equity / PREFLOP_BUCKETS.
#   2. _strength vectorized (one sort instead of a loop). Checked: identical
#      results to the old loop.
#   3. Postflop opponent hands come from a fixed random pool (_OPP_POOL) and
#      runouts from _runout_pairs, instead of argpartition every decision.
#   4. SAMPLES cut from flop (16,250)/turn (14,300)/river (1,800) to
#      (10,160)/(10,180)/(1,500).
#   5. Pacing (_time_scale) also counts CALL_OVERHEAD per request (an estimate
#      of judge-charged HTTP time). Scale 0.0 = on turn/river, reuse the
#      earlier street's analysis instead of simulating (emergency path).
#   6. Showdown learning (_learn_from_showdown): after each showdown, measure
#      how strong the opponent's revealed hand really was on each street where
#      their actions narrowed the range, and fit a per-opponent bluff_mix by
#      maximum likelihood over BLUFF_GRID (shrunk toward BLUFF_MIX by
#      SHOWDOWN_PRIOR). Checked in a 250-hand test: maniac -> 0.54, calling
#      station -> 0.54 (wide calling range), nit -> stays 0.07.
#      on_hand_end now also records that the opponent CALLED our last raise
#      when the hand reached showdown (HandResult has no action list).
#   7. tuned_params.json loader (_apply_tuned_params, bottom of this section)
#      + tools/self_improve.py: offline hill-climbing of the TUNABLE knobs with
#      paired arena matches. Learning cannot carry across tournament matches
#      (fresh process + wiped memory every half), so it has to happen offline
#      and be committed.
#
# NOT YET VERIFIED (do these next, in this order):
#   a. python -m pytest -q  (passed BEFORE these changes; not re-run since).
#   b. Time budget: last arena run (before change 5's reuse path) still used
#      2.6-6.7 s thinking per half (worst: vs trapper). Re-check with
#        python tools/arena.py --deals 1000
#      (use --deals 1000: with fewer deals the pacing targets a shorter half
#      and s/1000 looks inflated). Laptop timings swing 2-3x, so repeat runs.
#   c. Strength vs the ORIGINAL bot: git show ffe27c2:agent.py > old_agent.py,
#        python tools/arena.py --deals 1000 --only other_file --against old_agent.py
#      (--only must include other_file or the head-to-head is skipped).
#   d. Real judge, full match, every baseline, several seeds. On Windows use
#      127.0.0.1, not localhost (the judge could not connect via localhost):
#        python judge_engine.py --agent1 http://127.0.0.1:5008 --agent2 http://127.0.0.1:5009
#      Goal: 0 errors and no "TIME BANK EMPTY".
#   e. tools/self_improve.py has never been run. Try --rounds 2 --dry-run first.
#   f. tools/check_build.sh (needs Docker + bash) before submitting.
#
# OPEN QUESTIONS / IDEAS:
#   - CALL_OVERHEAD = 1.5 ms is a guess for the tournament's Linux box; the
#     original author saw ~1 ms/call locally. Too low -> bank runs out; too high
#     -> needlessly weak samples.
#   - Showdown learning only covers flop/turn/river (preflop skipped), and is
#     biased: we never see hands the opponent folded later.
#   - Finer opponent stats (per street / position, check-raises) not started.
# -----------------------------------------------------------------------------

#
# Strategy in one paragraph: every decision compares the expected value of
# folding, calling and raising. Equity is measured against a RANGE: we sample
# opponent hands, score them on sampled runouts, rank them, and weight them by
# how narrow this opponent's actions (learned this half) say their range is.
# Fold rates learned the same way decide how much a raise wins outright.
#
# Speed: _T[triple_id, pair_id] holds the score of every 3-board + 2-hole
# combination, so numpy scores thousands of hands per millisecond. A half has
# 8 s of time bank; the pacing code below keeps well under that.

N_CARDS = 36
_HOLE_I, _HOLE_J = map(np.array, zip(*combinations(range(5), 2)))
_BOARD_A, _BOARD_B, _BOARD_C = map(np.array, zip(*combinations(range(5), 3)))
_BIT = np.int64(1) << np.arange(N_CARDS, dtype=np.int64)
_N_PAIRS = 630  # C(36, 2)

# Tuning knobs (tools/arena.py measures the effect of a change).
BLUFF_MIX = 0.07          # share of any opponent range we treat as air
RANGE_SOFTNESS = 0.08     # width of the soft cut-off in a range
PRIOR_STREET_DECAY = 0.3  # how much earlier streets still narrow the range
RERAISE_PENALTY = 0.02    # equity shaved off raises that aren't all-in
BET_MARGIN = 0.05         # a raise must beat the alternatives by this many chips
PRIOR_WEIGHT = 5.0        # opponent stats: pseudo-observations behind each prior
PRIOR_FOLD = (0.30, 0.35)     # (preflop, postflop) folds to a raise
PRIOR_RAISE = (0.20, 0.15)    # (preflop, postflop) re-raises when facing a raise
PRIOR_OPEN = (0.40, 0.35)     # (preflop, postflop) raises when not facing a raise
# Share of raw equity we expect to realise: (in position, out of position).
REALIZE = {
    Street.PREFLOP: (1.00, 0.92),
    Street.FLOP: (1.00, 0.93),
    Street.TURN: (1.00, 0.97),
    Street.RIVER: (1.00, 1.00),
}
# Sample sizes per street: (runouts, opponent hands).
SAMPLES = {Street.FLOP: (10, 160), Street.TURN: (10, 180), Street.RIVER: (1, 500)}
# Preflop is a table lookup (tools/build_preflop_table.py writes the file).
PREFLOP_TABLE_FILE = "preflop_table.npz"
PREFLOP_BUCKETS = 32
# Our clock only sees choose_action, but the judge charges each request's whole
# round trip minus 1 ms. CALL_OVERHEAD is our estimate of the HTTP and JSON time
# it charges per request beyond that, so the pacing below budgets for it too.
CALL_OVERHEAD = 0.0015
TIME_TARGET = 5.0     # charged seconds per half we aim to stay under (bank: 8)
TIME_PANIC = 6.3      # beyond this, spend almost nothing per decision


def _build_score_table():
    """Return (T, pair_id, triple_id) built from the evaluator's table."""
    pair_id = np.zeros((N_CARDS, N_CARDS), dtype=np.int32)
    for i, (a, b) in enumerate(combinations(range(N_CARDS), 2)):
        pair_id[a, b] = pair_id[b, a] = i
    triple_id = np.zeros((N_CARDS, N_CARDS, N_CARDS), dtype=np.int32)
    n_triples = 0
    for n_triples, cards in enumerate(combinations(range(N_CARDS), 3), start=1):
        for perm in permutations(cards):
            triple_id[perm] = n_triples - 1
    hands = np.array(list(combinations(range(N_CARDS), 5)), dtype=np.int64)
    masks = _BIT[hands].sum(axis=1).tolist()
    table = get_table()
    scores = np.fromiter(map(table.__getitem__, masks), dtype=np.int32, count=len(masks))
    # A 0 entry means the pair and triple share a card (an impossible hand).
    score_table = np.zeros((n_triples, _N_PAIRS), dtype=np.int32)
    for i, j in combinations(range(5), 2):
        a, b, c = (k for k in range(5) if k not in (i, j))
        score_table[triple_id[hands[:, a], hands[:, b], hands[:, c]],
                    pair_id[hands[:, i], hands[:, j]]] = scores
    return score_table, pair_id, triple_id


_T, _PAIR_ID, _TRIPLE_ID = _build_score_table()
_RNG = np.random.default_rng(20261107)


def _pair_ids(hands: np.ndarray) -> np.ndarray:
    """(..., 5) hole cards -> (..., 10) pair ids."""
    return _PAIR_ID[hands[..., _HOLE_I], hands[..., _HOLE_J]]


def _pair_scores(boards: np.ndarray) -> np.ndarray:
    """(K, 5) full boards -> (K, 630): the best score each hole pair makes."""
    triples = _TRIPLE_ID[boards[:, _BOARD_A], boards[:, _BOARD_B], boards[:, _BOARD_C]]
    return _T[triples].max(axis=1)


def _sample_cards(deck: np.ndarray, rows: int, count: int) -> np.ndarray:
    """(rows, count) cards drawn from deck, distinct within each row."""
    return deck[_RNG.random((rows, deck.size)).argpartition(count - 1, axis=1)[:, :count]]


def _strength(scores: np.ndarray, valid: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Average percentile of each opponent hand across the runouts (0..1)."""
    rows, cols = scores.shape
    # Shift each runout's scores into its own range (scores < 2**24), so one
    # searchsorted over the flattened, sorted rows ranks every runout at once.
    offset = (np.arange(rows, dtype=np.int64) << 25)[:, None]
    masked = np.where(valid, scores, -1).astype(np.int64) + offset
    ordered = np.sort(masked, axis=1).ravel()
    invalid = cols - valid.sum(axis=1)
    ranks = np.searchsorted(ordered, masked.ravel()).reshape(rows, cols)
    ranks -= np.arange(rows)[:, None] * cols + invalid[:, None]
    share = np.where(valid, ranks, 0) / np.maximum(cols - invalid, 1)[:, None]
    return share.sum(axis=0) / np.maximum(counts, 1)


# Opponent hands for postflop sampling: a fixed pool of random hands, filtered
# by the cards we can see. Far cheaper than drawing fresh hands every decision.
_OPP_POOL = _sample_cards(np.arange(N_CARDS), 20000, 5)
_OPP_POOL_MASKS = _BIT[_OPP_POOL].sum(axis=1)
_PAIRS_OF: dict[int, np.ndarray] = {}   # deck size -> every pair of positions


def _opponent_hands(known_mask: np.int64, count: int) -> np.ndarray:
    """count random opponent hands that avoid the known cards."""
    fits = np.flatnonzero((_OPP_POOL_MASKS & known_mask) == 0)
    start = int(_RNG.integers(fits.size))
    return _OPP_POOL[fits[(start + np.arange(min(count, fits.size))) % fits.size]]


def _runout_pairs(deck: np.ndarray, count: int) -> np.ndarray:
    """count distinct 2-card runouts from deck."""
    pairs = _PAIRS_OF.get(deck.size)
    if pairs is None:
        pairs = _PAIRS_OF[deck.size] = np.array(list(combinations(range(deck.size), 2)))
    picked = _RNG.choice(len(pairs), size=min(count, len(pairs)), replace=False)
    return deck[pairs[picked]]


# ----- Preflop table -----------------------------------------------------------
# Renaming suits never changes a hand's equity, so a hand is stored under its
# canonical key: the smallest encoding over all 24 suit renamings.

_SUIT_PERMS = np.array(list(permutations(range(4))), dtype=np.int64)  # (24, 4)
_KEY_WEIGHTS = N_CARDS ** np.arange(4, -1, -1, dtype=np.int64)        # base-36 digits


def _canonical_keys(hands: np.ndarray) -> np.ndarray:
    """(..., 5) hands -> (...) canonical keys."""
    ranks, suits = hands >> 2, hands & 3
    renamed = np.sort(ranks[..., None, :] * 4 + _SUIT_PERMS[:, suits].swapaxes(0, -2), axis=-1)
    return (renamed @ _KEY_WEIGHTS).min(axis=-1)


def _all_canonical_hands() -> tuple[np.ndarray, np.ndarray]:
    """Sorted canonical keys, and the hand each key stands for."""
    hands = np.array(list(combinations(range(N_CARDS), 5)), dtype=np.int64)
    keys = np.unique(np.concatenate([_canonical_keys(chunk) for chunk in np.array_split(hands, 20)]))
    reps = (keys[:, None] // _KEY_WEIGHTS) % N_CARDS
    return keys, reps


def _load_preflop_table():
    try:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), PREFLOP_TABLE_FILE)
        with np.load(path) as data:
            table = {"keys": data["keys"], "equity": data["equity"].astype(np.float64) / 65535,
                     "strength": data["strength"].astype(np.float64)}
        if table["equity"].shape != (table["keys"].size, PREFLOP_BUCKETS):
            return None
        return table
    except Exception:
        return None  # fall back to live preflop sampling


def _build_preflop_pool(n_boards: int = 300, n_hands: int = 600) -> dict:
    """Fixed boards and opponent hands, so preflop needs no sampling at all."""
    every_card = np.arange(N_CARDS)
    boards = _sample_cards(every_card, n_boards, 5)
    hands = _sample_cards(every_card, n_hands, 5)
    pair_scores = _pair_scores(boards)
    scores = pair_scores[:, _pair_ids(hands)].max(axis=2)
    board_masks = _BIT[boards].sum(axis=1)
    hand_masks = _BIT[hands].sum(axis=1)
    valid = (board_masks[:, None] & hand_masks[None, :]) == 0
    return {
        "pair_scores": pair_scores,
        "scores": scores,
        "valid": valid,
        "board_masks": board_masks,
        "hand_masks": hand_masks,
        "strength": _strength(scores, valid, valid.sum(axis=0)),
    }


_PREFLOP = _load_preflop_table()
# Live sampling is only the fallback for a missing or broken table file.
_POOL = _build_preflop_pool() if _PREFLOP is None else None


def _preflop_analysis(hole: np.ndarray, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Equity and strength per opponent-strength bucket (or per sampled hand)."""
    if _PREFLOP is not None:
        key = _canonical_keys(hole)
        row = int(np.searchsorted(_PREFLOP["keys"], key))
        if row < _PREFLOP["keys"].size and _PREFLOP["keys"][row] == key:
            return _PREFLOP["equity"][row], _PREFLOP["strength"]
    return _sampled_preflop_analysis(hole, scale)


def _sampled_preflop_analysis(hole: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    global _POOL
    if _POOL is None:
        _POOL = _build_preflop_pool()
    my_mask = _BIT[hole].sum()
    boards_ok = (_POOL["board_masks"] & my_mask) == 0
    hands_ok = (_POOL["hand_masks"] & my_mask) == 0
    if scale < 1.0:  # short on time: look at every third board only
        boards_ok[np.arange(boards_ok.size) % 3 != 0] = False
    mine = _POOL["pair_scores"][boards_ok][:, _pair_ids(hole)].max(axis=1)[:, None]
    opp = _POOL["scores"][boards_ok][:, hands_ok]
    valid = _POOL["valid"][boards_ok][:, hands_ok]
    points = ((mine > opp) + 0.5 * (mine == opp)) * valid
    return points.sum(axis=0) / np.maximum(valid.sum(axis=0), 1), _POOL["strength"][hands_ok]


def _postflop_analysis(hole: np.ndarray, board: np.ndarray, street: Street,
                       scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-opponent-hand equity and strength on a flop, turn or river."""
    used = np.zeros(N_CARDS, dtype=bool)
    used[hole] = True
    used[board] = True
    deck = np.flatnonzero(~used)
    runouts, n_hands = SAMPLES[street]
    runouts = max(1, min(int(runouts * scale + 0.5), deck.size))
    n_hands = max(40, int(n_hands * scale))
    missing = 5 - board.size
    if missing == 0:
        boards = board[None, :]
    else:
        boards = np.empty((runouts, 5), dtype=np.int64)
        boards[:, :board.size] = board
        if missing == 1:
            boards[:, 4] = _RNG.choice(deck, size=runouts, replace=False)
        else:
            boards[:, board.size:] = _runout_pairs(deck, runouts)
    hands = _opponent_hands(_BIT[hole].sum() + _BIT[board].sum(), n_hands)
    pair_scores = _pair_scores(boards)
    opp = pair_scores[:, _pair_ids(hands)].max(axis=2)
    mine = pair_scores[:, _pair_ids(hole)].max(axis=1)[:, None]
    runout_masks = _BIT[boards[:, board.size:]].sum(axis=1)
    valid = (runout_masks[:, None] & _BIT[hands].sum(axis=1)[None, :]) == 0
    counts = valid.sum(axis=0)
    points = ((mine > opp) + 0.5 * (mine == opp)) * valid
    return points.sum(axis=0) / np.maximum(counts, 1), _strength(opp, valid, counts)


# ----- Opponent model --------------------------------------------------------

def _rate(hits: float, total: float, prior: float) -> float:
    return (hits + PRIOR_WEIGHT * prior) / (total + PRIOR_WEIGHT)


def _fold_rate(memory: dict, group: int, size: int) -> float:
    folds, calls, raises = memory["facing"][group][size]
    return _rate(folds, folds + calls + raises, PRIOR_FOLD[group])


def _raise_rate(memory: dict, group: int, size: int) -> float:
    folds, calls, raises = memory["facing"][group][size]
    return _rate(raises, folds + calls + raises, PRIOR_RAISE[group])


def _open_rate(memory: dict, group: int) -> float:
    raises, chances = memory["open"][group]
    return _rate(raises, chances, PRIOR_OPEN[group])


def _new_hand_state(obs: Observation) -> dict:
    my_blind, opp_blind = (1, 2) if obs.is_button else (2, 1)
    return {
        "index": obs.hand_index,
        "button": obs.is_button,
        "hole": tuple(obs.hole_cards),
        "seen": 0,                    # entries of action_history already replayed
        "street": Street.PREFLOP,
        "my_total": my_blind,
        "opp_total": opp_blind,
        "keep": [1.0, 1.0, 1.0, 1.0],  # share of hands surviving each street's actions
        "my_last": None,               # (kind, size, group, street) of our latest unanswered action
        "analysis": {},                # street -> (equity, strength)
    }


def _replay(obs: Observation, memory: dict) -> None:
    """Learn from the actions added to the history since our last decision."""
    hand = memory["hand"]
    for position in range(hand["seen"], len(obs.action_history)):
        act = obs.action_history[position]
        if act.street != hand["street"]:
            hand.update(street=act.street, my_total=0, opp_total=0)
        if act.is_me:
            hand["my_total"] = act.street_total
            continue

        group = min(int(act.street), 1)
        # The small blind's first action isn't a response to a raise.
        first_in = position == 0 and act.street == Street.PREFLOP
        facing = hand["opp_total"] < hand["my_total"] and not first_in
        raised = act.action in RAISE_ACTIONS
        if facing:
            size = hand["my_last"][1] if hand["my_last"] else 0
            memory["facing"][group][size][2 if raised else 1] += 1
            keep = _raise_rate(memory, group, size) if raised else 1.0 - _fold_rate(memory, group, size)
        else:
            memory["open"][group][0] += raised
            memory["open"][group][1] += 1
            keep = _open_rate(memory, group) if raised else 1.0
        hand["keep"][int(act.street)] *= max(keep, 0.05)
        hand["opp_total"] = act.street_total
        hand["my_last"] = None
    hand["seen"] = len(obs.action_history)


def _range_keep(hand: dict, street: Street) -> float:
    """The share of all hands the opponent's line so far is consistent with."""
    keep = hand["keep"][int(street)]
    for earlier in range(int(street)):
        keep *= hand["keep"][earlier] ** PRIOR_STREET_DECAY
    return keep


def _range_weights(strength: np.ndarray, keep: float, bluff_mix: float = None) -> np.ndarray:
    """How likely each hand is in the opponent's range: a soft cut that keeps
    the strongest `keep` share, plus a floor of bluff_mix for everything."""
    if bluff_mix is None:
        bluff_mix = BLUFF_MIX
    cut = 1.1 * (1.0 - keep) - 0.1
    x = np.clip((strength - cut) / RANGE_SOFTNESS, -30.0, 30.0)
    return bluff_mix + (1.0 - bluff_mix) / (1.0 + np.exp(-x))


# ----- Learning from showdowns ------------------------------------------------
# At a showdown we see the opponent's cards, so we can check our range model:
# on every street where their actions narrowed the range, was the hand they
# really held strong (top half) or weak (bottom half)? The model predicts the
# weak share from bluff_mix. We keep the log-likelihood of every bluff_mix on a
# grid and play the best fit, pulled toward the default while data is scarce.

BLUFF_GRID = np.linspace(0.01, 0.60, 60)
SHOWDOWN_PRIOR = 15.0      # pseudo-showdowns behind the default BLUFF_MIX
SHOWDOWN_SAMPLES = (6, 150)  # (runouts, comparison hands) per street checked


def _showdown_strength(opp: np.ndarray, board: np.ndarray, hole: np.ndarray,
                       runouts: int, n_hands: int) -> tuple[float, np.ndarray]:
    """The opponent's real hand strength on this board, and the strengths of
    comparison hands sampled the same way as in a live decision."""
    comps = _opponent_hands(_BIT[hole].sum() + _BIT[board].sum(), n_hands)
    missing = 5 - board.size
    if missing == 0:
        boards = board[None, :]
    else:
        used = np.zeros(N_CARDS, dtype=bool)
        used[hole] = used[board] = used[opp] = True
        deck = np.flatnonzero(~used)
        runouts = min(runouts, deck.size)
        boards = np.empty((runouts, 5), dtype=np.int64)
        boards[:, :board.size] = board
        if missing == 1:
            boards[:, 4] = _RNG.choice(deck, size=runouts, replace=False)
        else:
            boards[:, board.size:] = _runout_pairs(deck, runouts)
    pair_scores = _pair_scores(boards)
    comp_scores = pair_scores[:, _pair_ids(comps)].max(axis=2)
    runout_masks = _BIT[boards[:, board.size:]].sum(axis=1)
    valid = (runout_masks[:, None] & _BIT[comps].sum(axis=1)[None, :]) == 0
    counts = valid.sum(axis=0)
    opp_scores = pair_scores[:, _pair_ids(opp)].max(axis=1)[:, None]
    beaten = ((comp_scores < opp_scores) & valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
    return float(beaten.mean()), _strength(comp_scores, valid, counts)


def _learn_from_showdown(result: HandResult, hand: dict, memory: dict) -> None:
    opp = np.array(result.opp_hole, dtype=np.int64)
    hole = np.array(hand["hole"], dtype=np.int64)
    full_board = np.array(result.board, dtype=np.int64)
    runouts, n_hands = SHOWDOWN_SAMPLES
    for street in (Street.FLOP, Street.TURN, Street.RIVER):
        if hand["keep"][int(street)] >= 1.0 or full_board.size < int(street) + 2:
            continue  # the opponent's actions said nothing new on this street
        board = full_board[:int(street) + 2]
        theirs, comps = _showdown_strength(opp, board, hole, runouts, n_hands)
        weak_side = comps < np.median(comps)
        weak = theirs < np.median(comps)
        sig = _range_weights(comps, _range_keep(hand, street), 0.0)
        weights = BLUFF_GRID[:, None] + (1.0 - BLUFF_GRID[:, None]) * sig[None, :]
        p_weak = np.clip((weights * weak_side).sum(axis=1) / weights.sum(axis=1), 1e-4, 1 - 1e-4)
        memory["bluff_ll"] += np.log(p_weak if weak else 1.0 - p_weak)
        memory["bluff_n"] += 1
    n = memory["bluff_n"]
    if n:
        best = float(BLUFF_GRID[int(np.argmax(memory["bluff_ll"]))])
        memory["bluff_mix"] = (n * best + SHOWDOWN_PRIOR * BLUFF_MIX) / (n + SHOWDOWN_PRIOR)


# ----- Decisions -------------------------------------------------------------

def _time_scale(obs: Observation, memory: dict) -> float:
    """Shrink the samples if this half is on pace to overspend its time.
    0.0 means: don't simulate at all if an earlier street's analysis exists."""
    used = memory["time_used"] + memory.get("calls", 0) * CALL_OVERHEAD
    if used > TIME_PANIC:
        return 0.0
    projected = used / max(obs.hand_index, 20) * memory["num_hands"]
    if projected > TIME_TARGET * 1.3:
        return 0.0
    if projected > TIME_TARGET * 1.15:
        return 0.25
    if projected > TIME_TARGET:
        return 0.5
    return 1.0


def _analysis(obs: Observation, memory: dict) -> tuple[np.ndarray, np.ndarray]:
    cache = memory["hand"]["analysis"]
    if obs.street not in cache:
        hole = np.array(obs.hole_cards, dtype=np.int64)
        scale = _time_scale(obs, memory)
        earlier = [street for street in cache if Street.PREFLOP < street < obs.street]
        if obs.street == Street.PREFLOP:
            cache[obs.street] = _preflop_analysis(hole, max(scale, 0.3))
        elif scale == 0.0 and earlier:
            cache[obs.street] = cache[max(earlier)]  # out of time: reuse, slightly stale
        else:
            board = np.array(obs.board, dtype=np.int64)
            cache[obs.street] = _postflop_analysis(hole, board, obs.street, max(scale, 0.1))
    return cache[obs.street]


def _best_action(obs: Observation, memory: dict) -> Action:
    equity, strength = _analysis(obs, memory)
    weights = _range_weights(strength, _range_keep(memory["hand"], obs.street),
                             memory.get("bluff_mix", BLUFF_MIX))

    # tail_*[i] sums over hand i and everything stronger: raises need the
    # equity against just the part of the range that continues.
    order = np.argsort(strength)
    tail_weight = np.cumsum(weights[order][::-1])[::-1]
    tail_points = np.cumsum((weights * equity)[order][::-1])[::-1]
    total_weight = float(tail_weight[0])
    range_equity = float(tail_points[0]) / total_weight

    realize = REALIZE[obs.street][0 if obs.is_button else 1]
    to_call = obs.to_call
    group = min(int(obs.street), 1)

    values: dict[Action, float] = {}
    if Action.FOLD in obs.legal_actions:
        values[Action.FOLD] = 0.0
    share = 1.0 if to_call >= obs.my_stack else realize
    values[Action.CHECK_CALL] = share * range_equity * (obs.pot + to_call) - to_call

    for action in RAISE_ACTIONS:
        if action not in obs.legal_actions:
            continue
        added = obs.raise_to[action] - obs.my_bet
        opp_call = obs.raise_to[action] - obs.opp_bet
        all_in = added >= obs.my_stack
        size = int(opp_call >= 0.75 * (obs.pot + to_call))
        fold = _fold_rate(memory, group, size)
        cut = int(np.searchsorted(-tail_weight, -(1.0 - fold) * total_weight))
        cut = min(cut, order.size - 1)
        continue_equity = float(tail_points[cut] / tail_weight[cut])
        if not all_in:
            continue_equity = max(continue_equity - RERAISE_PENALTY, 0.0)
        share = 1.0 if all_in else realize
        showdown_value = share * continue_equity * (obs.pot + added + opp_call) - added
        values[action] = fold * obs.pot + (1.0 - fold) * showdown_value - BET_MARGIN

    return max(values, key=values.__getitem__)


def _safe_action(obs: Observation) -> Action:
    """A legal answer that needs no thinking (the emergency exit)."""
    if obs.to_call == 0 or obs.to_call * 4 <= obs.pot or Action.FOLD not in obs.legal_actions:
        return Action.CHECK_CALL
    return Action.FOLD


def new_half(config: AgentConfig, memory: dict) -> None:
    """Called at the start of each half-match (1000 hands)."""
    memory["num_hands"] = config.num_hands
    memory["time_used"] = 0.0   # seconds spent inside choose_action
    memory["calls"] = 0         # requests answered (each also costs CALL_OVERHEAD)
    memory["hand"] = None
    # facing[group][size] = [folds, calls, raises] when facing OUR raise.
    # group: 0 preflop, 1 postflop. size: 0 smaller, 1 about pot or more.
    memory["facing"] = [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]
    # open[group] = [raises, chances] when the opponent wasn't facing a raise.
    memory["open"] = [[0, 0], [0, 0]]
    # Showdown learning: log-likelihood of each BLUFF_GRID value, streets checked.
    memory["bluff_ll"] = np.zeros(BLUFF_GRID.size)
    memory["bluff_n"] = 0
    memory["bluff_mix"] = BLUFF_MIX


def choose_action(obs: Observation, memory: dict) -> Action:
    """Called every time it is your turn. Return one of obs.legal_actions."""
    started = time.perf_counter()
    try:
        if memory["hand"] is None or memory["hand"]["index"] != obs.hand_index:
            memory["hand"] = _new_hand_state(obs)
        _replay(obs, memory)
        action = _best_action(obs, memory)
    except Exception:
        # A bad decision costs chips; five crashes cost the whole match.
        action = _safe_action(obs)
    if action not in obs.legal_actions:
        action = _safe_action(obs)
    if memory["hand"] is not None:
        # Note this now: if the opponent folds, no later decision will see it.
        _note_my_action(obs, action, memory["hand"])
    memory["time_used"] += time.perf_counter() - started
    memory["calls"] = memory.get("calls", 0) + 1
    return action


def _note_my_action(obs: Observation, action: Action, hand: dict) -> None:
    group = min(int(obs.street), 1)
    if action in RAISE_ACTIONS:
        size = int(obs.raise_to[action] - obs.opp_bet >= 0.75 * (obs.pot + obs.to_call))
        hand["my_last"] = ("raise", size, group, obs.street)
    else:
        hand["my_last"] = ("check" if obs.to_call == 0 else "call", 0, group, obs.street)


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Called once for every finished hand: settle what the opponent did last,
    and learn from their cards if there was a showdown."""
    started = time.perf_counter()
    try:
        hand = memory["hand"]
        if hand is None or hand["index"] != result.hand_index:
            # We never acted, so the opponent folded their small blind.
            memory["open"][0][1] += 1
            return
        if hand["my_last"] is not None:
            kind, size, group, street = hand["my_last"]
            if kind == "raise":
                memory["facing"][group][size][1 if result.went_to_showdown else 0] += 1
                if result.went_to_showdown:  # they called our last raise
                    hand["keep"][int(street)] *= max(1.0 - _fold_rate(memory, group, size), 0.05)
            elif kind == "check" and result.went_to_showdown and not hand["button"]:
                memory["open"][1][1] += 1  # we checked the river and they checked back
        # Learning is worth less than deciding well: skip it when time is tight.
        spent = memory.get("time_used", 0.0) + memory.get("calls", 0) * CALL_OVERHEAD
        projected = spent / max(result.hand_index, 20) * memory.get("num_hands", 1000)
        if result.went_to_showdown and result.opp_hole and projected < TIME_TARGET:
            _learn_from_showdown(result, hand, memory)
    except Exception:
        pass  # a crash here would count as an error; one lost stat doesn't matter
    finally:
        if "time_used" in memory:
            memory["time_used"] += time.perf_counter() - started


# ----- Tuned settings ----------------------------------------------------------
# tools/self_improve.py plays matches offline and writes the knob values that
# won to tuned_params.json. Only these names are read, and any problem with the
# file just leaves the defaults above in place.
TUNED_PARAMS_FILE = "tuned_params.json"
TUNABLE = ("BLUFF_MIX", "RANGE_SOFTNESS", "PRIOR_STREET_DECAY", "RERAISE_PENALTY",
           "BET_MARGIN", "PRIOR_WEIGHT", "SHOWDOWN_PRIOR")


def _apply_tuned_params() -> None:
    try:
        import json
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), TUNED_PARAMS_FILE)
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as handle:
            tuned = json.load(handle)
        for name in TUNABLE:
            value = tuned.get(name)
            if isinstance(value, (int, float)) and np.isfinite(value) and value >= 0:
                globals()[name] = float(value)
    except Exception:
        pass


_apply_tuned_params()

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
