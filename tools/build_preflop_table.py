"""
tools/build_preflop_table.py - precompute preflop equities for agent.py.

    python tools/build_preflop_table.py        # writes preflop_table.npz (a few minutes)

Preflop, our equity depends only on our five hole cards, and suits only matter
up to renaming them, so there are just 19,998 distinct hands. For each one we
store the equity against every opponent-strength bucket (equal-sized slices of
all opponent hands, weakest first). agent.py then answers preflop with a table
lookup instead of a simulation, and the table can afford far more samples than
a live decision ever could (900 boards x 1350 hands, vs 300 x 600 live).

Rerun it if you change how strength or equity is measured in agent.py.
"""
import os
import sys
import time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import agent as A  # noqa: E402

N_BOARDS = 900
N_HANDS = 1350
N_BUCKETS = A.PREFLOP_BUCKETS
SEED = 7


def _pool():
    """The fixed boards and opponent hands (same in every worker: seeded)."""
    rng = np.random.default_rng(SEED)
    every_card = np.arange(A.N_CARDS)
    boards = every_card[rng.random((N_BOARDS, 36)).argpartition(4, axis=1)[:, :5]]
    hands = every_card[rng.random((N_HANDS, 36)).argpartition(4, axis=1)[:, :5]]
    pair_scores = A._pair_scores(boards)                       # (B, 630)
    scores = pair_scores[:, A._pair_ids(hands)].max(axis=2)    # (B, H)
    board_masks = A._BIT[boards].sum(axis=1)
    hand_masks = A._BIT[hands].sum(axis=1)
    valid = (board_masks[:, None] & hand_masks[None, :]) == 0
    strength = A._strength(scores, valid, valid.sum(axis=0))   # (H,)
    order = np.argsort(strength, kind="stable")
    bucket_of = np.empty(N_HANDS, dtype=np.int64)
    bucket_of[order] = np.arange(N_HANDS) * N_BUCKETS // N_HANDS
    # Doubled scores, and a negative marker where a hand clashes with the board.
    marked = np.where(valid, scores * 2, -1).astype(np.int32)
    return pair_scores, marked, board_masks, hand_masks, strength, bucket_of


_DATA = None


def _work(reps: np.ndarray) -> np.ndarray:
    global _DATA
    if _DATA is None:
        _DATA = _pool()
    pair_scores, marked, board_masks, hand_masks, _, bucket_of = _DATA
    out = np.empty((len(reps), N_BUCKETS), dtype=np.float32)
    for n, hole in enumerate(reps):
        my_mask = A._BIT[hole].sum()
        b_idx = np.flatnonzero((board_masks & my_mask) == 0)
        h_idx = np.flatnonzero((hand_masks & my_mask) == 0)
        mine = (pair_scores[b_idx][:, A._pair_ids(hole)].max(axis=1) * 2)[:, None]
        opp = marked[np.ix_(b_idx, h_idx)]
        ok = opp >= 0
        points = ((np.sign(mine - opp) + 1) * ok).sum(axis=0)    # 2 win, 1 tie, 0 loss
        per_hand = points / np.maximum(2 * ok.sum(axis=0), 1)
        buckets = bucket_of[h_idx]
        out[n] = np.bincount(buckets, per_hand, N_BUCKETS) / np.maximum(np.bincount(buckets, None, N_BUCKETS), 1)
    return out


def main() -> None:
    started = time.perf_counter()
    keys, reps = A._all_canonical_hands()
    chunks = np.array_split(reps, 40)
    workers = max(1, os.cpu_count() or 1)
    results = []
    with Pool(workers) as pool:
        for n, part in enumerate(pool.imap(_work, chunks), start=1):
            results.append(part)
            print(f"{n:>3}/{len(chunks)} chunks  {time.perf_counter() - started:.0f}s", flush=True)
    equity = np.concatenate(results)
    *_, strength, bucket_of = _pool()
    bucket_strength = np.bincount(bucket_of, strength, N_BUCKETS) / np.bincount(bucket_of, None, N_BUCKETS)
    path = os.path.join(ROOT, A.PREFLOP_TABLE_FILE)
    np.savez_compressed(path, keys=keys, equity=np.round(equity * 65535).astype(np.uint16),
                        strength=bucket_strength.astype(np.float32))
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB) in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
