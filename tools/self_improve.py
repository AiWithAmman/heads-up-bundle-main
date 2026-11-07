"""
tools/self_improve.py - let the bot tune itself offline, between your runs.

    python tools/self_improve.py                    # 20 rounds, ~1 min each
    python tools/self_improve.py --rounds 100 --deals 400
    python tools/self_improve.py --dry-run          # never writes tuned_params.json

Why offline: in the tournament every half starts a fresh process with wiped
memory, so nothing learned there survives into the next match. Learning that
lasts has to happen here and be committed (tuned_params.json, read by agent.py
at start-up).

Each round nudges one knob up or down and plays the candidate and the current
best ("incumbent") against the same sparring bots on the same fresh cards, plus
a duplicate match against each other. The change is kept only if it wins both
ways: better against the pool AND not worse head-to-head. Every round is logged
to tools/self_improve_log.csv.

Results against sparring bots are only a proxy for real opponents: watch out
for knobs drifting to extremes, and sanity-check with judge_engine.py.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
sys.path.insert(0, ROOT)

import arena  # noqa: E402  (also loads agent.py once)

# name: (lowest, highest, step)
SPACE = {
    "BLUFF_MIX": (0.02, 0.30, 0.02),
    "RANGE_SOFTNESS": (0.03, 0.20, 0.02),
    "PRIOR_STREET_DECAY": (0.0, 0.8, 0.1),
    "RERAISE_PENALTY": (0.0, 0.08, 0.01),
    "BET_MARGIN": (0.0, 0.5, 0.05),
    "PRIOR_WEIGHT": (2.0, 15.0, 1.5),
    "SHOWDOWN_PRIOR": (5.0, 40.0, 5.0),
}
POOL = ["random", "station", "maniac", "nit", "mc_loose", "mc_tight", "mc_aggro", "bully", "trapper"]
PARAMS_PATH = os.path.join(ROOT, "tuned_params.json")
LOG_PATH = os.path.join(TOOLS, "self_improve_log.csv")


def fresh_agent(params: dict, tag: str):
    module = arena.load("agent.py", f"tuned_{tag}")
    for name, value in params.items():
        setattr(module, name, value)
    return module


def pool_score(module, seed: str, deals: int, opponents: list) -> float:
    """Average bb/deal of this agent over the sparring pool."""
    total = 0.0
    for name in opponents:
        per_deal, _ = arena.match(arena.Module(module), arena.opponents()[name], seed, deals)
        total += sum(per_deal) / len(per_deal)
    return total / len(opponents)


def head_to_head(cand, inc, seed: str, deals: int) -> tuple[float, float]:
    """Candidate vs incumbent, duplicate: (bb/deal, 95% half-width)."""
    per_deal, _ = arena.match(arena.Module(cand), arena.Module(inc), seed, deals)
    mean = sum(per_deal) / len(per_deal)
    sd = math.sqrt(sum((x - mean) ** 2 for x in per_deal) / (len(per_deal) - 1))
    return mean, 1.96 * sd / math.sqrt(len(per_deal))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--deals", type=int, default=300)
    parser.add_argument("--opponents", type=int, default=4, help="sparring bots sampled per round")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = arena.AGENT
    best = {name: float(getattr(base, name)) for name in SPACE}
    print("starting from:", json.dumps(best))
    incumbent = fresh_agent(best, "inc")
    candidate = fresh_agent(best, "cand")
    rng = random.Random(time.time_ns())
    new_log = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as log_file:
        log = csv.writer(log_file)
        if new_log:
            log.writerow(["time", "param", "old", "new", "pool_delta", "h2h", "h2h_ci", "kept"])
        for round_no in range(1, args.rounds + 1):
            name = rng.choice(list(SPACE))
            low, high, step = SPACE[name]
            new = round(min(high, max(low, best[name] + rng.choice((-step, step)))), 4)
            if new == best[name]:
                continue
            trial = dict(best, **{name: new})
            for key, value in trial.items():
                setattr(candidate, key, value)
            seed = f"tune{rng.randrange(10**9)}"
            opponents = rng.sample(POOL, args.opponents)
            started = time.perf_counter()
            delta = (pool_score(candidate, seed, args.deals, opponents)
                     - pool_score(incumbent, seed, args.deals, opponents))
            h2h, ci = head_to_head(candidate, incumbent, seed, args.deals)
            kept = delta > 0.02 and h2h > -0.25 * ci and delta + h2h > 0.05
            old = best[name]
            if kept:
                best = trial
                for key, value in best.items():
                    setattr(incumbent, key, value)
                if not args.dry_run:
                    with open(PARAMS_PATH, "w", encoding="utf-8") as handle:
                        json.dump(best, handle, indent=2)
            log.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), name, old, new, f"{delta:+.3f}",
                          f"{h2h:+.3f}", f"{ci:.3f}", kept])
            log_file.flush()
            print(f"[{round_no:>3}] {name:<18} -> {new:<7} pool {delta:+.3f}  h2h {h2h:+.3f} +-{ci:.3f}"
                  f"  {'KEPT' if kept else 'rejected'}  ({time.perf_counter() - started:.0f}s)", flush=True)
    print("best:", json.dumps(best))
    if args.dry_run:
        print("(dry run: tuned_params.json not written)")


if __name__ == "__main__":
    main()
