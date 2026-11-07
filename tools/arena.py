"""
tools/arena.py - fast in-process duplicate matches, no web server or judge.

    python tools/arena.py                     # agent.py vs every sparring bot
    python tools/arena.py --deals 400 --seeds 2 --only mc_tight,maniac
    python tools/arena.py --set "BLUFF_MIX=0.2;PRIOR_FOLD=(0.3, 0.4)"
    python tools/arena.py --against old_agent.py,other_agent.py

Scores the same way as the judge (bb per deal: the two hands of a deal added
together, with a 95% CI).
It skips HTTP, so its timing is thinking time only. Use judge_engine.py for
the real thing before submitting.
"""
import argparse
import ast
import importlib.util
import math
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from evaluator import FLUSH, THREE_OF_A_KIND, category_of, omaha_score  # noqa: E402
from game import BIG_BLIND, Action, AgentConfig, Game, Street, make_deal  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AGENT = load("agent.py", "arena_agent")


class Bot:
    """Sparring partner interface: the same calls as agent.py."""

    def new_half(self, config):
        self.rng = random.Random(1234)

    def on_hand_end(self, result):
        pass


class Module(Bot):
    """Wraps a module with the agent.py functions."""

    def __init__(self, module):
        self.module = module

    def new_half(self, config):
        self.memory = {}
        self.module.new_half(config, self.memory)

    def act(self, obs):
        return self.module.choose_action(obs, self.memory)

    def on_hand_end(self, result):
        self.module.on_hand_end(result, self.memory)


class Fixed(Bot):
    """Plays the first legal action from a preference list."""

    def __init__(self, *preference):
        self.preference = preference

    def act(self, obs):
        return next(a for a in self.preference if a in obs.legal_actions)


class Random(Bot):
    def act(self, obs):
        return self.rng.choice(obs.legal_actions)


class Maniac(Bot):
    """Raises most of the time, never folds."""

    def act(self, obs):
        raises = [a for a in obs.legal_actions if a in (Action.RAISE_HALF, Action.RAISE_POT)]
        if raises and self.rng.random() < 0.7:
            return self.rng.choice(raises)
        return Action.CHECK_CALL


class Nit(Bot):
    """Folds to any real bet without a strong made hand on the river."""

    def act(self, obs):
        river = obs.street == Street.RIVER
        category = category_of(omaha_score(obs.hole_cards, obs.board)) if river else -1
        if obs.to_call == 0:
            if category >= FLUSH and Action.RAISE_POT in obs.legal_actions:
                return Action.RAISE_POT
            return Action.CHECK_CALL
        if category >= THREE_OF_A_KIND or (not river and obs.to_call <= obs.big_blind):
            return Action.CHECK_CALL
        return Action.FOLD


class MonteCarlo(Bot):
    """The bot the guide suggests: equity against a random hand, then pot odds."""

    def __init__(self, raise_at, pot_at, margin=0.0):
        self.raise_at, self.pot_at, self.margin = raise_at, pot_at, margin

    def act(self, obs):
        hole = np.array(obs.hole_cards, dtype=np.int64)
        if obs.street == Street.PREFLOP:
            points, _ = AGENT._preflop_analysis(hole)
        else:
            board = np.array(obs.board, dtype=np.int64)
            points, _ = AGENT._postflop_analysis(hole, board, obs.street, 0.5)
        equity = float(points.mean())
        if equity >= self.pot_at and Action.RAISE_POT in obs.legal_actions:
            return Action.RAISE_POT
        if equity >= self.raise_at and Action.RAISE_HALF in obs.legal_actions:
            return Action.RAISE_HALF
        price = obs.to_call / (obs.pot + obs.to_call)
        if obs.to_call == 0 or equity >= price + self.margin:
            return Action.CHECK_CALL
        return Action.FOLD


class Bully(Bot):
    """Bets pot whenever checked to, gives up to any raise without a real hand.
    It bluffs constantly, which punishes a bot that believes every bet."""

    def act(self, obs):
        if obs.to_call == 0 and Action.RAISE_POT in obs.legal_actions:
            return Action.RAISE_POT
        if obs.to_call <= obs.big_blind:
            return Action.CHECK_CALL
        hole = np.array(obs.hole_cards, dtype=np.int64)
        if obs.street == Street.PREFLOP:
            points, _ = AGENT._preflop_analysis(hole)
        else:
            points, _ = AGENT._postflop_analysis(hole, np.array(obs.board, dtype=np.int64), obs.street, 0.5)
        return Action.CHECK_CALL if points.mean() >= 0.65 else Action.FOLD


class Trapper(Bot):
    """Plays passively, then check-raises strong hands. This punishes a bot that
    reads checks and calls as weakness."""

    def act(self, obs):
        hole = np.array(obs.hole_cards, dtype=np.int64)
        if obs.street == Street.PREFLOP:
            return Action.CHECK_CALL
        points, _ = AGENT._postflop_analysis(hole, np.array(obs.board, dtype=np.int64), obs.street, 0.5)
        equity = float(points.mean())
        if obs.to_call > 0 and equity >= 0.75 and Action.RAISE_POT in obs.legal_actions:
            return Action.RAISE_POT
        price = obs.to_call / (obs.pot + obs.to_call)
        return Action.CHECK_CALL if obs.to_call == 0 or equity >= price + 0.08 else Action.FOLD


class Bluffer(MonteCarlo):
    """A decent bot that also bluffs a lot: it bets its weakest hands when checked to."""

    def act(self, obs):
        action = super().act(obs)
        if (obs.to_call == 0 and action == Action.CHECK_CALL and obs.street != Street.PREFLOP
                and Action.RAISE_POT in obs.legal_actions and self.rng.random() < 0.45):
            return Action.RAISE_POT
        return action


def clone(**changes):
    """A separate copy of agent.py with some constants changed."""
    def build():
        module = load("agent.py", f"arena_clone_{len(changes)}_{id(changes)}")
        for name, value in changes.items():
            setattr(module, name, value)
        return Module(module)
    return build


def opponents():
    return {
        "random": Random(),
        "always_half": Fixed(Action.RAISE_HALF, Action.CHECK_CALL),
        "always_pot": Fixed(Action.RAISE_POT, Action.CHECK_CALL),
        "station": Fixed(Action.CHECK_CALL),
        "maniac": Maniac(),
        "nit": Nit(),
        "mc_loose": MonteCarlo(0.60, 0.70),
        "mc_tight": MonteCarlo(0.68, 0.78, 0.05),
        "mc_aggro": MonteCarlo(0.52, 0.62),
        "bully": Bully(),
        "trapper": Trapper(),
        "bluffer": Bluffer(0.62, 0.72),
        "clone_aggro": clone(BET_MARGIN=-2.0),
        "clone_passive": clone(BET_MARGIN=3.0),
    }


def play_half(seats, seed, deals, config):
    """seats[0] and seats[1] are bots. Returns seat 0's net chips per deal."""
    for bot in seats:
        bot.new_half(config)
    nets = []
    for deal in range(deals):
        game = Game(make_deal(seed, deal), deal % 2, hand_index=deal)
        while not game.is_terminal:
            seat = game.to_act
            game.apply(seats[seat].act(game.observation(seat)))
        nets.append(game.result().nets[0])
        for seat in (0, 1):
            seats[seat].on_hand_end(game.hand_result(seat))
    return nets


def match(hero, villain, seed, deals):
    """Both halves on the same cards. Returns (bb per deal list, s per 1000 hands)."""
    config = AgentConfig(num_hands=deals)
    first = play_half((hero, villain), seed, deals, config)
    first_used = hero.memory["time_used"]
    second = [-n for n in play_half((villain, hero), seed, deals, config)]
    used = max(first_used, hero.memory["time_used"]) * 1000 / deals
    return [(a + b) / BIG_BLIND for a, b in zip(first, second)], used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--first-seed", type=int, default=0, help="start here for unseen cards")
    parser.add_argument("--only", default="", help="comma-separated opponent names")
    parser.add_argument("--set", default="", help="override agent.py constants, separated by ;")
    parser.add_argument("--against", default="", help="comma-separated agent files to play too")
    args = parser.parse_args()
    for item in filter(None, args.set.split(";")):
        name, value = item.split("=")
        setattr(AGENT, name, ast.literal_eval(value))
    pool = opponents()
    for number, path in enumerate(filter(None, args.against.split(","))):
        name = os.path.splitext(os.path.basename(path))[0]
        pool[name] = Module(load(os.path.abspath(path), f"arena_other_{number}"))
    names = args.only.split(",") if args.only else list(pool)
    means = []
    print(f"{'opponent':<14}{'bb/deal':>9}{'+-95%':>8}  verdict {'s/1000':>8}")
    for name in names:
        results, worst = [], 0.0
        for s in range(args.first_seed, args.first_seed + args.seeds):
            villain = pool[name]() if callable(pool[name]) else pool[name]
            per_deal, used = match(Module(AGENT), villain, f"arena{s}", args.deals)
            results += per_deal
            worst = max(worst, used)
        mean = sum(results) / len(results)
        sd = math.sqrt(sum((x - mean) ** 2 for x in results) / (len(results) - 1))
        ci = 1.96 * sd / math.sqrt(len(results))
        verdict = "WIN" if mean - ci > 0 else "LOSE" if mean + ci < 0 else "chop"
        print(f"{name:<14}{mean:>+9.3f}{ci:>8.3f}  {verdict:<8}{worst:>7.2f}", flush=True)
        means.append(mean)
    print(f"{'AVERAGE':<14}{sum(means) / len(means):>+9.3f}")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"({time.perf_counter() - start:.0f}s)")
