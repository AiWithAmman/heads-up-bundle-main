"""
Tests for YOUR bot (agent.py). Run them from the bundle folder:

    python -m pytest -q

They talk to your bot the same way the judge does (through its web routes,
but without starting a real server), so a passing run means the judge can
play with your bot. Add your own tests at the bottom!
"""

import importlib.util
import os
import random
import sys

import pytest
from fastapi.testclient import TestClient

# Works whether this file sits next to agent.py or inside a tests/ folder.
HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = HERE if os.path.isfile(os.path.join(HERE, "game.py")) else os.path.dirname(HERE)
sys.path.insert(0, BUNDLE)  # so agent.py can import game.py and evaluator.py

from game import Action, AgentConfig, Game, make_deal  # noqa: E402
from judge_engine import coerce_action  # noqa: E402  (reads answers exactly like the judge)

TOKEN = "test-token"
AUTH = {"X-Agent-Token": TOKEN}


def load_agent():
    """Load a fresh copy of agent.py (its own MEMORY and globals)."""
    spec = importlib.util.spec_from_file_location("agent_under_test", os.path.join(BUNDLE, "agent.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bot():
    """(agent module, test client) with the tournament's token check switched on."""
    module = load_agent()
    module.TOKEN = TOKEN
    return module, TestClient(module.app)


def start_half(client, half=1, seat=0):
    """POST /new-half the way the judge does."""
    body = {"half": half, "seat": seat, "config": AgentConfig().to_dict()}
    return client.post("/new-half", json=body, headers=AUTH)


def play_hands(client, num_hands, seed="test", raise_rate=0.3):
    """Play hands where YOUR bot is seat 0 and a random opponent is seat 1.

    Every decision goes through /act exactly like in a real match, and
    finished hands are delivered with the next /act (like the judge does).
    Returns the list of every action your bot chose.
    """
    rng = random.Random(seed)
    pending = []      # finished hands waiting to be delivered
    chosen = []
    for hand in range(num_hands):
        game = Game(make_deal(seed, hand), button_seat=hand % 2, hand_index=hand)
        while not game.is_terminal:
            seat = game.to_act
            obs = game.observation(seat)
            if seat == 0:
                r = client.post("/act", json={"observation": obs.to_dict(), "hand_results": pending},
                                headers=AUTH)
                pending = []
                assert r.status_code == 200, f"/act failed: {r.json()}"
                reply = r.json()
                assert reply.get("hand_end_errors", []) == [], \
                    f"on_hand_end crashed: {reply['hand_end_errors'][0]['traceback']}"
                action = coerce_action(reply["action"])
                assert action in obs.legal_actions, \
                    f"choose_action returned {reply['action']!r}; legal were {[a.name for a in obs.legal_actions]}"
                chosen.append(action)
            else:
                # Random opponent that raises sometimes, so your bot faces bets.
                raises = [a for a in obs.legal_actions if a in (Action.RAISE_HALF, Action.RAISE_POT)]
                action = rng.choice(raises) if raises and rng.random() < raise_rate else Action.CHECK_CALL
            game.apply(action)
        pending.append(game.hand_result(0).to_dict())
    r = client.post("/hand-end", json={"results": pending}, headers=AUTH)  # end of half
    assert r.status_code == 200 and r.json()["hand_end_errors"] == []
    return chosen


# ---------------------------------------------------------------------------
# Your bot plays correctly
# ---------------------------------------------------------------------------

def test_names_are_set(bot):
    _, client = bot
    info = client.get("/").json()
    assert isinstance(info["participant"], str) and info["participant"]
    assert isinstance(info["agent_name"], str) and info["agent_name"]


def test_new_half_works(bot):
    _, client = bot
    r = start_half(client)
    assert r.status_code == 200, f"new_half crashed: {r.json()}"


def test_every_decision_is_legal(bot):
    # 200 hands with a random opponent: every choose_action answer must be legal.
    _, client = bot
    start_half(client)
    play_hands(client, 200)


def test_works_in_both_seats_and_both_halves(bot):
    _, client = bot
    for half, seat in ((1, 0), (2, 1)):
        assert start_half(client, half, seat).status_code == 200
        play_hands(client, 30, seed=f"half{half}")


def test_handles_aggressive_opponent(bot):
    # An opponent that raises whenever it can: your bot must cope with
    # big bets and all-ins.
    _, client = bot
    start_half(client)
    play_hands(client, 100, seed="aggressive", raise_rate=1.0)


def test_memory_is_fresh_each_half(bot):
    module, client = bot
    start_half(client)
    play_hands(client, 10)
    start_half(client, half=2, seat=1)
    after_reset = dict(module.MEMORY)
    fresh = load_agent()
    fresh.new_half(AgentConfig(), fresh.MEMORY)
    assert after_reset.keys() == fresh.MEMORY.keys(), "memory should only hold what new_half() puts there"


# ---------------------------------------------------------------------------
# The server plumbing still works (only fails if you edited it)
# ---------------------------------------------------------------------------

def test_token_is_checked(bot):
    _, client = bot
    body = {"half": 1, "seat": 0, "config": {}}
    assert client.post("/new-half", json=body).status_code == 401
    assert client.post("/new-half", json=body, headers={"X-Agent-Token": "wrong"}).status_code == 401


def test_bad_token_is_refused_before_the_body_is_read(bot):
    # Without the right token the bot must answer 401 without parsing the body,
    # so junk sent by anyone else can't slow your bot down.
    _, client = bot
    junk = b"this is not JSON" * 1000
    assert client.post("/act", content=junk).status_code == 401
    assert client.post("/act", content=junk, headers={"X-Agent-Token": "wrong"}).status_code == 401


def test_malformed_body_is_rejected(bot):
    _, client = bot
    start_half(client)
    assert client.post("/act", content=b"not json", headers=AUTH).status_code == 400
    assert client.post("/act", json=[1, 2, 3], headers=AUTH).status_code == 400


def test_act_before_new_half_is_refused(bot):
    _, client = bot
    obs = Game(make_deal(1, 0), 0).observation(0).to_dict()
    assert client.post("/act", json={"observation": obs, "hand_results": []}, headers=AUTH).status_code == 409


def test_crashes_are_reported_to_the_judge(bot):
    # If choose_action raises, the judge must get a 500 with your traceback.
    module, client = bot

    def broken(obs, memory):
        raise ValueError("oops")

    module.choose_action = broken
    start_half(client)
    obs = Game(make_deal(1, 0), 0).observation(0).to_dict()
    r = client.post("/act", json={"observation": obs, "hand_results": []}, headers=AUTH)
    assert r.status_code == 500
    assert "ValueError: oops" in r.json()["error"] and "Traceback" in r.json()["traceback"]


def test_reads_port_from_environment():
    source = open(os.path.join(BUNDLE, "agent.py"), encoding="utf-8").read()
    assert 'os.environ.get("PORT"' in source, "agent.py must read its port from $PORT"
    assert '0.0.0.0' in source, "agent.py must listen on 0.0.0.0 so the judge can reach it"


# ---------------------------------------------------------------------------
# Add your own tests below, e.g. "with two aces preflop my bot raises".
# ---------------------------------------------------------------------------

def observation_for(deal, button, actions=()):
    """The observation the player to act gets after `actions`."""
    game = Game(deal, button)
    for action in actions:
        game.apply(action)
    return game.observation(game.to_act), game


def test_score_table_matches_the_official_evaluator():
    # Every decision relies on this table, so it must agree with omaha_score.
    from evaluator import omaha_score
    module = load_agent()
    rng = random.Random(5)
    np = module.np
    for _ in range(3000):
        cards = rng.sample(range(36), 10)
        hole, board = cards[:5], cards[5:]
        pair_scores = module._pair_scores(np.array([board]))
        fast = pair_scores[:, module._pair_ids(np.array(hole))].max()
        assert fast == omaha_score(hole, board)


def test_opponent_folds_to_our_raise_are_learned():
    # A fold ends the hand, so no later decision can see it: on_hand_end has to.
    module = load_agent()
    memory = {}
    module.new_half(AgentConfig(), memory)
    obs, game = observation_for(make_deal("fold", 0), 0)
    module._best_action = lambda o, m: Action.RAISE_POT
    module.choose_action(obs, memory)
    game.apply(Action.RAISE_POT)
    game.apply(Action.FOLD)
    module.on_hand_end(game.hand_result(0), memory)
    assert sum(memory["facing"][0][1]) == 1 and memory["facing"][0][1][0] == 1


def test_calls_after_our_raise_are_learned_at_showdown():
    module = load_agent()
    memory = {}
    module.new_half(AgentConfig(), memory)
    obs, game = observation_for(make_deal("call", 0), 0)
    module._best_action = lambda o, m: Action.RAISE_POT
    while not game.is_terminal:  # we shove every street, the opponent calls
        if game.to_act == 0:
            obs = game.observation(0)
            action = module.choose_action(obs, memory)
            game.apply(action)
        else:
            game.apply(Action.CHECK_CALL)
    module.on_hand_end(game.hand_result(0), memory)
    assert memory["facing"][1][1][0] == 0, "a called raise must not count as a fold"
    assert memory["facing"][1][1][1] >= 1 or memory["facing"][0][1][1] >= 1


def test_crash_inside_the_strategy_still_gives_a_legal_action():
    module = load_agent()
    memory = {}
    module.new_half(AgentConfig(), memory)

    def broken(obs, memory):
        raise RuntimeError("bug in the strategy")

    module._best_action = broken
    obs, game = observation_for(make_deal("crash", 0), 0)
    assert module.choose_action(obs, memory) in obs.legal_actions
    # And with a bet to face, too.
    obs, game = observation_for(make_deal("crash", 0), 0, [Action.RAISE_POT])
    assert module.choose_action(obs, memory) in obs.legal_actions


def test_on_hand_end_survives_bad_memory():
    module = load_agent()
    memory = {}  # new_half never ran: every lookup would raise
    result = Game(make_deal("x", 0), 0)
    result.apply(Action.FOLD)
    module.on_hand_end(result.hand_result(0), memory)


def test_never_folds_when_checking_is_free():
    module = load_agent()
    memory = {}
    module.new_half(AgentConfig(), memory)
    for hand in range(100):
        game = Game(make_deal("free", hand), hand % 2, hand_index=hand)
        while not game.is_terminal:
            obs = game.observation(game.to_act)
            action = module.choose_action(obs, memory)
            if obs.to_call == 0:
                assert action != Action.FOLD
            game.apply(action if game.to_act == 0 else Action.CHECK_CALL)


def test_uses_a_nut_hand_on_the_river():
    # Royal flush on the river: the bot must not fold or just give up the hand.
    from evaluator import cards_from_str
    from game import Deal
    module = load_agent()
    memory = {}
    module.new_half(AgentConfig(), memory)
    hole = tuple(sorted(cards_from_str("Ah Kh 6c 7d 8s")))
    opp = tuple(sorted(cards_from_str("6d 7c 8d 9c Tc")))
    deal = Deal((hole, opp), tuple(cards_from_str("Qh Jh Th 6s 7s")))
    game = Game(deal, 0)
    game.apply(Action.CHECK_CALL)
    game.apply(Action.CHECK_CALL)
    while game.street != 3:
        game.apply(Action.CHECK_CALL)
    obs = game.observation(game.to_act)
    if game.to_act == 0:
        assert module.choose_action(obs, memory) in (Action.RAISE_HALF, Action.RAISE_POT)
    else:
        game.apply(Action.RAISE_HALF)
        obs = game.observation(0)
        assert module.choose_action(obs, memory) in (Action.RAISE_HALF, Action.RAISE_POT)


def test_time_pacing_shrinks_samples_when_behind():
    module = load_agent()
    memory = {}
    module.new_half(AgentConfig(), memory)
    obs, _ = observation_for(make_deal("t", 0), 0)
    assert module._time_scale(obs, memory) == 1.0
    memory["time_used"] = module.TIME_PANIC + 0.1
    assert module._time_scale(obs, memory) < 0.2
