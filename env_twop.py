import math, random, numpy as np
import gymnasium as gym
from gymnasium import spaces
from core import Deck, greedy            # reuse existing code

class InBetweenEnv2P(gym.Env):
    """
    One episode = BOTH players take a bet on the same hand.
    Phase 'agent'  : RL agent acts.
    Phase 'opp'    : fixed opponent acts (greedy policy).
    Then draw third card, resolve both bets.
    Observation = (gap_bucket, pot_bucket, turn_flag)
        turn_flag = 0 if agent's turn, 1 if opponent's turn
    Action space  = 0‥20 (same 5 % buckets, only used on agent turn).
    Reward        = agent's chip delta (opp reward ignored).
    """
    metadata = {"render_modes": []}

    def __init__(self, ante: int = 1, ante_players: int = 2):
        super().__init__()
        self.ante = ante
        self.ante_players = ante_players    # both will ante
        self.deck = Deck()
        self.pot  = 0

        # 21 discrete bet choices
        self.action_space = spaces.Discrete(21)

        # gap 0-11, pot bucket 0-7, turn flag 0/1
        self.observation_space = spaces.MultiDiscrete([12, 8, 2])

        self.phase = "agent"   # will be set in reset()

    # helper buckets (same as your log version)
    def _gap_bucket(self): return min(self.high - self.low - 1, 11)
    def _pot_bucket(self): return min(int(math.log2(max(self.pot, 1))), 7)

    # --------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pot = self.ante * self.ante_players   # both ante
        if len(self.deck.cards) < 15:              # quick reshuffle
            self.deck.shuffle()

        c1, c2 = self.deck.draw(), self.deck.draw()
        self.low, self.high = sorted((c1, c2))
        self.phase = "agent"

        obs  = (self._gap_bucket(), self._pot_bucket(), 0)
        info = {}
        return obs, info

    # --------------------------------------------------
    def step(self, action):
        if self.phase == "agent":
            # ----- agent turn ---------------------------------
            self.agent_bet = int((action / 20.0) * self.pot)
            # opponent phase next
            self.phase = "opp"
            obs = (self._gap_bucket(), self._pot_bucket(), 1)
            return obs, 0.0, False, False, {}        # no reward yet

        else:
            # ----- opponent turn ------------------------------
            opp_action = self._opp_action()
            self.opp_bet = int((opp_action / 20.0) * self.pot)

            # ----- draw & resolve -----------------------------
            target = self.deck.draw()
            agent_reward = self._settle(self.agent_bet, target)
            _            = self._settle(self.opp_bet,  target)  # ignored

            terminated = True
            obs = (0, 0, 0)          # dummy
            return obs, agent_reward, terminated, False, {}

    # greedy opponent uses your existing function
    def _opp_action(self):
        a = greedy(self.low, self.high, self.pot, bal=None)
        return int(round(a / self.pot * 20)) if self.pot else 0

    # settle a single bet, update pot, return reward for that bettor
    def _settle(self, bet, target):
        if bet == 0:
            return 0
        if self.low < target < self.high:        # win
            self.pot -= bet
            return +bet
        elif target in (self.low, self.high):    # post
            self.pot += 2 * bet
            return -2 * bet
        else:                                    # outside loss
            self.pot += bet
            return -bet
