import math
import gymnasium as gym
from gymnasium import spaces
from core import Deck, greedy

class InBetweenEnv(gym.Env):
    metadata = {"render_modes":[]}
    def __init__(self, ante: int = 1):
        super().__init__()
        self.ante = max(1, ante)
        self.deck = Deck()
        self.pot  = 0

        self.action_space      = spaces.Discrete(21)      # 0 … 20
        self.observation_space = spaces.MultiDiscrete([12, 8, 2])   # gap_bucket and pot_bucket
    
    def _deal_pair(self):
        c1, c2 = self.deck.draw(), self.deck.draw()
        return sorted((c1, c2))
    
    def _gap_bucket(self, low, high):
        return min(max(high - low - 1, 0), 11)

    def _pot_bucket(self):
        return min(int(math.log2(max(self.pot, 1))), 7)

    def _decode_bet(self, action):
        return int((action / 20.0) * self.pot)
    
    def _settle(self, bet, low, high):
        if bet == 0:
            return 0
        target = self.deck.draw()
        if low < target < high:          # win
            self.pot -= bet
            return +bet
        elif target in (low, high):      # post (double loss)
            self.pot += 2 * bet
            return -2 * bet
        else:                            # outside loss
            self.pot += bet
            return -bet
        
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.pot = self.ante * 2
        self.pending_reward = -self.ante      # ante cost for agent

        if len(self.deck.cards) < 20:
            self.deck.shuffle()

        self.low, self.high = self._deal_pair()
        self.phase = "agent"

        obs = (self._gap_bucket(self.low, self.high),
               self._pot_bucket(),
               0)                          # turn_flag = 0 (agent)
        info = {}
        return obs, info

    def step(self, action):
        if self.phase == "agent":

            bet     = self._decode_bet(action)
            reward  = self.pending_reward 
            reward  += self._settle(bet, self.low, self.high)
            self.pending_reward = 0        # clear ante cost

            self.low, self.high = self._deal_pair()
            self.phase = "opp"

            obs = (self._gap_bucket(self.low, self.high),
                   self._pot_bucket(),
                   1)                      # turn_flag = 1 (opp)
            return obs, reward, False, False, {}

        opp_bet  = greedy(self.low, self.high, self.pot, bal=None)
        self._settle(opp_bet, self.low, self.high)

        terminated = True
        obs  = (0, 0, 0)     # dummy after terminal
        info = {}
        return obs, 0.0, terminated, False, info