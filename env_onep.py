import math
import gymnasium as gym
from gymnasium import spaces
from core import Deck

class InBetweenEnv(gym.Env):
    metadata = {"render_modes":[]}
    def __init__(self,ante: int = 1, ante_players: int = 4):
        super().__init__()
        self.ante = ante
        self.ante_players = ante_players
        self.deck = Deck()
        self.pot = 0
        
        self.action_space = spaces.Discrete(21)                 # we let 0=pass 1 be 5% and so on to 20 being FULL POT
        self.observation_space = spaces.MultiDiscrete([12,8])    # gap_bucket and pot_bucket
   
    def _gap_bucket(self):
        return min(max(self.high - self.low - 1, 0), 11)
    
    def _pot_bucket(self):
        return min(int(math.log2(max(self.pot, 1))), 7)
    
    def reset(self, seed=None, options=None,):
        super().reset(seed=seed)
        
        if self.pot == 0:
            self.pot += self.ante * self.ante_players

        if len(self.deck.cards) < 3:           
            self.deck.shuffle()
            
        c1, c2 = self.deck.draw(), self.deck.draw()
        self.low, self.high = sorted((c1,c2))

        
        obs = (self._gap_bucket(), self._pot_bucket())
        info = {"low": self.low, "high": self.high}
        return obs, info
    
    def step(self, action):
        if self.low == self.high:           # pair → forced pass
            bet = 0
        else:
             bet = int((action / 20.0) * self.pot)

        reward = 0
        if bet > 0:
            target = self.deck.draw()
            if self.low < target < self.high:        # win
                self.pot  -= bet
                reward = +bet
            elif target in (self.low, self.high):    # post
                self.pot  += 2 * bet
                reward = -2 * bet
            else:                                    # outside
                self.pot  += bet
                reward = -bet

        terminated = True    # hand finished
        obs = (0, 0)         # dummy; not used after done
        info = {}
        return obs, reward, terminated, False, info