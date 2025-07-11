from env_twop import InBetweenEnv2P
from env_onep import InBetweenEnv
from core import greedy, Deck, cautious, Player, print_q_table
import numpy as np, math, random, matplotlib.pyplot as plt
from collections import defaultdict

def train_q_agent(env,
                  episodes: int = 400_000,
                  alpha: float = 0.1,
                  eps_start: float = 0.5,
                  eps_end: float = 0.01,
                  eps_decay: float = 2e6,
                  print_every: int = 100_000):
    def key(obs):
        return obs[:2]
    
    Q = defaultdict(lambda: np.zeros(env.action_space.n, dtype=float))

    for ep in range(episodes):
        obs, _ = env.reset()
        state  = key(obs)

        if hasattr(env, "pending_reward") and env.pending_reward != 0:
            Q[state][0] += alpha * (env.pending_reward - Q[state][0])
            
        done = False

        while not done:
            agent_turn = not hasattr(env, "phase") or env.phase == "agent"

            eps = eps_end + (eps_start - eps_end) * math.exp(-ep / eps_decay)
            action = (
                env.action_space.sample() if agent_turn and random.random() < eps
                else int(np.argmax(Q[state])) if agent_turn
                else 0                       # dummy during opp turn
            )

            next_obs, reward, done, _, _ = env.step(action)
            next_state = key(next_obs)

            if agent_turn:
                Q[state][action] += alpha * (reward - Q[state][action])

            state = next_state

        if (ep + 1) % print_every == 0:
            print(f"[{ep+1:_}/{episodes:_}]  "
                  f"ε = {eps:.3f}")

    return Q

def make_q_policy(Q):
    def policy(low, high, pot, bal):
        gap_bucket = min(max(high - low - 1, 0), 11)
        pot_bucket = min(int(math.log2(max(pot, 1))), 7)
        state      = (gap_bucket, pot_bucket)
        action     = int(np.argmax(Q[state]))
        return int(action/20 * pot)
    return policy

# -------------------- main -----------------------------------
if __name__ == "__main__":
    # env = InBetweenEnv()          # 1-player training
    env = InBetweenEnv2P()      # 2-player training
    
    print("Training …")
    Q = train_q_agent(env, episodes=100+_000_000)

    from core import print_q_table
    print_q_table(Q, n_gap_buckets=12, n_pot_buckets=8)


    rl_policy = make_q_policy(Q)


    from core import Player, Deck, greedy, ante_up, turn

    LOG_EVERY = 1_000
    N_HANDS   = 2_000_000
    deck      = Deck()

    players = [
        Player("RL-Bot", rl_policy),
        Player("Greedy", greedy),
    ]

    pot, ante = 0, 1
    n_samples = (N_HANDS // LOG_EVERY) + 1
    for p in players:
        p.history = [0] * n_samples
    sample_idx = 0

    for hand in range(N_HANDS):
        pot = ante_up(players, pot, ante)
        for pl in players:
            pot = turn(pl, deck, pot)
        if len(deck.cards) < 15:
            deck.shuffle()

        if hand % LOG_EVERY == 0:
            for pl in players:
                pl.history[sample_idx] = pl.balance
            sample_idx += 1

    for pl in players:
        pl.history[sample_idx] = pl.balance

    import matplotlib.pyplot as plt
    x = range(0, N_HANDS + 1, LOG_EVERY)

    plt.figure(figsize=(9, 4))
    for p in players:
        plt.plot(x, p.history, label=f"{p.name}  (final {p.balance:+})")
    plt.xlabel("Hands played")
    plt.ylabel("Bankroll")
    plt.title("RL-Bot vs. Greedy (sampled every 1 000 hands)")
    plt.legend()
    plt.tight_layout()
    plt.show()
