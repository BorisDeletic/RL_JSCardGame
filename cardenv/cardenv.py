import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pokerkit
'''
This game is inspired by a Jane street card game
The player asks for any subset of possible 52 cards from the dealer.
The dealer deals cards until one from the subset appears.
He gives this to the player and takes the rest for himself.
This repeats until the player has 5 cards.
If the dealer has less than 8 cards at this point, he deals himself cards until he has 8.
Whoever has the higher poker hand wins. Dealer wins ties.

We use this embedding for cards
https://github.com/atinm/poker-eval/blob/master/include/deck_std.h
'''

NUM_CARDS = 52


class CardGameEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(self, render_mode=None):
        self._counter = 0
        self._deck = np.arange(NUM_CARDS)

        # Each players board is multibinary representing if we have that card or not
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.MultiBinary(NUM_CARDS),
                "dealer": spaces.MultiBinary(NUM_CARDS),
            }
        )

        # We have 52 actions, corresponding to choosing / not choosing a card
        self.action_space = spaces.MultiBinary(NUM_CARDS)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_info(self):
        return {}

    def _get_obs(self):
        return {"agent": self._agent_board, "dealer": self._dealer_board}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # shuffle deck
        self._counter = 0
        self.np_random.shuffle(self._deck)

        # Reset to empty boards
        self._agent_board = np.zeros(NUM_CARDS, dtype=np.int8)
        self._dealer_board = np.zeros(NUM_CARDS, dtype=np.int8)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _outcome(self):
        agent_cards = [pokerkit.Deck.STANDARD[c] for c in np.flatnonzero(self._agent_board)]
        dealer_cards = [pokerkit.Deck.STANDARD[c] for c in np.flatnonzero(self._dealer_board)]

        h_agent = pokerkit.StandardHighHand.from_game(agent_cards)
        h_dealer = pokerkit.StandardHighHand.from_game(dealer_cards)

        if h_agent > h_dealer:
            return 1
        else:
            return -1

    def step(self, action):
        terminated = False
        agent_hit = False
        while self._counter < NUM_CARDS and not agent_hit:
            card = self._deck[self._counter]
            self._counter += 1

            if action[card] == 1:
                self._agent_board[card] = 1
                agent_hit = True
            else:
                self._dealer_board[card] = 1

        if self._counter == NUM_CARDS or self._agent_board.sum() >= 5:
            terminated = True

        reward = 0
        if terminated:
            while self._dealer_board.sum() < 8:
                card = self._deck[self._counter]
                self._dealer_board[card] = 1
                self._counter += 1

            reward = self._outcome()

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "ansi":
            agent_cards = [pokerkit.Deck.STANDARD[c] for c in np.flatnonzero(self._agent_board)]
            dealer_cards = [pokerkit.Deck.STANDARD[c] for c in np.flatnonzero(self._dealer_board)]

            # h_agent = pokerkit.StandardHighHand.from_game(agent_cards)
            # h_dealer = pokerkit.StandardHighHand.from_game(dealer_cards)

            state = "AGENT: {}\nDEALER: {}\n".format(
                agent_cards,
                dealer_cards
            )

            if len(agent_cards) >= 5:
                h_agent = pokerkit.StandardHighHand.from_game(agent_cards)
                h_dealer = pokerkit.StandardHighHand.from_game(dealer_cards)

                state += "AGENT: {}\nDEALER: {}\nAGENT {}\n".format(
                    h_agent,
                    h_dealer,
                    "WINS" if h_agent > h_dealer else "LOSES"
                )

            return state
