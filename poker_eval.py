import pokerkit.utilities
from pokerkit import *
import numpy as np
# h0 = StandardLowHand('TsJsQsKsAs')
# h1 = StandardLowHand('AcAsAd2s4s5c9cTh')

hero = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0]

dealer = [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1, 0, 0, 0]

agent_cards = [pokerkit.Deck.STANDARD[c] for c in np.flatnonzero(hero)]
dealer_cards = [pokerkit.Deck.STANDARD[c] for c in np.flatnonzero(dealer)]

h_agent = pokerkit.StandardHighHand.from_game(agent_cards)
h_dealer = pokerkit.StandardHighHand.from_game(dealer_cards)

print(h_agent)
print(h_dealer)

c1 = pokerkit.utilities.Deck.STANDARD[10]
c2 = pokerkit.utilities.Deck.STANDARD[11]
c3 = pokerkit.utilities.Deck.STANDARD[12]
c4 = pokerkit.utilities.Deck.STANDARD[13]
c5 = pokerkit.utilities.Deck.STANDARD[14]
c6 = pokerkit.utilities.Deck.STANDARD[15]
print(type(pokerkit.utilities.Deck.STANDARD))

hand = StandardHighHand.from_game([c1, c2, c3, c4, c5, c6])

print(hand)