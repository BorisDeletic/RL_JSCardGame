from gymnasium.envs.registration import register

register(
    id='JSCardGame-v0',
    entry_point='cardenv.cardenv:CardGameEnv'
)