import gymnasium


def register_envs():
    gymnasium.register(
        id="PrusaCoreOne-v0",
        entry_point="digiprinter.envs.single_agent:PrusaCoreOneEnv",
    )
