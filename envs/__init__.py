# from square import SquareEnv


def incremental_environment_factory(cls):
    class AbstractEnv(cls):
        def add_new_skill(self):
            pass

    return AbstractEnv
