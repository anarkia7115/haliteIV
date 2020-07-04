import gym

def get_env():
    # env = gym.make("MountainCar-v0")
    env = gym.make('FrozenLake-v0')
    return env


def main():
    env = get_env()
    #print(env.action_space)
    env.render()


main()
