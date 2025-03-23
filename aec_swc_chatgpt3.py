import functools
import gymnasium
import numpy as np
from gymnasium.spaces import Discrete
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
import random

# Constants for the Star Systems and Game mechanics
NUM_ITERS = 100

# Actions and moves for the Star Trek Conquest-inspired game
MOVES = ["BUILD_FLEET", "MOVE_FLEET", "CAPTURE_PLANET", "DEFEND"]

# Game Reward Map for different planet captures
REWARD_MAP = {
    ("captured", "captured"): (0, 0),
    ("captured", "defended"): (-1, 1),
    ("defended", "captured"): (1, -1),
    ("defended", "defended"): (0, 0),
}

# Define the StarSystem class
class StarSystem:
    def __init__(self, xpos, ypos, name):
        self.xpos = xpos
        self.ypos = ypos
        self.name = name
        self.hyperlanes = []

# Initialize a list to store all the star systems
star_systems = []

# Add function to connect star systems with hyperlanes
def add_hyperlanes(system1, system2):
    system1.hyperlanes.append(system2)
    system2.hyperlanes.append(system1)

# Create the star systems (example provided, feel free to expand)
alderaan = StarSystem(1200, 1510, "Alderaan")
alsakan = StarSystem(1060, 1520, "Alsakan")
coruscant = StarSystem(700, 1200, "Coruscant")
dagobah = StarSystem(950, 1350, "Dagobah")
tatooine = StarSystem(1100, 1400, "Tatooine")
hoth = StarSystem(950, 1600, "Hoth")
# Add more star systems here...

# Connect the star systems with hyperlanes (example)
add_hyperlanes(alderaan, alsakan)
add_hyperlanes(coruscant, dagobah)
add_hyperlanes(tatooine, hoth)
# Add more hyperlanes connections here...

# Define the main environment class for the PettingZoo environment
class StarTrekConquestEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "StarTrekConquest"}

    def __init__(self, render_mode=None):
        self.possible_agents = ["AI_" + str(i) for i in range(6)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self._action_spaces = {agent: Discrete(4) for agent in self.possible_agents}  # Actions defined earlier
        self._observation_spaces = {agent: Discrete(len(star_systems)) for agent in self.possible_agents}
        self.render_mode = render_mode

        # Initialize game state
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def observation_space(self, agent):
        return Discrete(len(star_systems))

    def action_space(self, agent):
        return Discrete(4)

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn("Render method called without specifying a render mode.")
            return
        if len(self.agents) == 6:
            print("Current game state:")
            for agent in self.agents:
                print(f"Agent {agent} controls: {self.state[agent]}")

    def observe(self, agent):
        return np.array(self.observations[agent])

    def close(self):
        pass

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        # Store action of the current agent
        self.state[self.agent_selection] = action

        # Handle the rewards at the end of the round
        if self._agent_selector.is_last():
            # Implement your reward structure here, e.g. based on planets controlled
            self.rewards[self.agents[0]], self.rewards[self.agents[1]] = REWARD_MAP[(
                self.state[self.agents[0]], self.state[self.agents[1]]
            )]

            self.num_moves += 1
            self.truncations = {agent: self.num_moves >= NUM_ITERS for agent in self.agents}

            # Update the observations for each agent
            for i in self.agents:
                self.observations[i] = self.state[self.agents[1 - self.agent_name_mapping[i]]]
        else:
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = None
            self._clear_rewards()

        # Select next agent
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

# Environment wrapper for easier access and additional features
def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = StarTrekConquestEnv(render_mode=internal_render_mode)

    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)

    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

# AI agents play against each other using random actions
def run_game():
    environment = env(render_mode="human")
    environment.reset()
    for _ in range(NUM_ITERS):
        actions = {agent: random.choice([0, 1, 2, 3]) for agent in environment.agents}  # Random actions
        environment.step(actions)

if __name__ == "__main__":
    run_game()