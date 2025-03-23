from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
import numpy as np
import os
import gymnasium
from gymnasium.spaces import Discrete, Dict, MultiBinary
import pygame
import aec_swc

class SWCEnv(aec_swc.AECEnv):
    def __init__(self, render_mode="human"):
        super().__init__()
        self.render_mode = render_mode
        self.screen = None
        self.assets = {}
        self.load_assets()
        
        # Define possible agents (example: just some names for agents)
        self.possible_agents = ["Agent_1", "Agent_2", "Agent_3", "Agent_4"]
        self.num_agents = len(self.possible_agents)
    
    def load_assets(self):
        faction_data = {
            "Confederacy of Independent Systems": "Confederacy of Independent Systems - Dreadnaught.png",
            "Galactic Empire": "Galactic Empire - Dreadnaught.png",
            "Galactic Republic": "Galactic Republic - Dreadnaught.png",
            "Mon Calamari": "Mon Calamari - Dreadnaught.png",
            "Reconstituted Sith Empire": "Reconstituted Sith Empire - Dreadnaught.png",
            "Yuuzhan Vong": "Yuuzhan Vong - Dreadnaught.png",
            "Neutral": "Neutral - Hyperlane.png"
        }
        for faction, file_name in faction_data.items():
            file_path = os.path.join("sprites", file_name)
            self.assets[faction] = pygame.image.load(file_path)
    
    def reset(self, seed=None):
        # Initialize the game state, agents, and other necessary parameters
        self.star_systems = []  # Reset star systems or planet configurations
        self.agents = self.possible_agents.copy()  # Reset agents list
        
        # Reset any other game state (planet ownership, fleet positions, etc.)
        self.planet_ownership = np.zeros((self.num_agents, self.num_planets))
        self.fleet_positions = np.zeros((self.num_agents, self.num_planets))
        
        # Reset rendering state if necessary
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((1366, 768))
            pygame.display.set_caption("Star Wars Conquest")
        
        # You could seed the random number generator if needed (optional)
        if seed is not None:
            np.random.seed(seed)

        # Return the initial observations
        return self._get_observations()
    
    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((1366, 768))
            pygame.display.set_caption("Star Wars Conquest")
        
        self.screen.fill((0, 0, 0))  # Clear the screen
        
        for planet in self.star_systems:
            xpos, ypos = planet.xpos, planet.ypos
            faction = planet.controller
            if faction in self.assets:
                self.screen.blit(self.assets[faction], (xpos, ypos))
        
        pygame.display.flip()

    def _get_observations(self):
        # Return the observations for each agent
        return {agent: self.observe(agent) for agent in self.agents}
    
    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((1366, 768))
            pygame.display.set_caption("Star Wars Conquest")
        
        self.screen.fill((0, 0, 0))  # Clear the screen
        
        for planet in self.star_systems:
            xpos, ypos = planet.xpos, planet.ypos
            faction = planet.controller
            if faction in self.assets:
                self.screen.blit(self.assets[faction], (xpos, ypos))
        
        pygame.display.flip()

    def _get_observations(self):
        # Return the observations for each agent
        return {agent: self.observe(agent) for agent in self.agents}

# Initialize and run the environment
env = SWCEnv(render_mode="human")
env.reset(seed=42)

running = True
while running:
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample()
        
        env.step(action)
    
    env.render()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
env.close()