# import aec_swc
import aec_swc_chatgpt1

env = aec_swc.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()

    env.step(action)
env.close()

###

import aec_swc
import pygame
from pettingzoo.utils import wrappers

class SWCEnv(aec_swc.env):
    def __init__(self, render_mode="human"):
        super().__init__(render_mode=render_mode)
        self.screen = None
        self.assets = {}
        self.load_assets()
    
    def load_assets(self):
        faction_data = {
            "Confederacy of Independant Systems": "Dreadnaught.png",
            "Galactic Empire": "Dreadnaught.png",
            "Galactic Republic": "Dreadnaught.png",
            "Mon Calamari": "Dreadnaught.png",
            "Reconstituted Sith Empire": "Dreadnaught.png",
            "Yuuzhan Vong": "Dreadnaught.png",
            "Neutral": "Neutral - Hyperlane.png"
        }
        for faction, file_name in faction_data.items():
            self.assets[faction] = pygame.image.load(file_name)
    
    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((1366, 768))
            pygame.display.set_caption("Star Wars Conquest")
        
        self.screen.fill((0, 0, 0))  # Clear screen
        
        for planet in self.star_systems:
            xpos, ypos = planet.xpos, planet.ypos
            faction = planet.controller
            if faction in self.assets:
                self.screen.blit(self.assets[faction], (xpos, ypos))
        
        pygame.display.flip()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        if self.screen is not None:
            pygame.quit()
            self.screen = None

# Running the game with the correct API structure
env = SWCEnv(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()
    
    env.step(action)

env.close()