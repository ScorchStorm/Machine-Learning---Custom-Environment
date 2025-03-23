import functools
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium.spaces import Discrete, MultiDiscrete, Box
import pygame # used to render the map for humans
import os
import time

# Initialize pygame
pygame.init()
screen_width, screen_height = 1366, 768  # Your monitor resolution
screen = pygame.display.set_mode((screen_width, screen_height))

# Load images
background_img = pygame.image.load(os.path.join(os.getcwd(),r"sprites\Galactic Map.png"))
# background_img = pygame.transform.scale(background_img, (screen_width, screen_height))
planet_img = pygame.image.load(os.path.join(os.getcwd(),r"sprites\planet.png")).convert_alpha()
fleet_images = {
    "Confederacy": pygame.image.load(os.path.join(os.getcwd(),r"sprites\Confederacy of Independent Systems - Dreadnaught.png")).convert_alpha(),
    "Empire": pygame.image.load(os.path.join(os.getcwd(),r"sprites\Galactic Empire - Dreadnaught.png")).convert_alpha(),
    "Republic": pygame.image.load(os.path.join(os.getcwd(),r"sprites\Galactic Republic - Dreadnaught.png")).convert_alpha(),
    "Mon_Calamari": pygame.image.load(os.path.join(os.getcwd(),r"sprites\Mon Calamari - Dreadnaught.png")).convert_alpha(),
    "Sith": pygame.image.load(os.path.join(os.getcwd(),r"sprites\Reconstituted Sith Empire - Dreadnaught.png")).convert_alpha(),
    "Yuuzhan_Vong": pygame.image.load(os.path.join(os.getcwd(),r"sprites\Yuuzhan Vong - Dreadnaught.png")).convert_alpha(),
}
hyperlane_images = {
    "Confederacy": pygame.image.load(os.path.join(os.getcwd(),r"sprites\Confederacy of Independent Systems - Hyperlane.png")).convert_alpha(),
    "Empire": pygame.image.load(os.path.join(os.getcwd(),r"sprites\Galactic Empire - Hyperlane.png")).convert_alpha(),
    "Republic": pygame.image.load(os.path.join(os.getcwd(),r"sprites\Galactic Republic - Hyperlane.png")).convert_alpha(),
    "Mon_Calamari": pygame.image.load(os.path.join(os.getcwd(),r"sprites\Mon Calamari - Hyperlane.png")).convert_alpha(),
    "Sith": pygame.image.load(os.path.join(os.getcwd(),r"sprites\Reconstituted Sith Empire - Hyperlane.png")).convert_alpha(),
    "Yuuzhan_Vong": pygame.image.load(os.path.join(os.getcwd(),r"sprites\Yuuzhan Vong - Hyperlane.png")).convert_alpha(),
    "Neutral": pygame.image.load(os.path.join(os.getcwd(),r"sprites\Neutral - Hyperlane.png")).convert_alpha(),
}
planet_glow_images = {
    "Confederacy": pygame.image.load(os.path.join(os.getcwd(),r"sprites\Confederacy of Independent Systems - Planet Glow.png")).convert_alpha(),
    "Empire": pygame.image.load(os.path.join(os.getcwd(),r"sprites\Galactic Empire - Planet Glow.png")).convert_alpha(),
    "Republic": pygame.image.load(os.path.join(os.getcwd(),r"sprites\Galactic Republic - Planet Glow.png")).convert_alpha(),
    "Mon_Calamari": pygame.image.load(os.path.join(os.getcwd(),r"sprites\Mon Calamari - Planet Glow.png")).convert_alpha(),
    "Sith": pygame.image.load(os.path.join(os.getcwd(),r"sprites\Reconstituted Sith Empire - Planet Glow.png")).convert_alpha(),
    "Yuuzhan_Vong": pygame.image.load(os.path.join(os.getcwd(),r"sprites\Yuuzhan Vong - Planet Glow.png")).convert_alpha(),
    "Neutral": pygame.image.load(os.path.join(os.getcwd(),r"sprites\Neutral - Planet Glow.png")).convert_alpha(),
}
insignia_images = {
    "Confederacy": pygame.image.load(os.path.join(os.getcwd(),r"sprites\Confederacy of Independent Systems - Insignia.png")).convert_alpha(),
    "Empire": pygame.image.load(os.path.join(os.getcwd(),r"sprites\Galactic Empire - Insignia.png")).convert_alpha(),
    "Republic": pygame.image.load(os.path.join(os.getcwd(),r"sprites\Galactic Republic - Insignia.png")).convert_alpha(),
    "Mon_Calamari": pygame.image.load(os.path.join(os.getcwd(),r"sprites\Mon Calamari - Insignia.png")).convert_alpha(),
    "Sith": pygame.image.load(os.path.join(os.getcwd(),r"sprites\Reconstituted Sith Empire - Insignia.png")).convert_alpha(),
    "Yuuzhan_Vong": pygame.image.load(os.path.join(os.getcwd(),r"sprites\Yuuzhan Vong - Insignia.png")).convert_alpha(),
}
screen = pygame.display.set_mode((1366, 768))  # Match your resolution

# Custom variables for your game
NUM_PLANETS = 42
NUM_ITERS = 100

# Define constants for actions
MOVE_FLEET = 0
BUILD_FLEET = 1

class StarSystem:
    def __init__(self, xpos, ypos, name, faction="Neutral"):
        self.xpos = xpos
        self.ypos = ypos
        self.name = name
        self.hyperlanes = []
        self.controlled_by = faction  # None means unoccupied

# Initialize planets
star_systems = {
    "Alderaan": StarSystem(641, 365, "Alderaan"),
    "Alsakan": StarSystem(581, 355, "Alsakan"),
    "Bespin": StarSystem(528, 607, "Bespin"),
    "Bilbringi": StarSystem(511, 290, "Bilbringi"),     
    "Bonadan": StarSystem(778, 77, "Bonadan"),
    "Bothawui": StarSystem(806, 515, "Bothawui"),       
    "Corellia": StarSystem(608, 462, "Corellia"),       
    "Coruscant": StarSystem(528, 357, "Coruscant", "Republic"),     
    "Dantooine": StarSystem(618, 160, "Dantooine"),     
    "Denon": StarSystem(713, 530, "Denon"),
    "Dromund Kaas": StarSystem(888, 80, "Dromund Kaas", "Sith"),
    "Eriadu": StarSystem(646, 692, "Eriadu"),
    "Endor": StarSystem(416, 630, "Endor", "Empire"),
    "Felucia": StarSystem(841, 217, "Felucia"),
    "Geonosis": StarSystem(826, 637, "Geonosis", "Confederacy"),       
    "Helska IV": StarSystem(603, 35, "Helska IV", "Yuuzhan_Vong"),
    "Honoghr": StarSystem(906, 385, "Honoghr"),
    "Hoth": StarSystem(508, 692, "Hoth"),
    "Kamino": StarSystem(938, 552, "Kamino"),
    "Kashyyyk": StarSystem(791, 367, "Kashyyyk"),
    "Korriban": StarSystem(831, 135, "Korriban"),
    "Kuat": StarSystem(651, 417, "Kuat"),
    "Lego": StarSystem(898, 262, "Lego"),
    "Mandalore": StarSystem(731, 225, "Mandalore"),
    "Mon Cala": StarSystem(966, 237, "Mon Cala", "Mon_Calamari"),
    "Mustafar": StarSystem(563, 737, "Mustafar"),
    "Muunilinst": StarSystem(501, 95, "Muunilinst"),
    "Mygeeto": StarSystem(533, 177, "Mygeeto"),
    "Naboo": StarSystem(711, 662, "Naboo"),
    "Nal Hutta": StarSystem(846, 452, "Nal Hutta"),
    "Onderon": StarSystem(716, 355, "Onderon"),
    "Ord Mantell": StarSystem(563, 245, "Ord Mantell"),
    "Ryloth": StarSystem(818, 715, "Ryloth"),
    "Saleucami": StarSystem(833, 310, "Saleucami"),
    "Serenno": StarSystem(716, 145, "Serenno"),
    "Sernpidal": StarSystem(648, 82, "Sernpidal"),
    "Sullust": StarSystem(633, 617, "Sullust"),
    "Taris": StarSystem(661, 267, "Taris"),
    "Tatooine": StarSystem(876, 582, "Tatooine"),
    "Tund": StarSystem(986, 320, "Tund"),
    "Yag'Dhul": StarSystem(591, 550, "Yag'Dhul"),
    "Yavin IV": StarSystem(783, 195, "Yavin IV"),
}
# # Initialize planets
# star_systems = {
#     "Alderaan": StarSystem(641, 365, "Alderaan", "Republic"),
#     "Alsakan": StarSystem(581, 355, "Alsakan", "Republic"),
#     "Bespin": StarSystem(528, 607, "Bespin", "Empire"),
#     "Bilbringi": StarSystem(511, 290, "Bilbringi", "Republic"),     
#     "Bonadan": StarSystem(778, 77, "Bonadan", "Sith"),
#     "Bothawui": StarSystem(806, 515, "Bothawui", "Confederacy"),       
#     "Corellia": StarSystem(608, 462, "Corellia", "Republic"),       
#     "Coruscant": StarSystem(528, 357, "Coruscant", "Republic"),     
#     "Dantooine": StarSystem(618, 160, "Dantooine", "Yuuzhan_Vong"),     
#     "Denon": StarSystem(713, 530, "Denon", "Confederacy"),
#     "Dromund Kaas": StarSystem(888, 80, "Dromund Kaas", "Sith"),
#     "Eriadu": StarSystem(646, 692, "Eriadu", "Empire"),
#     "Endor": StarSystem(416, 630, "Endor", "Empire"),
#     "Felucia": StarSystem(841, 217, "Felucia", "Sith"),
#     "Geonosis": StarSystem(826, 637, "Geonosis", "Confederacy"),       
#     "Helska IV": StarSystem(603, 35, "Helska IV", "Yuuzhan_Vong"),
#     "Honoghr": StarSystem(906, 385, "Honoghr", "Mon_Calamari"),
#     "Hoth": StarSystem(508, 692, "Hoth", "Empire"),
#     "Kamino": StarSystem(938, 552, "Kamino", "Confederacy"),
#     "Kashyyyk": StarSystem(791, 367, "Kashyyyk", "Mon_Calamari"),
#     "Korriban": StarSystem(831, 135, "Korriban", "Sith"),
#     "Kuat": StarSystem(651, 417, "Kuat", "Republic"),
#     "Lego": StarSystem(898, 262, "Lego", "Mon_Calamari"),
#     "Mandalore": StarSystem(731, 225, "Mandalore", "Sith"),
#     "Mon Cala": StarSystem(966, 237, "Mon Cala", "Mon_Calamari"),
#     "Mustafar": StarSystem(563, 737, "Mustafar", "Empire"),
#     "Muunilinst": StarSystem(501, 95, "Muunilinst", "Yuuzhan_Vong"),
#     "Mygeeto": StarSystem(533, 177, "Mygeeto", "Yuuzhan_Vong"),
#     "Naboo": StarSystem(711, 662, "Naboo", "Confederacy"),
#     "Nal Hutta": StarSystem(846, 452, "Nal Hutta", "Mon_Calamari"),
#     "Onderon": StarSystem(716, 355, "Onderon", "Republic"),
#     "Ord Mantell": StarSystem(563, 245, "Ord Mantell", "Yuuzhan_Vong"),
#     "Ryloth": StarSystem(818, 715, "Ryloth", "Confederacy"),
#     "Saleucami": StarSystem(833, 310, "Saleucami", "Mon_Calamari"),
#     "Serenno": StarSystem(716, 145, "Serenno", "Sith"),
#     "Sernpidal": StarSystem(648, 82, "Sernpidal", "Yuuzhan_Vong"),
#     "Sullust": StarSystem(633, 617, "Sullust", "Empire"),
#     "Taris": StarSystem(661, 267, "Taris", "Yuuzhan_Vong"),
#     "Tatooine": StarSystem(876, 582, "Tatooine", "Confederacy"),
#     "Tund": StarSystem(986, 320, "Tund", "Mon_Calamari"),
#     "Yag'Dhul": StarSystem(591, 550, "Yag'Dhul", "Empire"),
#     "Yavin IV": StarSystem(783, 195, "Yavin IV", "Sith"),
# }
planet_keys = list(star_systems.keys())
planet_index_map = {planet: i for i, planet in enumerate(star_systems.keys())} # Create a mapping of planet names to their indices
        

# Create hyperlanes list
hyperlanes = []

def add_hyperlanes(system1, system2):
    hyperlanes.append((system1, system2))
    star_systems[system1].hyperlanes.append(system2)
    star_systems[system2].hyperlanes.append(system1)

# Example hyperlanes
add_hyperlanes("Alderaan", "Alsakan")
add_hyperlanes("Alderaan", "Kuat")
add_hyperlanes("Alsakan", "Coruscant")
add_hyperlanes("Alsakan", "Corellia")
add_hyperlanes("Bespin", "Endor")
add_hyperlanes("Bespin", "Hoth")
add_hyperlanes("Bespin", "Yag'Dhul")
add_hyperlanes("Bilbringi", "Coruscant")
add_hyperlanes("Bilbringi", "Ord Mantell")
add_hyperlanes("Bonadan", "Dromund Kaas")
add_hyperlanes("Bonadan", "Serenno")
add_hyperlanes("Bothawui", "Denon")
add_hyperlanes("Bothawui", "Kamino")
add_hyperlanes("Bothawui", "Nal Hutta")
add_hyperlanes("Corellia", "Kuat")
add_hyperlanes("Corellia", "Denon")
add_hyperlanes("Corellia", "Yag'Dhul")
add_hyperlanes("Dantooine", "Mygeeto")
add_hyperlanes("Dantooine", "Sernpidal")
add_hyperlanes("Denon", "Ryloth")
add_hyperlanes("Denon", "Sullust")
add_hyperlanes("Dromund Kaas", "Korriban")
add_hyperlanes("Eriadu", "Mustafar")
add_hyperlanes("Eriadu", "Naboo")
add_hyperlanes("Eriadu", "Sullust")
add_hyperlanes("Endor", "Hoth")
add_hyperlanes("Felucia", "Korriban")
add_hyperlanes("Felucia", "Saleucami")
add_hyperlanes("Felucia", "Yavin IV")
add_hyperlanes("Geonosis", "Ryloth")
add_hyperlanes("Geonosis", "Tatooine")
add_hyperlanes("Helska IV", "Muunilinst")
add_hyperlanes("Helska IV", "Sernpidal")
add_hyperlanes("Honoghr", "Nal Hutta")
add_hyperlanes("Honoghr", "Saleucami")
add_hyperlanes("Honoghr", "Tund")
add_hyperlanes("Hoth", "Mustafar")
add_hyperlanes("Kamino", "Tatooine")
add_hyperlanes("Kashyyyk", "Nal Hutta")
add_hyperlanes("Kashyyyk", "Onderon")
add_hyperlanes("Kashyyyk", "Saleucami")
add_hyperlanes("Kuat", "Onderon")
add_hyperlanes("Lego", "Mon Cala")
add_hyperlanes("Lego", "Saleucami")
add_hyperlanes("Mandalore", "Serenno")
add_hyperlanes("Mandalore", "Taris")
add_hyperlanes("Mandalore", "Yavin IV")
add_hyperlanes("Mon Cala", "Tund")
add_hyperlanes("Muunilinst", "Mygeeto")
add_hyperlanes("Mygeeto", "Ord Mantell")
add_hyperlanes("Naboo", "Ryloth")
add_hyperlanes("Onderon", "Taris")
add_hyperlanes("Ord Mantell", "Taris")
add_hyperlanes("Serenno", "Sernpidal")
add_hyperlanes("Serenno", "Yavin IV")
add_hyperlanes("Sullust", "Yag'Dhul")

class SpaceConquestEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "space_conquest"}
    def __init__(self, render_mode=None):
        # Define agents (factions)
        self.possible_agents = ["Confederacy", "Empire", "Republic", "Mon_Calamari", "Sith", "Yuuzhan_Vong"]
        self.agent_name_mapping = dict(zip(self.possible_agents, range(len(self.possible_agents))))
        # Define homeworlds
        self.homeworlds = {
            "Confederacy": "Geonosis",
            "Empire": "Endor",
            "Republic": "Coruscant",
            "Mon_Calamari": "Mon Cala",
            "Sith": "Dromund Kaas",
            "Yuuzhan_Vong": "Helska IV"
        }
        # Define initial fleet positions (each agent has 1 fleet at their capital)
        self.fleet_positions = {
            "Confederacy": ["Geonosis", None, None],
            "Empire": ["Endor", None, None],
            "Republic": ["Coruscant", None, None],
            "Mon_Calamari": ["Mon Cala", None, None],
            "Sith": ["Dromund Kaas", None, None],
            "Yuuzhan_Vong": ["Helska IV", None, None]
        }
        # self.fleet_positions = {
        #     "Confederacy": ["Naboo", "Denon", "Bothawui"],
        #     "Empire": ["Yag'Dhul", "Sullust", "Eriadu"],
        #     "Republic": ["Bilbringi", "Onderon", "Corellia"],
        #     "Mon_Calamari": ["Saleucami", "Kashyyyk", "Nal Hutta"],
        #     "Sith": ["Serenno", "Mandalore", "Felucia"],
        #     "Yuuzhan_Vong": ["Sernpidal", "Taris", "Ord Mantell"]
        # }
        self.original_fleet_positions = self.fleet_positions.copy()
        self.planets_controlled = {
            "Confederacy": ["Geonosis"],
            "Empire": ["Endor"],
            "Republic": ["Coruscant"],
            "Mon_Calamari": ["Mon Cala"],
            "Sith": ["Dromund Kaas"],
            "Yuuzhan_Vong": ["Helska IV"]
        }
        self.original_planets_controlled = self.planets_controlled.copy()
        # Planets and ownership tracking
        # self.planets = ["Planet_" + str(i) for i in range(42)]
        self.planets = [planet for planet in star_systems]
        self.planet_owners = {star_systems[planet].controlled_by for planet in star_systems}  # Track which faction owns each planet
        print(f'{self.planet_owners = }')
        # Define action space (each fleet can move to a different planet)
        NUM_PLANETS = 42  # Assuming 42 planets in the game
        self._action_spaces = {agent: MultiDiscrete([NUM_PLANETS] * 3) for agent in self.possible_agents}
        # Define observation space (each agent sees its 3 fleets' positions)
        self._observation_spaces = {agent: Box(low=0, high=NUM_PLANETS, shape=(3,), dtype=int) for agent in self.possible_agents}
        # Action space: Each fleet can move (or remain idle)
        self._action_spaces = {agent: MultiDiscrete([42, 42, 42]) for agent in self.possible_agents}
        # Rewards tracking
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        # Render mode
        self.render_mode = render_mode
        self.render()
        # Initialize environment state
        # self.reset()

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(2)

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        # for planet in star_systems.values():
        #     planet.controlled_by = None
        self.fleet_positions = self.original_fleet_positions
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        
    def center(self, img, pos):
        xpos, ypos = pos
        img_rect = img.get_rect()  # Get image size
        screen.blit(img, (xpos-img_rect.width//2, ypos-img_rect.height//2))

    def render(self):
        screen.fill([0,0,0])
        self.center(background_img, (screen_width//2, screen_height//2))  # Draw background
        
        for system1, system2 in hyperlanes: # Function to draw hyperlane between two planets
            if (star_systems[system1].controlled_by == star_systems[system2].controlled_by and star_systems[system1].controlled_by != None):
                hyperlane_img = hyperlane_images.get(star_systems[system1].controlled_by)
            else:
                hyperlane_img = hyperlane_images.get("Neutral")
            x1, y1 = star_systems[system1].xpos, star_systems[system1].ypos # get coordinates of planet1
            x2, y2 = star_systems[system2].xpos, star_systems[system2].ypos # get coordinates of planet2
            angle = np.arctan2(y2 - y1, x2 - x1) # calculate the angleprint(obj.__class__.__name__)
            distance = np.hypot(x2 - x1, y2 - y1) # calculate the distance
            scaled_hyperlane = pygame.transform.scale(hyperlane_img, (int(distance), 60)) # sale the hyperlane image to the distance
            rotated_hyperlane = pygame.transform.rotate(scaled_hyperlane, -np.degrees(angle)) # rotate the hyperlane
            hyperlane_rect = rotated_hyperlane.get_rect(center=((x1 + x2) // 2, (y1 + y2) // 2)) # Calculate the position to center the hyperlane image between the two planets
            screen.blit(rotated_hyperlane, hyperlane_rect.topleft) # Draw the rotated and scaled hyperlane on the screen

        for planet in star_systems.keys(): # Draw planet glows
            faction = star_systems[planet].controlled_by
            planet_glow_img = planet_glow_images.get(faction)
            x, y = star_systems[planet].xpos, star_systems[planet].ypos
            self.center(planet_glow_img, (x, y))

        for planet in star_systems.values():  # Draw planets
            if planet.name not in self.homeworlds.values():  # Ensure it's not a homeworld
                if not any(planet.name in fleet for fleet in self.fleet_positions.values()):
                    self.center(planet_img, (planet.xpos, planet.ypos))

        for home_planet in self.homeworlds.values(): # Draw home system insignias
            faction = star_systems[home_planet].controlled_by
            if home_planet not in self.fleet_positions[faction]:
                x, y = star_systems[home_planet].xpos, star_systems[home_planet].ypos
                insignia_img = insignia_images.get(faction)
                self.center(insignia_img, (x, y))

        for agent, fleet_positions in self.fleet_positions.items(): # Draw fleets
            fleet_img = fleet_images.get(agent)
            if fleet_img:
                for planet in fleet_positions:
                    if planet != None and planet != -1:  # Ignore None fleets
                        if type(planet) == type(np.int64(23)):
                            planet = planet_keys[planet]
                        x, y = star_systems[planet].xpos, star_systems[planet].ypos
                        self.center(fleet_img, (x, y))
        pygame.display.flip()  # Update display

    def step(self, action_dict):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return
        rewards = {agent: 0 for agent in self.possible_agents}
        # fleet_counts = {agent: sum(1 for pos in self.fleet_positions[agent] if pos is not None) for agent in self.possible_agents}
        # print(f'{fleet_counts = }')
        built_fleet = {agent: False for agent in self.possible_agents}
        for agent, action in action_dict.items():
            if self.fleet_positions[agent] != [None, None, None] or star_systems[self.homeworlds[agent]].controlled_by == agent:  # If the faction has not already lost
                self.render() # render game state
                # time.sleep(0.1)
                for i, move in enumerate(action):  # Each fleet's action
                    if self.fleet_positions[agent][i] != None:
                        if self.fleet_positions[agent][i] != None and move != self.fleet_positions[agent][i] and move is not None and planet_keys[move] not in self.fleet_positions[agent]:
                            planet = planet_keys[move]
                            self.fleet_positions[agent][i] = planet
                            faction = star_systems[planet].controlled_by
                            if star_systems[planet].controlled_by != agent:
                                if star_systems[planet].controlled_by != "Neutral" and planet in self.fleet_positions[faction]: # if a fleet is in the system that the current fleet is trying to move to
                                    loser = agent if np.random.uniform() > 0.5 else faction  # figure out who loses the battle
                                    if loser == faction:
                                        print(f'{agent} lost the battle at {planet} against {faction} and lost their fleet')
                                        self.fleet_positions[faction][self.fleet_positions[faction].index(planet)] = None
                                    elif loser == agent:
                                        print(f'{faction} lost the battle at {planet} against {agent} and lost their fleet')
                                        self.fleet_positions[agent][i] = None
                                else: # if a faction does not have a fleet in the system that is being invaded, they lose it by default
                                    loser = faction
                                if loser == faction:
                                    star_systems[planet].controlled_by = agent
                                    rewards[agent] += 10  # Reward for gaining a planet
                                    if faction != "Neutral":
                                        rewards[faction] -= 5  # Punish the old owner
                    else:
                        if star_systems[self.homeworlds[agent]].controlled_by == agent and not built_fleet[agent] and self.homeworlds[agent] not in self.fleet_positions[agent]:
                            print(f'building new fleet for {agent}')
                            self.fleet_positions[agent][i] = self.homeworlds[agent]
                            built_fleet[agent] = True
                        else:
                            print(f'could not build new fleet for {agent}')
                            print(f'{i = }, {star_systems[self.homeworlds[agent]].controlled_by == agent = }, {not built_fleet[agent] = }, {self.homeworlds[agent] not in self.fleet_positions[agent] = }')
        self._cumulative_rewards = {agent: self._cumulative_rewards[agent] + rewards[agent] for agent in self.possible_agents}
        return self.observe(agent), rewards, self.terminations, self.truncations, {}

    def observe(self, agent):
        fleet_obs = [planet_index_map.get(fleet, -1) for fleet in self.fleet_positions[agent]] # Convert fleet positions to indices using the mapping
        ownership_obs = [self.agent_name_mapping[self.planet_owners[planet]] if planet in self.planet_owners else -1 for planet in star_systems] # Encode planet ownership: Store faction index controlling each planet (-1 if unowned)
        fleet_presence_obs = []
        for planet in star_systems: # Encode fleet presence: Store the faction index of fleets at each planet (-1 if no fleet)
            fleets_at_planet = self.get_fleets_at_planet(planet)  # Dictionary {faction: number_of_fleets}
            if sum(fleets_at_planet.values()) == 0:
                fleet_presence_obs.append(-1)  # No fleets present
            else: # Store the first faction found with a fleet (or a more complex multi-agent encoding)
                occupying_faction = next((faction for faction, count in fleets_at_planet.items() if count > 0), -1)
                fleet_presence_obs.append(self.agent_name_mapping[occupying_faction])
        observation = np.array(fleet_obs + ownership_obs + fleet_presence_obs, dtype=np.int32) # Combine all observation elements
        return observation

    def get_fleets_at_planet(self, planet):
        fleets_present = {agent: 0 for agent in self.possible_agents}  # Track fleets at the given planet
        for agent in self.possible_agents:
            for fleet in self.fleet_positions[agent]:
                if fleet == planet:
                    fleets_present[agent] += 1
        return fleets_present  # Example: {'Empire': 1, 'Republic': 0, 'Sith': 2, ...}

# Example of running the environment
env = SpaceConquestEnv(render_mode="human")
env.reset(seed=42)
turn = -1

for agent in env.agent_iter():
    turn += 1
    print(f"Beginning turn {turn}")
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        actions = None
    else:
        # action = env.action_space(agent).sample()  # Random action for now
        actions = {agent: env._action_spaces[agent].sample() for agent in env.agents}
    env.step(actions)
env.close()