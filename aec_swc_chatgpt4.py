import functools
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from gymnasium.spaces import Discrete

# Custom variables for your game
NUM_PLANETS = 42
NUM_ITERS = 100

# Define constants for actions
MOVE_FLEET = 0
BUILD_FLEET = 1

class StarSystem:
    def __init__(self, xpos, ypos, name):
        self.xpos = xpos
        self.ypos = ypos
        self.name = name
        self.hyperlanes = []
        self.controlled_by = None  # None means unoccupied

# initialize planets
alderaan = StarSystem(1200, 1510, "Alderaan")
alsakan = StarSystem(1060, 1520, "Alsakan")
bespin = StarSystem(750, 2430, "Bespin")
bilbringi = StarSystem(780, 1260, "Bilbringi")
bonadan = StarSystem(1750, 510, "Bonadan")
bothawui = StarSystem(1910, 2160, "Bothawui")
corellia = StarSystem(1170, 1850, "Corellia")
coruscant = StarSystem(900, 1480, "Coruscant")
dantooine = StarSystem(1060, 690, "Dantooine")
denon = StarSystem(1390, 2120, "Denon")
dromundKaas = StarSystem(2040, 620, "Dromund Kaas")
eriadu = StarSystem(1220, 2770, "Eriadu")
felucia = StarSystem(1930, 920, "Felucia")
geonosis = StarSystem(1940, 2550, "Geonosis")
helska = StarSystem(1050, 340, "Helska IV")
honoghr = StarSystem(2210, 1540, "Honoghr")
hoth = StarSystem(670, 2770, "Hoth")
kamino = StarSystem(2140, 2260, "Kamino")
kashyyyk = StarSystem(1700, 1470, "Kashyyyk")
korriban = StarSystem(1910, 740, "Korriban")
kuat = StarSystem(1240, 1670, "Kuat")
lego = StarSystem(2180, 1100, "Lego")
mandalore = StarSystem(1560, 1050, "Mandalore")
monCala = StarSystem(2400, 1000, "Mon Cala")
mustafar = StarSystem(890, 3000, "Mustafar")
muunilinst = StarSystem(740, 630, "Muunilinst")
mygeeto = StarSystem(870, 810, "Mygeeto")
naboo = StarSystem(1480, 2600, "Naboo")
nalHutta = StarSystem(2020, 1810, "Nal Hutta")
onderon = StarSystem(1450, 1470, "Onderon")
ordMantell = StarSystem(940, 1080, "Ord Mantell")
ryloth = StarSystem(1910, 2760, "Ryloth")
saleucami = StarSystem(1970, 1240, "Saleucami")
serenno = StarSystem(1150, 680, "Serenno")
sernpidal = StarSystem(1180, 480, "Sernpidal")
sullust = StarSystem(1120, 2570, "Sullust")
taris = StarSystem(1280, 1070, "Taris")
tatooine = StarSystem(2040, 2430, "Tatooine")
tund = StarSystem(2480, 1280, "Tund")
yagDhul = StarSystem(1000, 2200, "Yag'Dhul")
yavin = StarSystem(1720, 830, "Yavin IV")
star_systems = [alderaan, alsakan, bespin, bilbringi, bonadan, bothawui, corellia, coruscant, dantooine, denon, dromundKaas, eriadu, felucia, geonosis, helska, honoghr, hoth, kamino, kashyyyk, korriban, kuat, lego, mandalore, monCala, mustafar, muunilinst, mygeeto, naboo, nalHutta, onderon, ordMantell, ryloth, saleucami, serenno, sernpidal, sullust, taris, tatooine, tund, yagDhul, yavin]

# Create hyperlanes list
hyperlanes = []

def add_hyperlanes(system1, system2):
    hyperlanes.append((system1, system2))
    system1.hyperlanes.append(system2)
    system2.hyperlanes.append(system1)

# Creating hyperlanes
add_hyperlanes(alderaan, alsakan)
add_hyperlanes(alderaan, kuat)
add_hyperlanes(alsakan, coruscant)
add_hyperlanes(alsakan, corellia)
add_hyperlanes(bespin, hoth)
add_hyperlanes(bespin, yagDhul)
add_hyperlanes(bilbringi, coruscant)
add_hyperlanes(bilbringi, ordMantell)
add_hyperlanes(bonadan, dromundKaas)
add_hyperlanes(bonadan, serenno)
add_hyperlanes(bothawui, denon)
add_hyperlanes(bothawui, kamino)
add_hyperlanes(bothawui, nalHutta)
add_hyperlanes(corellia, kuat)
add_hyperlanes(corellia, denon)
add_hyperlanes(corellia, yagDhul)
add_hyperlanes(dantooine, mygeeto)
add_hyperlanes(dantooine, sernpidal)
add_hyperlanes(denon, ryloth)
add_hyperlanes(denon, sullust)
add_hyperlanes(dromundKaas, korriban)
add_hyperlanes(eriadu, mustafar)
add_hyperlanes(eriadu, naboo)
add_hyperlanes(eriadu, sullust)
add_hyperlanes(felucia, korriban)
add_hyperlanes(felucia, saleucami)
add_hyperlanes(felucia, yavin)
add_hyperlanes(geonosis, ryloth)
add_hyperlanes(geonosis, tatooine)
add_hyperlanes(helska, muunilinst)
add_hyperlanes(helska, sernpidal)
add_hyperlanes(honoghr, nalHutta)
add_hyperlanes(honoghr, saleucami)
add_hyperlanes(honoghr, tund)
add_hyperlanes(hoth, mustafar)
add_hyperlanes(kamino, tatooine)
add_hyperlanes(kashyyyk, nalHutta)
add_hyperlanes(kashyyyk, onderon)
add_hyperlanes(kashyyyk, saleucami)
add_hyperlanes(kuat, onderon)
add_hyperlanes(lego, monCala)
add_hyperlanes(lego, saleucami)
add_hyperlanes(mandalore, serenno)
add_hyperlanes(mandalore, taris)
add_hyperlanes(mandalore, yavin)
add_hyperlanes(monCala, tund)
add_hyperlanes(muunilinst, mygeeto)
add_hyperlanes(mygeeto, ordMantell)
add_hyperlanes(naboo, ryloth)
add_hyperlanes(onderon, taris)
add_hyperlanes(ordMantell, taris)
add_hyperlanes(serenno, sernpidal)
add_hyperlanes(serenno, yavin)
add_hyperlanes(sullust, yagDhul)

# Define the environment class for the space conquest game
class SpaceConquestEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "space_conquest"}
    
    def __init__(self, render_mode=None):
        print("we got here 0.")
        self.possible_agents = ["faction_" + str(i) for i in range(6)]  # One agent per faction
        self.agent_name_mapping = dict(zip(self.possible_agents, range(len(self.possible_agents))))
        # self._action_spaces = {agent: Discrete(2) for agent in self.possible_agents}  # Two actions: move fleet, build fleet
        self._action_spaces = {agent: Discrete(2) for agent in self.possible_agents}  # Two actions: move fleet, build fleet
        self._observation_spaces = {agent: Discrete(NUM_PLANETS) for agent in self.possible_agents}  # Observation: list of planets
        self.render_mode = render_mode

    # @functools.lru_cache(maxsize=None)
    # def observation_space(self, agent):
    #     # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
    #     return Discrete(42)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        print("we got here 1.")
        return Discrete(2)

    def reset(self, seed=None, options=None):
        print("we got here 2.")
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Reset the game state
        for planet in star_systems:
            planet.controlled_by = None  # Reset all planets as unoccupied
        self.fleets = {agent: [] for agent in self.agents}  # Track fleet locations per faction
        
        # Setup for the first turn
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        # print("we got here 3.")
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return
        
        agent = self.agent_selection
        # Perform the action for the current agent

        if action["type"] == MOVE_FLEET:
            start = action["from"]
            destination = action["to"]

            if start in self.fleet_positions[agent] and self.fleet_positions[agent][start] > 0:
                # Check if the move is valid (1 or 2 hyperlanes away)
                if self.is_valid_move(start, destination):
                    # Move fleet
                    self.fleet_positions[agent][start] -= 1
                    if self.fleet_positions[agent][start] == 0:
                        del self.fleet_positions[agent][start]  # Remove empty entries
                    
                    if destination in self.fleet_positions[agent]:
                        self.fleet_positions[agent][destination] += 1
                    else:
                        self.fleet_positions[agent][destination] = 1
        elif action == BUILD_FLEET:
            # Implement fleet building logic
            for fleet in self.fleets:
                if fleet.position == "none":
                    fleet.position = self.homeworld
        
        # Update rewards and termination conditions
        self.rewards[agent] = self.calculate_rewards(agent)
        
        # Move to the next agent
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def calculate_rewards(self, agent):
        # print("we got here 4.")
        # Calculate rewards based on the number of planets controlled
        print(f'rewards = {sum(1 for planet in star_systems if planet.controlled_by == agent)}')
        return sum(1 for planet in star_systems if planet.controlled_by == agent)

    def render(self):
        # print("we got here 5.")
        if self.render_mode == "human":
            # Render game state for the human player
            print(f"Current state: {self.fleets}")
        
    def observe(self, agent):
        # print("we got here 6.")
        # Observation logic
        return np.array([planet.controlled_by for planet in star_systems])

    def close(self):
        print("we got here 7.")
        pass

# Example of running the environment
env = SpaceConquestEnv(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    print("New turn.")
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()  # Random action for now
    env.step(action)
env.close()