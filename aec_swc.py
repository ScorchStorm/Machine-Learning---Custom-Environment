import functools

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

ROCK = 0
PAPER = 1
SCISSORS = 2
NONE = 3
MOVES = ["ROCK", "PAPER", "SCISSORS", "None"]
NUM_ITERS = 100
REWARD_MAP = {
    (ROCK, ROCK): (0, 0),
    (ROCK, PAPER): (-1, 1),
    (ROCK, SCISSORS): (1, -1),
    (PAPER, ROCK): (1, -1),
    (PAPER, PAPER): (0, 0),
    (PAPER, SCISSORS): (-1, 1),
    (SCISSORS, ROCK): (-1, 1),
    (SCISSORS, PAPER): (1, -1),
    (SCISSORS, SCISSORS): (0, 0),
}


def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    """
    The metadata holds environment constants. From gymnasium, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, render_mode=None):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.possible_agents = ["player_" + str(r) for r in range(2)]

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # optional: we can define the observation and action spaces here as attributes to be used in their corresponding methods
        self._action_spaces = {agent: Discrete(3) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: Discrete(4) for agent in self.possible_agents
        }
        self.render_mode = render_mode

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Discrete(4)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(3)

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if len(self.agents) == 2:
            string = "Current state: Agent1: {} , Agent2: {}".format(
                MOVES[self.state[self.agents[0]]], MOVES[self.state[self.agents[1]]]
            )
        else:
            string = "Game over"
        print(string)

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: NONE for agent in self.agents}
        self.observations = {agent: NONE for agent in self.agents}
        self.num_moves = 0
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # stores action of current agent
        self.state[self.agent_selection] = action

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary
            self.rewards[self.agents[0]], self.rewards[self.agents[1]] = REWARD_MAP[
                (self.state[self.agents[0]], self.state[self.agents[1]])
            ]

            self.num_moves += 1
            # The truncations dictionary must be updated for all players.
            self.truncations = {
                agent: self.num_moves >= NUM_ITERS for agent in self.agents
            }

            # observe the current state
            for i in self.agents:
                self.observations[i] = self.state[
                    self.agents[1 - self.agent_name_mapping[i]]
                ]
        else:
            # necessary so that observe() returns a reasonable observation at all times.
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = NONE
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

hyperlanes = []
star_systems = []

def add_hyperlanes(system1, system2):
    hyperlanes.append((system1, system2))
    system1.hyperlanes.append(system2)
    system2.hyperlanes.append(system1)

class StarSystem:
    def __init__(self, xpos, ypos, name):
        self.xpos = xpos
        self.ypos = ypos
        self.name = name
        self.hyperlanes = []
        star_systems.append(self)

# Creating star systems
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

# def add_hyperlanes(system1, system2):
#     hyperlanes.Append((system1, system2));
#     system1.hyperlanes.Append(system2)
#     system2.hyperlanes.append(system1)

# class star_system:
    
#     def __init__(self, xpos, ypos, name):
#         self.xpos = xpos
#         self.ypos = ypos
#         self.name = name
#         self.hyperlanes = []
#         star_systems.Append(self)



# star_systems = {}
# star_systems["alderaan"] = (1200, 1510, "Alderaan")
# star_systems["alsakan"] = (1060, 1520, "Alsakan")
# star_systems["bespin"] = (750, 2430, "Bespin")
# star_systems["bilbringi"] = (780, 1260, "Bilbringi")
# star_systems["bonadan"] =  (1750, 510, "Bonadan")
# star_systems["bothawui"] = (1910, 2160, "Bothawui")
# star_systems["corellia"] = (1170, 1850, "Corellia")
# star_systems["coruscant"] = (900, 1480, "Coruscant")
# star_systems["dantooine"] = (1060, 690, "Dantooine")
# star_systems["denon"] = (1390, 2120, "Denon")
# star_systems["dromundKaas"] = (2040, 620, "Dromund Kaas")
# star_systems["eriadu"] = (1220, 2770, "Eriadu")
# star_systems["felucia"] = (1930, 920, "Felucia")
# star_systems["geonosis"] = (1940, 2550, "Geonosis")
# star_systems["helska"] = (1050, 340, "Helska IV")
# star_systems["honoghr"] = (2210, 1540, "Honoghr")
# star_systems["hoth"] = (670, 2770, "Hoth")
# star_systems["kamino"] = (2140, 2260, "Kamino")
# star_systems["kashyyyk"] = (1700, 1470, "kashyyyk", [], [])
# star_systems["korriban"] = (1910, 740, "Korriban", [], [])
# star_systems["kuat"] = (1240, 1670, "Kuat", [], [])
# star_systems["lego"] = (2180, 1100, "Lego", [], [])
# star_systems["mandalore"] = (1560, 1050, "Mandalore", [], [])
# star_systems["monCala"] = (2400, 1000, "Mon Cala", [], [])
# star_systems["mustafar"] = (890, 3000, "Mustafar", [], [])
# star_systems["muunilinst"] = (740, 630, "Muunilinst", [], [])
# star_systems["mygeeto"] = (870, 810, "Mygeeto", [], [])
# star_systems["naboo"] = (1480, 2600, "Naboo", [], [])
# star_systems["nalHutta"] = (2020, 1810, "Nal Hutta", [], [])
# star_systems["onderon"] = (1450, 1470, "Onderon", [], [])
# star_systems["ordMantell"] = (940, 1080, "Ord Mantell", [], [])
# star_systems["ryloth"] = (1910, 2760, "Ryloth", [], [])
# star_systems["saleucami"] = (1970, 1240, "Saleucami", [], [])
# star_systems["serenno"] = (1150, 680, "Serenno", [], [])
# star_systems["sernpidal"] = (1180, 480, "Sernpidal", [], [])
# star_systems["sullust"] = (1120, 2570, "Sullust", [], [])
# star_systems["taris"] = (1280, 1070, "Taris", [], [])
# star_systems["tatooine"] = (2040, 2430, "Tatooine", [], [])
# star_systems["tund"] = (2480, 1280, "Tund", [], [])
# star_systems["yagDhul"] = (1000, 2200, "Yag'Dhul", [], [])
# star_systems["yavin"] = (1720, 830, "Yavin IV", [], [])

# add_hyperlanes("alderaan", "alsakan");
# add_hyperlanes("alderaan", "kuat");
# add_hyperlanes("alsakan", "coruscant");
# add_hyperlanes("alsakan", "corellia");
# add_hyperlanes("bespin", "endor");
# add_hyperlanes("bespin", "hoth");
# add_hyperlanes("bespin", "yagDhul");
# add_hyperlanes("bilbringi", "coruscant");
# add_hyperlanes("bilbringi", "ordMantell");
# add_hyperlanes("bonadan", "dromundKaas");
# add_hyperlanes("bonadan", "serenno");
# add_hyperlanes("bothawui", "denon");
# add_hyperlanes("bothawui", "kamino");
# add_hyperlanes("bothawui", "nalHutta");
# add_hyperlanes("corellia", "kuat");
# add_hyperlanes("corellia", "denon");
# add_hyperlanes("corellia", "yagDhul");
# add_hyperlanes("dantooine", "mygeeto");
# add_hyperlanes("dantooine", "sernpidal");
# add_hyperlanes("denon", "ryloth");
# add_hyperlanes("denon", "sullust");
# add_hyperlanes("dromundKaas", "korriban");
# add_hyperlanes("endor", "hoth");
# add_hyperlanes("eriadu", "mustafar");
# add_hyperlanes("eriadu", "naboo");
# add_hyperlanes("eriadu", "sullust");
# add_hyperlanes("felucia", "korriban");
# add_hyperlanes("felucia", "saleucami");
# add_hyperlanes("felucia", "yavin");
# add_hyperlanes("geonosis", "ryloth");
# add_hyperlanes("geonosis", "tatooine");
# add_hyperlanes("helska", "muunilinst");
# add_hyperlanes("helska", "sernpidal");
# add_hyperlanes("honoghr", "nalHutta");
# add_hyperlanes("honoghr", "saleucami");
# add_hyperlanes("honoghr", "tund");
# add_hyperlanes("hoth", "mustafar");
# add_hyperlanes("kamino", "tatooine");
# add_hyperlanes("kashyyyk", "nalHutta");
# add_hyperlanes("kashyyyk", "onderon");
# add_hyperlanes("kashyyyk", "saleucami");
# add_hyperlanes("kuat", "onderon");
# add_hyperlanes("lego", "monCala");
# add_hyperlanes("lego", "saleucami");
# add_hyperlanes("mandalore", "serenno");
# add_hyperlanes("mandalore", "taris");
# add_hyperlanes("mandalore", "yavi"n);
# add_hyperlanes("monCala", "tund");
# add_hyperlanes("muunilinst", "mygeeto");
# add_hyperlanes("mygeeto", "ordMantell");
# add_hyperlanes("naboo", "ryloth");
# add_hyperlanes("onderon", "taris");
# add_hyperlanes("ordMantell", "taris");
# add_hyperlanes("serenno", "sernpidal");
# add_hyperlanes("serenno", "yavin");
# add_hyperlanes("sullust", "yagDhul");