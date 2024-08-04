import NN
from random import randint, random
from scipy.spatial import distance_matrix
import numpy as np
from math import pi, cos, sin
import json
import os

# Define agent types
AGENT_TYPES = ["Food", "Prey", "Predator"]


class Agent:
    # Note that we put the type hint for the simulation parameter as a string because the Simulation class is defined after the Agent class
    def __init__(self, type: str,
                 x_pos: float, y_pos: float, speed: float, energy: float,
                 angle: float, angular_speed: float, simulation: "Simulation"):
        """
        Initializes an agent for the simulation

        Parameters
        ----------
        type: str - the type of agent
        x_pos: float - the x position of the agent
        y_pos: float - the y position of the agent
        speed: float - the speed of the agent
        energy: float - the energy of the agent
        angle: float - the angle of the agent
        angular_speed: float - the angular speed of the agent
        simulation: Simulation - the simulation the agent is in        
        """
        assert type in AGENT_TYPES, f"Invalid agent type: {type}"

        self.simulation = simulation
        self.id = f"{type}_{simulation.get_next_id(type)}"
        self.type = type

        self.x = x_pos
        self.y = y_pos
        self.speed = speed
        self.angle = angle
        self.angular_speed = angular_speed
        self.energy = energy
        self.alive = True
        self.age = 0

        self.nn = NN.createNeuralNetwork(3, [4, 2], 2, NN.relu, 0.5)

    def move(self, closest_x: float, closest_y: float, closest_type: int) -> None:
        """
        Moves the agent based on the given inputs
        
        Parameters
        ----------
        closest_x: float - the x position of the closest agent
        closest_y: float - the y position of the closest agent
        closest_type: int - the type of the closest agent
        
        Returns
        -------
        None
        """
        
        self.speed, self.angular_speed = self.nn.forward(np.array([closest_x, closest_y, closest_type]))
        
        self.angle += self.angular_speed % (2 * pi)
        self.x += self.speed * cos(self.angle)
        self.y += self.speed * sin(self.angle)
        self.x %= self.simulation.width
        self.y %= self.simulation.height
        
        self.energy -= self.simulation.config['energy_loss_per_epoch']  # Energy cost of moving
        
        if self.energy <= 0:
            self.die()
            
    def die(self) -> None:
        """
        Kills the agent, setting its alive status to False and energy to 0

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.alive = False
        self.energy = 0

    def __str__(self) -> str:
        return f"Agent {self.id} at ({self.x}, {self.y}) with speed {self.speed} and angular speed {self.angular_speed}"


class Simulation:
    def __init__(self, width: int, height: int, num_agents: list[int]) -> None:
        """
        Initializes the simulation with the given width, height, and number of agents

        Parameters
        ----------
        width: int - the width of the simulation
        height: int - the height of the simulation
        num_agents: int - the number of agents in the simulation

        Returns
        -------
        None
        """
        
        self.read_config()
        
        self.width = width
        self.height = height
        self.current_epoch = 0
        self.current_generation = 0
        self.agents = []
        self.agent_ids = {type: 0 for type in AGENT_TYPES}
        self.log_messages = []

        self.history = {
            "Epoch": [0],
            "Food": [num_agents[0]],
            "Prey": [num_agents[1]],
            "Predator": [num_agents[2]]
        }

        for i, n in enumerate(num_agents):
            for _ in range(n):
                while 1:
                    x_pos = randint(0, width)
                    y_pos = randint(0, height)
                    if not any([agent.x == x_pos and agent.y == y_pos for agent in self.agents]):
                        break

                speed = random()
                angle = random() * 2 * pi
                # Random angular speed in the range [-pi/12, pi/12] (~ -15 to 15 degrees per second)
                angular_speed = random() * pi/6 - pi/12
                if AGENT_TYPES[i] == "Food":
                    energy = 100
                else:
                    energy = randint(50, 100)

                self.agents.append(
                    Agent(AGENT_TYPES[i], x_pos, y_pos, speed, energy, angle, angular_speed, self))
        self.log(
            f"Started simulation with {num_agents[0]} food, {num_agents[1]} prey, and {num_agents[2]} predators")

    def read_config(self) -> None:
        """
        Reads the configuration file for the simulation (config.json)
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        assert "config.json" in os.listdir(), "config.json not found"
        
        self.config = {}
        
        with open("config.json", "r") as f:
            self.config = json.load(f)        
        
        
    def get_next_id(self, type: str) -> int:
        """
        Gets the next id for an agent of the given type

        Parameters
        ----------
        type: str - the type of agent

        Returns
        -------
        int - the next id for an agent of the given type
        """
        id = self.agent_ids[type]
        self.agent_ids[type] += 1
        return id

    def get_alive_agents(self, include_food=True) -> list[Agent]:
        """
        Gets the list of agents that are alive

        Parameters
        ----------
        include_food: bool, defaults to True - whether to include food agents in the list

        Returns
        -------
        list[Agent] - the list of agents that are alive
        """

        if include_food:
            return [agent for agent in self.agents if agent.alive]
        else:
            return [agent for agent in self.agents if agent.alive and agent.type != "Food"]

    def step(self) -> None:
        """
        Runs a single step of the simulation

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        alive_agents = self.get_alive_agents(include_food=True)
        alive_agents_no_food = self.get_alive_agents(include_food=False)
        
        x_pos = [agent.x for agent in alive_agents]
        y_pos = [agent.y for agent in alive_agents]
        pos = np.column_stack((x_pos, y_pos))
        distances = distance_matrix(pos, pos)

        # Make the diagonal elements infinity so that the agent doesn't consider itself as the closest agent
        distances[range(len(distances)), range(len(distances))] = float("inf")

        # If there are no agents left, end the current generation
        if len(alive_agents_no_food) == 0:
            self.log(f"No agents left. Skipping to next generation")
            self.current_generation += 1
            return

        # Update the position of each agent
        for ag_id, agent in enumerate(alive_agents):            
            if agent.type == "Food":
                continue  # Food doesn't move

            agent.age += 1

            # Run the neural network to get the speed and angular speed
            # Inputs to the neural network are the x and y positions of the closest agent and the type of the closest agent
            # Outputs are the speed and angular speed
            closest_agent = self.agents[distances[ag_id].argmin()]
            agent.move(closest_agent.x, closest_agent.y, AGENT_TYPES.index(closest_agent.type))
 
        # Check if any 2 agents are in the same position
        self.check_collision()

        for a in self.agents[::-1]:
            if a.energy <= 0:
                a.die()

        # Food respawns randomly with a certain probability
        if random() < self.config['food_respawn_prob']:
            self.agents.append(
                Agent("Food", randint(0, self.width), randint(0, self.height), 0, 100, 0, 0, self))

        self.current_epoch += 1
        self.__update_history()

    def check_collision(self) -> None:
        """
        Checks if any 2 agents are in the same position (or very close) and updates their energy accordingly
        """
        for agent1 in self.get_alive_agents():
            for agent2 in self.get_alive_agents():
                if agent1 == agent2:
                    continue
                # if agent1.energy <= 0 or agent2.energy <= 0:
                #     continue

                if ((agent1.x - agent2.x) ** 2 + (agent1.y - agent2.y) ** 2) < 1:
                    # Both Preys and predators can eat food, but Preys get more energy from it. Predators eat food only if they are at low energy

                    # Prey eats food
                    if (agent1.type == "Food" and agent2.type == "Prey") or (agent1.type == "Prey" and agent2.type == "Food"):
                        agent2.energy = min(agent2.energy + 20, 100)
                        if agent1.type == "Food":
                            agent1.die()
                        else:
                            agent2.die()

                    # Predator eats food
                    elif (agent1.type == "Food" and agent2.type == "Predator" and agent2.energy < 25) or (agent1.type == "Predator" and agent2.type == "Food" and agent1.energy < 25):
                        if agent1.type == "Food":
                            agent1.die()
                            agent2.energy = min(agent2.energy + 2, 100)
                        else:
                            agent1.energy = min(agent1.energy + 2, 100)
                            agent2.die()

                    # Prey eats predator
                    elif (agent1.type == "Prey" and agent2.type == "Predator"):
                        agent2.energy = min(agent2.energy + 10, 100)
                        agent1.die()
                    elif (agent1.type == "Predator" and agent2.type == "Prey"):
                        agent1.energy = min(agent1.energy + 10, 100)
                        agent2.die()
                    else:
                        pass

    def __update_history(self) -> None:
        """
        Updates the history of the simulation

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.history["Epoch"].append(self.current_epoch)
        self.history["Food"].append(
            len([agent for agent in self.agents if agent.type == "Food"]))
        self.history["Prey"].append(
            len([agent for agent in self.agents if agent.type == "Prey"]))
        self.history["Predator"].append(
            len([agent for agent in self.agents if agent.type == "Predator"]))

    def log(self, msg: str) -> None:
        """
        Adds a message to the log. These are displayed by the app

        Parameters
        ----------
        msg: str - the message to log

        Returns
        -------
        None
        """

        self.log_messages.append(msg)

    def __str__(self):
        return f"Simulation with {len(self.agents)} agents in a {self.width}x{self.height} grid"
