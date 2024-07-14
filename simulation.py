import NN
from random import randint, random
from math import pi, cos, sin

# Define agent types
AGENT_TYPES = ["Food", "Pray", "Predator"]

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

        self.nn = NN.createNeuralNetwork(2, [4, 2], 1, NN.relu, 0.5)

    def __str__(self):
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
        self.width = width
        self.height = height
        self.current_epoch = 0
        self.agents = []
        self.agent_ids = {type: 0 for type in AGENT_TYPES}
        
        self.history = {
            "Epoch": [0],
            "Food": [num_agents[0]],
            "Pray": [num_agents[1]],
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
        
        # Update the position of each agent
        for agent in self.agents:
            if agent.type == "Food":
                continue # Food doesn't move
            
            agent.angle += agent.angular_speed % (2 * pi)
            agent.x += agent.speed * cos(agent.angle)
            agent.y += agent.speed * sin(agent.angle)
            agent.x %= self.width
            agent.y %= self.height
            
            agent.energy -= 1
            if agent.energy <= 0:
                self.agents.remove(agent)
            
        # Check if any 2 agents are in the same position
        self.check_collision()
                    
        for a in self.agents[::-1]:
            if a.energy <= 0:
                self.agents.remove(a)
                            
        self.current_epoch += 1
        self.__update_history()

    def check_collision(self) -> None:
        """
        Checks if any 2 agents are in the same position and updates their energy accordingly
        """
        for agent1 in self.agents:
            for agent2 in self.agents:
                if agent1 == agent2:
                    continue
                if agent1.energy <= 0 or agent2.energy <= 0:
                    continue

                if ((agent1.x - agent2.x) ** 2 + (agent1.y - agent2.y) ** 2) < 1:
                    # Both prays and predators can eat food, but prays get more energy from it
                    if (agent1.type == "Food" and agent2.type == "Pray") or (agent1.type == "Pray" and agent2.type == "Food"):                        
                        agent2.energy = min(agent2.energy + 10, 100)
                        if agent1.type == "Food":
                            agent1.energy = 0
                        else:
                            agent2.energy = 0
                    elif (agent1.type == "Food" and agent2.type == "Predator") or (agent1.type == "Predator" and agent2.type == "Food"):                        
                        agent2.energy = min(agent2.energy + 2, 100)
                        if agent1.type == "Food":
                            agent1.energy = 0
                        else:
                            agent2.energy = 0
                    elif (agent1.type == "Pray" and agent2.type == "Predator"):
                        agent2.energy = min(agent2.energy + 10, 100)
                        agent1.energy = 0
                    elif (agent1.type == "Predator" and agent2.type == "Pray"):
                        agent1.energy = min(agent1.energy + 10, 100)
                        agent2.energy = 0
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
        self.history["Food"].append(len([agent for agent in self.agents if agent.type == "Food"]))
        self.history["Pray"].append(len([agent for agent in self.agents if agent.type == "Pray"]))
        self.history["Predator"].append(len([agent for agent in self.agents if agent.type == "Predator"]))
                        
    def __str__(self):
        return f"Simulation with {len(self.agents)} agents in a {self.width}x{self.height} grid"
