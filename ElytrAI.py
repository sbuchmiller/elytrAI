#CS175 Fall 2021 Project
#Creators: Alec Grogan-Crane, Alexandria Meng, Scott Buchmiller


try:
    from malmo import MalmoPython
except:
    import MalmoPython

import os
import sys
import time
import json
from priority_dict import priorityDictionary as PQ
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
import gym, ray
from gym.spaces import Discrete, Box
from ray.rllib.agents import a2c

# sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
class elytraFlyer(gym.env):

    def __init__(self,degreeChange=5):
        self.numObservations = 6
        self.actionList = self.getPotentialActions(degreeChange)

        #RLlib params
        self.action_space = Discrete(len(self.action_dict))
        self.observation_space = Box(0,360, shape=(self.numObservations,), dtype=np.float32)


        #agent parameters
        self.lastx = 0
        self.lasty = 0
        self.lastz = 0
        self.xvelocity = 0
        self.yvelocity = 0
        self.zvelocity = 0


        self.episodeStep = 0
        self.episodeReturn = 0
        self.returns = []
        self.steps = []
        self.episodes = []
        self.flightDistances = []


    def reset(self):
        '''
            clear all per-episode variables and reset world for next episode
        '''
        #resets malmo world to xml file
        world_state = self.init_malmo()

        #logs episode number and returns
        self.returns.append(self.episode_return)
        currentStep = self.steps[-1] if len(self.steps) > 0 else 0
        self.steps.append(currentStep + self.episodeStep)
        self.episodeReturn = 0
        self.episodeStep = 0
        self.clearObservationVariables()


    def step(self, action):

        #take observation

        #choose action

        #get reward

        #check if mission ended


        return ()


    def GetMissionXML(seed, gp, size=10):
        return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

                    <About>
                        <Summary>Hello world!</Summary>
                    </About>

                <ServerSection>
                    <ServerInitialConditions>
                        <Time>
                            <StartTime>1000</StartTime>
                            <AllowPassageOfTime>false</AllowPassageOfTime>
                        </Time>
                        <Weather>clear</Weather>
                    </ServerInitialConditions>
                    <ServerHandlers>
                        <FlatWorldGenerator generatorString="3;7,44*49,73,35:1,159:4,95:13,35:13,159:11,95:10,159:14,159:6,35:6,95:6;12;"/>
                        <DrawingDecorator>
                                <DrawBlock x="0" y="100" z="0" type="lapis_block"/>
                        </DrawingDecorator>
                        <ServerQuitFromTimeUp timeLimitMs="10000"/>
                        <ServerQuitWhenAnyAgentFinishes/>
                    </ServerHandlers>
                </ServerSection>

                <AgentSection mode="Survival">
                    <Name>elytrAI</Name>
                    <AgentStart>
                        <Placement x="0.5" y="101" z="0.5" yaw="0"/>
                        <Inventory>
                            <InventoryItem slot="38" type="elytra"/>
                        </Inventory>
                    </AgentStart>
                    <AgentHandlers>
                        <HumanLevelCommands/>
                        <AbsoluteMovementCommands/>
                        <InventoryCommands/>
                        <ObservationFromGrid>
                            <Grid name="floorAll">
                                <min x="-10" y="-1" z="-10"/>
                                <max x="10" y="-1" z="10"/>
                            </Grid>
                        </ObservationFromGrid>
                    </AgentHandlers>
                </AgentSection>
                </Mission>'''

    def load_grid(world_state):
        """
        Used the agent observation API to get a 21 X 21 grid box around the agent (the agent is in the middle).

        Args
            world_state:    <object>    current agent world state

        Returns
            grid:   <list>  the world grid blocks represented as a list of blocks (see Tutorial.pdf)
        """
        while world_state.is_mission_running:
            #sys.stdout.write(".")
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            if len(world_state.errors) > 0:
                raise AssertionError('Could not load grid.')

            if world_state.number_of_observations_since_last_state > 0:
                msg = world_state.observations[-1].text
                observations = json.loads(msg)
                grid = observations.get(u'floorAll', 0)
                break
        return grid

    def log_returns(self):
        """
        Log the current returns as a graph and text file

        Args:
            steps (list): list of global steps after each episode
            returns (list): list of total return of each episode
        """
        try:
            box = np.ones(self.log_frequency) / self.log_frequency
            returns_smooth = np.convolve(self.returns[1:], box, mode='same')
            plt.clf()
            plt.plot(self.steps[1:], returns_smooth)
            plt.title('Elytrai Flight Rewards')
            plt.ylabel('Return')
            plt.xlabel('Steps')
            plt.savefig('returns.png')
            with open('returns.txt', 'w') as f:
                for step, value in zip(self.steps[1:], self.returns[1:]):
                    f.write("{}\t{}\n".format(step, value)) 
        except:
            print("unable to log reward results")

        try:
            box = np.ones(self.log_frequency) / self.log_frequency
            returns_smooth = np.convolve(self.flightDistances[1:], box, mode='same')
            plt.clf()
            plt.plot(self.episodes[1:], returns_smooth)
            plt.title('Elytrai Flight Rewards')
            plt.ylabel('Distance')
            plt.xlabel('Episodes')
            plt.savefig('DistanceFlown.png')
            with open('DistanceFlown.txt', 'w') as f:
                for step, value in zip(self.episodes[1:], self.flightDistances[1:]):
                    f.write("{}\t{}\n".format(step, value)) 
        except:
            print("unable to log flight distance results")

    def getPotentialActions(self, degreeChange = 5):
        '''
            degreeChange: how many degrees we are allowed to change per angle, smaller degree = more angles
            return: [(str,str)] of camera angles that the agent will be able to pick from as a discrete actions space
        '''
        out = []
        for i in range(0,360,degreeChange):
            for j in range(0,360,degreeChange):
                out.append[(f"setYaw {i}",f"setPitch {j}")]
        return out

    def clearObservationVariables(self):
        '''
        sets all temp observation variables(lastx, lasty, lastz, x,y,z velocity) to 0
        '''
        self.lastx = 0
        self.lasty = 0
        self.lastz = 0
        self.xvelocity = 0
        self.yvelocity = 0
        self.zvelocity = 0



# Create default Malmo objects:
agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

if agent_host.receivedArgument("test"):
    num_repeats = 1
else:
    num_repeats = 10

for i in range(num_repeats):
    size = int(6 + 0.5 * i)
    print("Size of maze:", size)
    my_mission = MalmoPython.MissionSpec(GetMissionXML("0", 0.4 + float(i / 20.0), size), True)
    my_mission_record = MalmoPython.MissionRecordSpec()
    my_mission.requestVideo(800, 500)
    my_mission.setViewpoint(1)
    # Attempt to start a mission:
    max_retries = 3
    my_clients = MalmoPython.ClientPool()
    my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_clients, my_mission_record, 0, "%s-%d" % ('Moshe', i) )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission", (i+1), ":",e)
                exit(1)
            else:
                time.sleep(2)

    # Loop until mission starts:
    print("Waiting for the mission", (i+1), "to start ",)
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        #sys.stdout.write(".")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)

    print()
    print("Mission", (i+1), "running.")

    grid = load_grid(world_state)

    #put calls to other functions here


    # Action loop:
    action_index = 0
    initial_jump_complete = False
    while world_state.is_mission_running:

        # Tells agent to jump forward off starter platform
        if not initial_jump_complete:
            agent_host.sendCommand("forward 1")
            time.sleep(.1)
            agent_host.sendCommand("jump 1")
            time.sleep(.1)
            agent_host.sendCommand("jump 0")
            agent_host.sendCommand("forward 0")
            time.sleep(0.3)
            agent_host.sendCommand("jump 1")
            time.sleep(.1)
            agent_host.sendCommand("jump 0")
            initial_jump_complete = True

        # Some example Yaw commands
        agent_host.sendCommand("setYaw 0")
        time.sleep(0.3)
        agent_host.sendCommand("setYaw -50")
        time.sleep(0.3)
        agent_host.sendCommand("setYaw 60")

        """
        Stuff below this is probably not needed anymore but I don't want to delete yet 
        just in case lol.
        """
        # Sending the next commend from the action list -- found using the Dijkstra algo.
        action_list = []
        if action_index >= len(action_list):
            print("Error:", "out of actions, but mission has not ended!")
            time.sleep(2)
        else:
            agent_host.sendCommand(action_list[action_index])
        action_index += 1
        if len(action_list) == action_index:
            # Need to wait few seconds to let the world state realise I'm in end block.
            # Another option could be just to add no move actions -- I thought sleep is more elegant.
            time.sleep(2)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)

    print()
    print("Mission", (i+1), "ended")
    # Mission has ended.