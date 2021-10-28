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
import gym
import ray
from gym.spaces import Discrete, Box
from ray.rllib.agents import ppo
# from ray.rllib.agents import a2c


# sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
class elytraFlyer(gym.Env):

    def __init__(self, degreeChange=5):
        self.numObservations = 6
        self.actionDict = self.getPotentialActions()
        self.log_frequency = 1

        # RLlib params
        self.action_space = Discrete(len(self.actionDict))
        self.observation_space = Box(0, 1000, shape=(self.numObservations,), dtype=np.float32)

        # Malmo Parameters
        self.agent_host = MalmoPython.AgentHost()
        try:
            self.agent_host.parse(sys.argv)
        except RuntimeError as e:
            print('ERROR:', e)
            print(self.agent_host.getUsage())
            exit(1)

        # agent parameters
        self.obs = None
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
        """
        Clear all per-episode variables and reset world for next episode
        """
        # resets malmo world to xml file
        world_state = self.init_malmo()

        # logs episode number and returns
        self.returns.append(self.episodeReturn)
        currentStep = self.steps[-1] if len(self.steps) > 0 else 0
        self.steps.append(currentStep + self.episodeStep)
        self.episodeReturn = 0
        self.episodeStep = 0
        self.clearObservationVariables()

        self.obs = self.get_observation(world_state)

        return self.obs

    def step(self, action):
        # choose action
        print(f"Action: {action}")

        # take observation
        world_state = self.agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
        print(world_state)
        self.obs = self.get_observation(world_state)

        # Get Reward
        reward = 0
        for r in world_state.rewards:
            reward += r.getValue()
        self.episodeReturn += reward

        # check if mission ended
        done = not world_state.is_mission_running

        return self.obs, reward, done, dict()


    def GetMissionXML(self):
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
                        <FlatWorldGenerator generatorString="2;57;1;"/>
                        <DrawingDecorator>
                                <DrawBlock x="0" y="100" z="0" type="lapis_block"/>
                        </DrawingDecorator>
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
                        <ObservationFromFullStats/>
                        <ObservationFromRay/>
                        <AgentQuitFromTouchingBlockType>
                            <Block type="diamond_block" />
                        </AgentQuitFromTouchingBlockType>
                        </AgentHandlers>
                </AgentSection>
                </Mission>
                '''

    def log_returns(self):
        """
        Log the current returns as a graph and text file
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

    def init_malmo(self):
        """
        Initialize new malmo mission.
        """
        my_mission = MalmoPython.MissionSpec(self.GetMissionXML(), True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        my_mission.requestVideo(800, 500)
        my_mission.setViewpoint(1)
        # Attempt to start a mission:
        max_retries = 3
        my_clients = MalmoPython.ClientPool()
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000))  # add Minecraft machines here as available

        for retry in range(max_retries):
            try:
                self.agent_host.startMission(my_mission, my_clients, my_mission_record, 0, 'Moshe')
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission", e)
                    exit(1)
                else:
                    time.sleep(2)

        world_state = self.agent_host.getWorldState()
        print(f"init_malmo() - World State: {world_state}")
        while not world_state.has_mission_begun:
            # sys.stdout.write(".")
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)

        self.agentJumpOffStartingBlock()
        return world_state

    def get_observation(self, world_state):
        """

        """
        obs = np.zeros((self.numObservations,))
        while world_state.is_mission_running:
            world_state = self.agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                # First we get the json from the observation API
                msg = world_state.observations[-1].text
                observations = json.loads(msg)
                # print(f"get_observation() - observations: {observations}")

        return obs

    @staticmethod
    def getPotentialActions(degreeChange=5):
        """
            degreeChange: how many degrees we are allowed to change per angle, smaller degree = more angles
            return: {0: (str,str), 1: (str, str)} of camera angles that the agent will be able to pick from
                     as a discrete actions space
        """
        out = dict()
        x = 0
        for i in range(0, 360, degreeChange):
            for j in range(0, 360, degreeChange):
                out[x] = (f"setYaw {i}", f"setPitch {j}")
        return out

    def clearObservationVariables(self):
        """
        sets all temp observation variables(lastx, lasty, lastz, x,y,z velocity) to 0
        """
        self.lastx = 0
        self.lasty = 0
        self.lastz = 0
        self.xvelocity = 0
        self.yvelocity = 0
        self.zvelocity = 0

    def agentJumpOffStartingBlock(self):
        self.agent_host.sendCommand("forward 1")
        time.sleep(.1)
        self.agent_host.sendCommand("jump 1")
        time.sleep(.1)
        self.agent_host.sendCommand("jump 0")
        self.agent_host.sendCommand("forward 0")
        time.sleep(.5)
        print("before jump")
        self.agent_host.sendCommand("jump 1")
        print("after jump")
        time.sleep(.3)
        self.agent_host.sendCommand("jump 0")


if __name__ == '__main__':
    ray.init()
    trainer = ppo.PPOTrainer(env=elytraFlyer, config={
        'env_config': {},           # No environment parameters to configure
        'framework': 'torch',       # Use pyotrch instead of tensorflow
        'num_gpus': 0,              # We aren't using GPUs
        'num_workers': 0            # We aren't using parallelism
    })

    while True:
        print(trainer.train())
