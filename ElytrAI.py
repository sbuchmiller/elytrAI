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


class elytraFlyer(gym.Env):

    def __init__(self, env_config):
        self.num_observations = 7
        self.log_frequency = 1
        self.move_mult = 50
        self.distance_reward_gamma = 0.02
        self.velocity_reward_gamma = 0.1
        self.damage_taken_reward_gamma = 10
        self.pillar_frequency = 0.005

        # RLlib params
        self.action_space = Box(-2, 2, shape=(2,), dtype=np.float32)
        self.observation_space = Box(-10000, 10000, shape=(self.num_observations,), dtype=np.float32)

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
        self.damage_taken = 0

        self.episode_step = 0
        self.episode_return = 0
        self.returns = []
        self.steps = []
        self.episodes = []
        self.flightDistances = []

        # Set NP to print decimal numbers rather than scientific notation.
        np.set_printoptions(suppress=True)

    def reset(self):
        """
        Clear all per-episode variables and reset world for next episode

        Returns
            observation: <np.array> [X, Y, Z, xVelo, yVelo, zVelo, BlockSightDistance]
        """
        # resets malmo world to xml file
        world_state = self.init_malmo()

        # Append return value and flight distance
        self.returns.append(self.episode_return)
        self.flightDistances.append(self.lastz)

        # Get the value of the current step
        currentStep = self.steps[-1] if len(self.steps) > 0 else 0
        self.steps.append(currentStep + self.episode_step)

        # Get the value of the current episode and append to the episodes list
        if len(self.episodes) > 0:
            currentEpisode = self.episodes[-1] + 1
        else:
            currentEpisode = 0
        self.episodes.append(currentEpisode)

        # Reset values of the run
        self.episode_return = 0
        self.episode_step = 0
        self.clearObservationVariables()

        # Log (Right now I have it logging after every flight
        # if len(self.returns) > self.log_frequency + 1 and \
        #         len(self.returns) % self.log_frequency == 0:
        self.log_returns()

        # Get Observations
        self.obs = self.get_observation(world_state)

        return self.obs

    def step(self, action):
        """
        Take an action in the environment and return the results.

        Args
            action: <box> 2x1 box defining action - X and Y for where to move mouse.

        Returns
            observation: <np.array> [X, Y, Z, xVelo, yVelo, zVelo, BlockSightDistance] location
            reward: <float> reward from taking action
            done: <bool> indicates terminal state
            info: <dict> dictionary of extra information
        """
        # print(f"step() - action = {action}")

        # Send command to move mouse
        self.agent_host.sendCommand(f"moveMouse {int(action[0] * self.move_mult)} {int(action[1] * self.move_mult)}")

        # Sleep and increment the episode by one
        time.sleep(.1)
        self.episode_step += 1

        # Take observation
        world_state = self.agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
        self.obs = self.get_observation(world_state)

        # Check if mission ended
        done = not world_state.is_mission_running

        # Get Reward
        reward = 0
        # Get's rewards defined in XML (We have none right now)
        for r in world_state.rewards:
            reward += r.getValue()

        # Reward for going far in the Z direction
        reward += self.obs[2] * self.distance_reward_gamma
        reward += self.obs[5] * self.velocity_reward_gamma

        # Punish for hitting a pillar in midflight
        if self.obs[1] > 3:
            reward -= self.damage_taken * self.damage_taken_reward_gamma

        # add reward for this step to the episode return value.
        self.episode_return += reward

        return self.obs, reward, done, dict()

    def getPillarLocations(self, width=300, length=1000):
        return_string = ""
        for x in range(-1 * int(width/2), int(width/2)):
            for z in range(length):
                if randint(1/self.pillar_frequency) == 1:
                    return_string += f"<DrawLine x1='{x}' y1='2' z1='{z}' x2 = '{x}' y2 = '100' z2 = '{z}' type='diamond_block'/>\n"
        return return_string


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
                        <FlatWorldGenerator generatorString="2;7,11;1;"/>
                        <DrawingDecorator>
                            <DrawBlock x="0" y="60" z="0" type="lapis_block"/>
                            <DrawCuboid x1="-150" y1="2" z1="1" x2="150" y2="100" z2="1000" type="air"/>
                            ''' + \
                                self.getPillarLocations() + '''
                            <DrawCuboid x1="-20" y1="2" z1="1" x2="-10" y2="100" z2="30" type="air"/>
                        </DrawingDecorator>
                        <ServerQuitWhenAnyAgentFinishes/>
                    </ServerHandlers>
                </ServerSection>

                <AgentSection mode="Survival">
                    <Name>elytrAI</Name>
                    <AgentStart>
                        <Placement x="0.5" y="61" z="0.5" yaw="0"/>
                        <Inventory>
                            <InventoryItem slot="38" type="elytra"/>
                        </Inventory>
                    </AgentStart>
                    <AgentHandlers>
                        <HumanLevelCommands/>
                        <ObservationFromFullStats/>
                        <ObservationFromRay/>
                        <AgentQuitFromTimeUp timeLimitMs="180000"/>
                    </AgentHandlers>
                </AgentSection>
                </Mission>
                '''

    def log_returns(self):
        """
        Log the current returns as a graph and text file
        """

        # Log the reward scores.
        try:
            # Create graph
            box = np.ones(self.log_frequency) / self.log_frequency
            returns_smooth = np.convolve(self.returns[1:], box, mode='same')
            plt.clf()
            plt.plot(self.steps[1:], returns_smooth)
            plt.title('Elytrai Flight Rewards')
            plt.ylabel('Return')
            plt.xlabel('Steps')
            plt.savefig('outputs/returns.png')

            # Write to TXT file
            with open('outputs/returns.txt', 'w') as f:
                for step, value in zip(self.steps[1:], self.returns[1:]):
                    f.write("{}\t{}\n".format(step, value))
        except:
            print("unable to log reward results")

        # Log the flight distances
        try:
            # Create graph
            box = np.ones(self.log_frequency) / self.log_frequency
            returns_smooth = np.convolve(self.flightDistances[1:], box, mode='same')
            plt.clf()
            plt.plot(self.episodes[1:], returns_smooth)
            plt.title('Elytrai Flight Rewards')
            plt.ylabel('Distance')
            plt.xlabel('Episodes')
            plt.savefig('outputs/DistanceFlown.png')

            # Write to TXT file
            with open('outputs/DistanceFlown.txt', 'w') as f:
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
                self.agent_host.startMission(my_mission, my_clients, my_mission_record, 0, 'elytraFlyer')
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission", e)
                    exit(1)
                else:
                    time.sleep(2)

        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)

        self.agentJumpOffStartingBlock()
        return world_state

    def get_observation(self, world_state):
        """
        Use the agent observation API to get the current X, Y, and Z values of the agent

        Args
            world_state: <object> current agent world state

        Returns
            observation: <np.array> the state observation [X, Y, Z, xVelo, yVelo, zVelo]
        """
        obs = np.zeros((self.num_observations,))  # Initialize zero'd obs return

        while world_state.is_mission_running:
            world_state = self.agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                # Get observation json
                msg = world_state.observations[-1].text
                jsonLoad = json.loads(msg)

                self.damage_taken = 20 - jsonLoad['Life']

                # Get the distance of the block at the center of screen. -1 if no block there
                try:
                    blockInSightDistance = jsonLoad['LineOfSight']['distance']
                except:
                    blockInSightDistance = -1

                # Get the X, Y, and Z positions of the agent
                xPos = jsonLoad['XPos']
                yPos = jsonLoad['YPos']
                zPos = jsonLoad['ZPos']


                # calculate velocities
                xVelocity = xPos - self.lastx
                yVelocity = yPos - self.lasty
                zVelocity = zPos - self.lastz

                # update the self.last values
                self.lastx = xPos
                self.lasty = yPos
                self.lastz = zPos

                # Create obs np array and return
                obsList = [xPos, yPos, zPos, xVelocity, yVelocity, zVelocity, blockInSightDistance]
                obs = np.array(obsList)
                break

        return obs

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
        self.damage_taken = 0

    def agentJumpOffStartingBlock(self):
        """
        Tells the agent to jump off the starting platform and open the elytra
        """
        self.agent_host.sendCommand("forward 1")
        time.sleep(.15)
        self.agent_host.sendCommand("jump 1")
        time.sleep(.1)
        self.agent_host.sendCommand("jump 0")
        self.agent_host.sendCommand("forward 0")
        time.sleep(1)
        self.agent_host.sendCommand("jump 1")
        time.sleep(.1)
        self.agent_host.sendCommand("jump 0")
        time.sleep(.1)


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
