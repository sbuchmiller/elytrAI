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
#from ray.rllib.agents.a3c import a2c


class elytraFlyer(gym.Env):

    def __init__(self, env_config, log_frequency = 1, move_mult = 50, ):
        self.num_observations = 10
        self.log_frequency = 1
        self.move_mult = 50
        self.distance_reward_gamma = 0.02
        self.velocity_reward_gamma = 0.1
        self.damage_taken_reward_gamma = 10
        self.pillar_frequency = 0.005
        self.pillar_touch_punishment = 0

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

        if env_config == {}:
            self.episode_step = 0
            self.episode_return = 0
            self.returns = []
            self.steps = []
            self.episodes = []
            self.flightDistances = []
            self.damageTakenPercentLast20Episodes = []
            self.damageFromLast20 = [0]*20 #array of 20 zeroes
        else:
            self.episode_step = env_config["episode_step"]
            self.episode_return = env_config["episode_return"]
            self.returns = env_config["returns"]
            self.steps = env_config["steps"]
            self.episodes = env_config["episodes"]
            self.flightDistances = env_config["flightDistances"]
            self.damageTakenPercentLast20Episodes = env_config["damageTakenPercentLast20Episodes"]
            self.damageFromLast20 = env_config["damageFromLast20"]


        # Set NP to print decimal numbers rather than scientific notation.
        np.set_printoptions(suppress=True)

        self.ranOnce = False

    def reset(self):
        """
        Clear all per-episode variables and reset world for next episode

        Returns
            observation: <np.array> [xPos, yPos, zPos, xVelocity, yVelocity, zVelocity,
                                                            blockInSightDistance, blockInSightX, blockInSightY,
                                                            blockInSightZ]
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
        self.damageTakenPercentLast20Episodes.append(sum(self.damageFromLast20)/20)

        # Log (Right now I have it logging after every flight
        # if len(self.returns) > self.log_frequency + 1 and \
        #         len(self.returns) % self.log_frequency == 0:
        self.log_returns()

         # Reset values of the run
        self.episode_return = 0
        self.episode_step = 0
        self.clearObservationVariables()

        # Get Observations
        self.obs = self.get_observation(world_state)

        return self.obs

    def step(self, action):
        """
        Take an action in the environment and return the results.

        Args
            action: <box> 2x1 box defining action - X and Y for where to move mouse.

        Returns
            observation: <np.array> [xPos, yPos, zPos, xVelocity, yVelocity, zVelocity,
                                                            blockInSightDistance, blockInSightX, blockInSightY,
                                                            blockInSightZ]
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
        # Get's rewards defined in XML
        for r in world_state.rewards:
            if r.getValue() != 0:
                # print(f"step() - Punish for touching diamond_block = {r.getValue()}")
                reward += r.getValue()

        #get additional rewards

            # Reward for going far in the Z direction
        reward += self.obs[2] * self.distance_reward_gamma


        # Punish for hitting a pillar in midflight
        """
        if self.obs[1] > 3:
            punishment = self.damage_taken * self.damage_taken_reward_gamma
            if punishment != 0:
                reward -= punishment
                # print(f"step() - Punishment for taking damage = -{punishment}")
        """

        # add reward for this step to the episode return value.
        self.episode_return += reward
        #print(f"step() - Episode Return so far = {self.episode_return}")
        return self.obs, reward, done, dict()

    def getPillarLocations(self, width=300, length=1000):
        return_string = ""
        for x in range(-1 * int(width/2), int(width/2)):
            for z in range(30, length):
                if randint(1/self.pillar_frequency) == 1:
                    if x >= 15 or x <= -15:
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
                            <DrawCuboid x1="-150" y1="2" z1="0" x2="150" y2="100" z2="1000" type="air"/>
                            ''' + \
                                self.getPillarLocations() + '''
                            <DrawCuboid x1="-20" y1="2" z1="1" x2="-10" y2="100" z2="30" type="air"/>
                            <DrawBlock x="0" y="60" z="0" type="lapis_block"/>
                        </DrawingDecorator>
                        <ServerQuitWhenAnyAgentFinishes/>
                        <ServerQuitFromTimeUp timeLimitMs="45000"/>
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
                        <RewardForTouchingBlockType>
                            ''' + \
                            f'<Block reward="{self.pillar_touch_punishment}" type="diamond_block"/>' + ''' 
                        </RewardForTouchingBlockType>  
                    </AgentHandlers>
                </AgentSection>
                </Mission>
                '''

    def log_returns(self):
        """
        Log the current returns as a graph and text file
        """
        self.log_returns_as_text()
        self.log_returns_as_graph()
        

    def log_returns_as_text(self):
        #log damage taken % in last 20 episodes
        try:
            with open('outputs/DamagePercent.txt', 'w') as f:
                for step, value in zip(self.episodes[1:], self.damageTakenPercentLast20Episodes[1:]):
                    f.write("{}\t{}\n".format(step, value))
        except Exception as e:
            print("unable to log damage taken Percent results in text")
            print(e)

        #log flight distances
        try:
            with open('outputs/DistanceFlown.txt', 'w') as f:
                for step, value in zip(self.episodes[1:], self.flightDistances[1:]):
                    f.write("{}\t{}\n".format(step, value))
        except Exception as e:
            print("unable to log flight distances in text")
            print(e)

        #log rewards per step
        try:
            with open('outputs/returns.txt', 'w') as f:
                for step, value in zip(self.steps[1:], self.returns[1:]):
                    f.write("{}\t{}\n".format(step, value))
        except Exception as e:
            print("unable to log rewards as text")
            print(e)

    def log_returns_as_graph(self):
        #log damage taken % from last 20 flights
        try:
            #box = np.ones(self.log_frequency)/ self.log_frequency
            #returns_smooth = np.convolve(self.episodes[1:], box, mode='same')
            plt.clf()
            plt.plot(self.episodes[1:], self.damageTakenPercentLast20Episodes[1:])
            plt.title('Percent of episodes with damage taken in last 20 episodes')
            plt.ylabel('Damage Percent')
            plt.xlabel('Episodes')
            plt.savefig('outputs/DamagePercent.png')
            # Write to TXT file
        except Exception as e:
            print("unable to log damage taken Percent results")
            print(e)

        # Log the flight distances
        try:
            # Create graph
            plt.clf()
            plt.plot(self.episodes[1:], self.flightDistances[1:])
            plt.title('Elytrai Distance Flown in Z direction')
            plt.ylabel('Distance')
            plt.xlabel('Episodes')
            plt.savefig('outputs/DistanceFlown.png')

            # Write to TXT file
        except Exception as e:
            print("unable to log flight distance results")
            print(e)

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


        except Exception as e:
            print("unable to log reward results")
            print(e)

    def close(self):
        print("Where would you like to save files")
        print(input())

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
            observation: <np.array> the state observation [xPos, yPos, zPos, xVelocity, yVelocity, zVelocity,
                                                            blockInSightDistance, blockInSightX, blockInSightY,
                                                            blockInSightZ]
        """
        obs = np.zeros((self.num_observations,))  # Initialize zero'd obs return

        while world_state.is_mission_running:
            world_state = self.agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                # Get observation json
                msg = world_state.observations[-1].text
                jsonLoad = json.loads(msg)
                

                # Get the distance of the block at the center of screen. -1 if no block there
                try:
                    blockType = jsonLoad['LineOfSight']['type']
                    if blockType == "diamond_block":
                        blockInSightX = jsonLoad['LineOfSight']['x']
                        blockInSightY = jsonLoad['LineOfSight']['x']
                        blockInSightZ = jsonLoad['LineOfSight']['x']
                        blockInSightDistance = jsonLoad['LineOfSight']['distance']
                    else:
                        blockInSightDistance = 10000
                        blockInSightX = 10000
                        blockInSightY = 10000
                        blockInSightZ = 10000

                except:
                    blockInSightDistance = 10000
                    blockInSightX = 10000
                    blockInSightY = 10000
                    blockInSightZ = 10000

                # Get the X, Y, and Z positions of the agent
                xPos = jsonLoad['XPos']
                yPos = jsonLoad['YPos']
                zPos = jsonLoad['ZPos']

                # determine if damage was taken from hitting a pillar
                if yPos > 3:
                    self.damage_taken = 20 - jsonLoad['Life']
                    if self.damage_taken > 0:
                        self.damageFromLast20[0] = 1

                # calculate velocities
                xVelocity = xPos - self.lastx
                yVelocity = yPos - self.lasty
                zVelocity = zPos - self.lastz

                # update the self.last values
                self.lastx = xPos
                self.lasty = yPos
                self.lastz = zPos

                # Create obs np array and return
                obsList = [xPos, yPos, zPos, xVelocity, yVelocity, zVelocity, blockInSightDistance, blockInSightX, blockInSightY, blockInSightZ]
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

        if len(self.damageFromLast20) >= 20:
            self.damageFromLast20 = self.damageFromLast20[:-1]
            self.damageFromLast20.insert(0,0)

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

    def saveDataAsJson(self,location,fileName = "envVariables.json"):
        envDict = {}
        envDict["episode_step"] = self.episode_step
        envDict["episode_return"] = self.episode_return
        envDict["returns"] = self.returns 
        envDict["steps"] = self.steps
        envDict["episodes"] = self.episodes
        envDict["flightDistances"] = self.flightDistances
        envDict["damageTakenPercentLast20Episodes"] = self.damageTakenPercentLast20Episodes
        envDict["damageFromLast20"] = self.damageFromLast20
        try:
            with open(location + "\\" +fileName, 'w+') as f:
                json.dump(envDict,f)
        except Exception as e:
            print("unable to save env as json")
            print(e)
            print(e.__traceback__)



if __name__ == '__main__':
    loadPath = ''
    if len(sys.argv) > 1:
        if sys.argv[1] == '-l':
            print("loading file from path", sys.argv[2])
            loadPath = sys.argv[2]
            sys.argv = [sys.argv[0]]

    ray.init()
    stepsPerCheckpoint = 2500 #change this to have more or less frequent saves
    config = ppo.DEFAULT_CONFIG.copy()
    config['framework'] = 'torch'
    config['num_gpus'] = 0
    config['num_workers'] = 0
    config['train_batch_size'] = stepsPerCheckpoint 
    config['rollout_fragment_length'] = stepsPerCheckpoint
    config['sgd_minibatch_size'] = stepsPerCheckpoint
    config['batch_mode'] = 'complete_episodes'

    if loadPath != '':
        jsonFilePath = loadPath.split("\\")[:-1]
        jsonFilePath.append("envVariables.json")
        jsonFilePath = "\\".join(jsonFilePath)
        try:
            with open(jsonFilePath, 'r') as f:
                config['env_config'] = json.load(f)
        except Exception as e:
            print("could not read json file, creating new environment")
            config['env_config'] = {}
    else:
        config['env_config'] = {}
    trainer = ppo.PPOTrainer(env=elytraFlyer, config=config)
    if loadPath != '':
        trainer.restore(r""+loadPath)
    

    while True:
        a = trainer.train()
        saveLocation = trainer.save()
        print("Checkpoint saved, Save Location is:",saveLocation)
        jsonFileName = "envVariables.json"
        folderLocation = saveLocation.split('\\')[:-1]
        folderLocation = "\\".join(folderLocation)
        trainer.workers.local_worker().env.saveDataAsJson(folderLocation)
        

        