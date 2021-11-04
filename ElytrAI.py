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

    def __init__(self, env_config, log_frequency=1, move_mult=50, ):
        self.num_player_observations = 6
        self.log_frequency = 1
        self.move_mult = 50
        self.distance_reward_gamma = 0.02
        self.velocity_reward_gamma = 0.1
        self.damage_taken_reward_gamma = 10
        self.pillar_frequency = 0.005
        self.pillar_touch_punishment = 0

        self.vision_width = 15
        self.vision_distance = 60
        self.vision_height = 30
        self.pillarEffectRadius = 20
        self.num_vision_observations = ((self.vision_width * 2) + 1) * (self.vision_distance + 1)
        self.num_observations = self.num_player_observations + self.num_vision_observations

        # RLlib params
        self.action_space = Box(-1, 1, shape=(2,), dtype=np.float32)
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
        self.damageTakenPercentLast20Episodes = []
        self.damageFromLast20 = [0]*20  # array of 20 zeroes

        # Set NP to print decimal numbers rather than scientific notation.
        np.set_printoptions(suppress=True)

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
        # Get's rewards defined in XML (We have none right now)
        for r in world_state.rewards:
            if r.getValue() != 0:
                # print(f"step() - Punish for touching diamond_block = {r.getValue()}")
                reward += r.getValue()

        # Reward for going far in the Z direction
        reward += self.lastz * self.distance_reward_gamma

        # Create gradient reward decrease around the poles. Less reward the closer steve is to the poles.
        steve_location_index = 6 + (self.vision_width * 2 + 1) + self.vision_width

        reward *= self.obs[steve_location_index]
        # print(f"step() - Step reward reduction multiplier = {self.obs[steve_location_index]}")
        print(f"step() - Step reward = {reward}")


        # add reward for this step to the episode return value.
        self.episode_return += reward

        return self.obs, reward, done, dict()

    def getPillarLocations(self, width=300, length=1000):
        return_string = ""
        for x in range(-1 * int(width/2), int(width/2)):
            for z in range(30, length):
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
                        <ObservationFromGrid>
                            <Grid name="floorAll">
                                <min x="-'''+str(self.vision_width)+'''" y="0" z="0"/>
                                <max x="'''+str(self.vision_width)+'''" y="0" z="'''+str(self.vision_distance)+'''"/>
                            </Grid>
                        </ObservationFromGrid>
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
            #box = np.ones(self.log_frequency) / self.log_frequency
            #returns_smooth = np.convolve(self.flightDistances[1:], box, mode='same')
            plt.clf()
            plt.plot(self.episodes[1:], self.flightDistances[1:])
            plt.title('Elytrai Distance Flown in Z direction')
            plt.ylabel('Distance')
            plt.xlabel('Episodes')
            plt.savefig('outputs/DistanceFlown.png')

            # Write to TXT file
            with open('outputs/DistanceFlown.txt', 'w') as f:
                for step, value in zip(self.episodes[1:], self.flightDistances[1:]):
                    f.write("{}\t{}\n".format(step, value))
        except:
            print("unable to log flight distance results")


    def convertFOVlistToNPArray1s0s(self, visionList):
        """
        Takes a list of blocks in within the agents field of view and converts the 'air' blocks to 0
        and the 'diamond_block' blocks to 1

        Args
            visionList: <list> current agent field of view

        Returns
            visionList: <np.array> agent field of view as NP array of 1s and 0s
        """
        outputList = []
        for block in visionList:
            if block == "air":
                outputList.append(0)
            elif block == "diamond_block":
                outputList.append(1)
            else:
                outputList.append(2)

        outputList = np.array(outputList)
        return outputList

    def convertFOVlistToNPArrayReturPenalizers(self, visionList):
        """
        Takes a list of blocks in within the agents field of view and converts it to an array of score decreasers that
        are more penalizing the closer they are to a diamond_block

        Args
            visionList: <list> current agent field of view

        Returns
            visionList: <np.array> agent field of view as NP array of score penalizers.
        """
        total_width = self.vision_width * 2 + 1  # Total width of the agent vision
        reducPerBlockDist = 1 / self.pillarEffectRadius  # how much less the score is reduced each block you get closer to the pillar
        visionListSize = len(visionList)  # size of the vision array
        outputArray = np.ones((len(visionList),))  # Initialize NP array of size visionListSize to all 1s

        # For each block in the item list
        for i in range(visionListSize):
            # If block is a diamond block
            if visionList[i] == 'diamond_block':
                # The multiplier at that exact location should be 0
                outputArray[i] = 0

                # For the self.pillarEffectRadius number of rows before the pillar
                for z in range(-self.pillarEffectRadius, 1):
                    # Calculate the starting reduction value used at the block at row z directly infront of the pillar
                    # This value will be 0 one block infront of the pillar and will slowly increase the further back
                    # from the pillar the agent is.
                    startingReduc = (reducPerBlockDist * abs(z))

                    # Calculate the amount to increase each block to the side for the given row. I.e for row right
                    # behind the pillar, the starting reduction is 0 and the block 1 radius to the left or right should
                    # be 1 and everything in between needs to be smoothly transitioned.
                    reducIncrement = (1 - startingReduc) / self.pillarEffectRadius

                    # Calculate the index of the block that is at the center of row z directly inline with the pillar
                    z_index_center = i + (total_width * z)

                    # For each block in row z to the left and right of center that is within radius distance of
                    # the center.
                    for x in range(self.pillarEffectRadius):

                        # Calculate the reduction multiplier based on how far off of the center block we are.
                        reduction = startingReduc + (reducIncrement * x)

                        # If the index we are looking at is within bounds, apply the reduction
                        if 0 <= z_index_center + x < visionListSize:
                            outputArray[z_index_center + x] = reduction
                        if 0 <= z_index_center - x < visionListSize:
                            outputArray[z_index_center - x] = reduction

        return outputArray



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
            observation: <np.array> the state observation [xPos, yPos, zPos, xVelocity, yVelocity, zVelocity, ....]
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
                obsArray = np.array([xPos, yPos, zPos, xVelocity, yVelocity, zVelocity])

                # Get the blocks around steve
                blockArray = self.convertFOVlistToNPArrayReturPenalizers(jsonLoad['floorAll'])
                obs = np.concatenate((obsArray, blockArray), axis=None)
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
            self.damageFromLast20.insert(0, 0)

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
