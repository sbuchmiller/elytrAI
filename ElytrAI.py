# CS175 Fall 2021 Project
# Creators: Alec Grogan-Crane, Alexandria Meng, Scott Buchmiller

try:
    from malmo import MalmoPython
except:
    import MalmoPython

import sys
import time
import json
import numpy as np
from numpy.random import randint
import gym
import ray
from gym.spaces import Box, Discrete
import torch
from torch import nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.agents import ppo
from PIL import Image, ImageOps
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf
import math
tf1, tf, tfv = try_import_tf()

# Global constants
image_height = 240
image_width = 320
input_layers = 3
input_number = 15


# Gym Class
class elytraFlyer(gym.Env):

    # ---------RLLIB Initialization, Overriding, and Helper Methods------------
    # Overridden __init__()
    def __init__(self, env_config):
        # Video / Observation Configuration
        self.video_width = image_width
        self.video_height = image_height
        self.num_observations = self.video_width * self.video_height

        # General constants & flags
        self.move_mult = 50
        self.pillar_touch_flag = .5452
        self.step_time_delta = 0.1

        # Reward Gammas & Multipliers
        self.distance_reward_gamma = 0.02
        self.pillar_hit_reward_multiplier = 10

        # Pillar Constants
        self.max_pillar_frequency = 0.005
        self.start_pillar_frequency = 0.0035
        self.pillar_freq_inc_delta = 50

        # RLlib Params
        self.action_dict = {0: "Go Left", 1: "Go Right", 2: "Go Up", 3: "Go Down"}
        self.action_space = Discrete(len(self.action_dict))
        self.observation_space = Box(low = np.array([-360, -360, -100, -100, -100, 0,0,0,0,0,0,0,0,0,0]),\
            high = np.array([360,360,100,100,100,255,255,255,255,255,255,255,255,255,255]),\
            shape=(input_number,), dtype=np.float32)

        # Malmo Params
        self.agent_host = MalmoPython.AgentHost()
        self.obs = None
        try:
            self.agent_host.parse(sys.argv)
        except RuntimeError as e:
            print('ERROR:', e)
            print(self.agent_host.getUsage())
            exit(1)

        # Agent Tracking Params
        self.current_x = 0
        self.current_y = 0
        self.current_z = 0
        self.pillars_touched_in_run = 0

        # Current Run Params & Tracking
        self.current_pillar_frequency = self.start_pillar_frequency
        self.ran_once = False
        if env_config == {}:
            self.episode_step = 0
            self.episode_return = 0
            self.returns = []
            self.steps = []
            self.episodes = []
            self.episode_num = 0
            self.flight_distances = []
            self.pillars_touched = []
        else:
            self.episode_step = env_config["episode_step"]
            self.episode_return = env_config["episode_return"]
            self.returns = env_config["returns"]
            self.steps = env_config["steps"]
            self.episodes = env_config["episodes"]
            self.episode_num = env_config["episode_num"]
            self.flight_distances = env_config["flight_distances"]
            self.pillars_touched = env_config["pillars_touched"]

        # Set NP to print decimal numbers rather than scientific notation.
        np.set_printoptions(suppress=True)

    # Reset run variables
    def reset_run_variables(self):
        """
        sets all temp observation variables to 0
        """
        self.current_x = 0
        self.current_y = 0
        self.current_z = 0
        self.pillars_touched_in_run = 0
        self.episode_return = 0
        self.episode_step = 0

    # Set the pillar frequency according to the episode number
    def set_pillar_frequency(self):
        if self.episode_num // self.pillar_freq_inc_delta == 0:
            self.current_pillar_frequency = self.start_pillar_frequency
        else:
            self.current_pillar_frequency = min(
                self.start_pillar_frequency + (0.0001 * (self.episode_num // self.pillar_freq_inc_delta)),
                self.max_pillar_frequency)

    # Calculate the total distance the agent has traveled from the starting platform.
    def calculate_distance(self):
        return math.sqrt(abs(self.current_z) ** 2 + abs(self.current_x) ** 2)

    # Append the scores from the current run to their appropriate lists
    def append_episode_scores(self):
        self.returns.append(self.episode_return)
        self.flight_distances.append(self.calculate_distance())
        self.pillars_touched.append(self.pillars_touched_in_run)

    # Update the step and episode lists with the current episode and step
    def update_step_and_episode_list(self):
        # Get the value of the current step
        currentStep = self.steps[-1] if len(self.steps) > 0 else 0
        self.steps.append(currentStep + self.episode_step)

        # Get the value of the current episode and append to the episodes list
        self.episodes.append(self.episode_num)

    # Overridden reset()
    def reset(self):
        """
        Clear all per-episode variables and reset world for next episode

        Returns
            observation
        """
        # resets malmo world to xml file
        world_state = self.init_malmo()

        # Append return value, flight distance, and pillars touched
        self.append_episode_scores()

        # Update the lists containing the episode numbers and step numbers.
        self.update_step_and_episode_list()

        # Set the pillar_frequency
        self.set_pillar_frequency()

        # Log
        self.log_returns()

        # Increase the episode by one
        self.episode_num += 1

        # Reset values of the run
        self.reset_run_variables()

        # Get Observations
        self.obs = self.get_observation(world_state)

        return self.obs

    # Helper Method for step() to calculate reward for a given step
    def calculate_reward(self, world_state):
        """
        Reward Calculation Function

        Args
            world_state: The current world state

        Returns
            reward: The reward for the current step

        """
        reward = self.calculate_distance() * self.distance_reward_gamma
        for r in world_state.rewards:
            if r.getValue() == self.pillar_touch_flag:
                reward = -1*(reward * self.pillar_hit_reward_multiplier)
                self.pillars_touched_in_run += 1

        return reward

    # Overriden step()
    def step(self, action):
        """
        Take an action in the environment and return the results.

        Args
            action: <box> 2x1 box defining action - X and Y for where to move mouse.

        Returns
            observation: <np.array> []
            reward: <float> reward from taking action
            done: <bool> indicates terminal state
            info: <dict> dictionary of extra information
        """

        # Send command to move mouse
        if action == 0:
            self.agent_host.sendCommand(f"moveMouse -{self.move_mult} 0")
        elif action == 1:
            self.agent_host.sendCommand(f"moveMouse {self.move_mult} 0")
        elif action == 2:
            self.agent_host.sendCommand(f"moveMouse 0 {self.move_mult}")
        elif action == 3:
            self.agent_host.sendCommand(f"moveMouse 0 -{self.move_mult}")

        # Sleep and increment the episode by one
        time.sleep(self.step_time_delta)
        self.episode_step += 1

        # Get World State
        world_state = self.agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)

        # Check if mission ended
        done = not world_state.is_mission_running

        # Get the observations for this step
        self.obs = self.get_observation(world_state)

        # Determine if a pillar was hit just now and reward for total distance from the starting point
        reward = self.calculate_reward(world_state)
        if reward < 0:
            self.agent_host.sendCommand(f"tp {self.current_x} 1 {self.current_z}")
            time.sleep(self.step_time_delta)

        # Add reward for this step to the episode return value.
        self.episode_return += reward

        return self.obs, reward, done, dict()

    # Draw the observation image using PIL
    @staticmethod
    def draw_image(image):
        image = image.reshape((image_height, image_width, input_layers))
        pil_image = Image.fromarray(image, "RGB")
        pil_image.show()

    # Process the frame from the observation.
    @staticmethod
    def process_frame(frame, numColumns = 10, columnWidth = 20):
        try:
            view = ImageOps.grayscale(Image.frombytes("RGB", (image_width, image_height), np.array(frame)))
            imageColumns = []
            columnStart = (image_width - (numColumns*columnWidth)) / 2
            for i in range(numColumns):
                temp = np.array(view.crop(((columnStart + (i*columnWidth)), 0, (columnStart + ((i+1)*columnWidth)), image_height)))
                accu = 0
                for w in range(0,columnWidth):
                    for h in range(0,image_height):
                        accu += temp[h][w]
                imageColumns.append(accu / (columnWidth * image_height))
        except ValueError:
            print("image failed")
            return [0]*numColumns
        return imageColumns
        

    def process_agent_obs(self, world_state):
        '''
        processes the relevant agent observations from world state and returns as an np array
        '''
        msg = world_state.observations[-1].text
        jsonLoad = json.loads(msg)
        #get the pitch and yaw that the agent is facing
        pitch = jsonLoad['Pitch']
        yaw = jsonLoad['Yaw']
        #get agent velocities
        xvelocity = jsonLoad["XPos"] - self.current_x
        yvelocity = jsonLoad["YPos"] - self.current_y
        zvelocity = jsonLoad["ZPos"] - self.current_z 

        return [pitch, yaw, xvelocity, yvelocity, zvelocity]

    # Update self.current_x, self.current_y, and self.current_z with current values
    def update_agent_position(self, world_state):
        # Get observation json
        msg = world_state.observations[-1].text
        jsonLoad = json.loads(msg)

        # Get the X, Y, and Z positions of the agent
        try:
            self.current_x = jsonLoad['XPos']
            self.current_y = jsonLoad['YPos']
            self.current_z = jsonLoad['ZPos']
        except KeyError:
            self.current_x = 0
            self.current_y = 0
            self.current_z = 0

    # get the current observation for the step
    def get_observation(self, world_state):
        """
        Get the screen that the agent sees

        Args
            world_state: <object> current agent world state

        Returns
            observation: <np.array> []
        """
        obs = np.zeros((input_number,))  # Initialize zero'd obs return
        
        # While the mission is running, wait for a new observation
        while world_state.is_mission_running:
            
            # Once a new observation is found
            if world_state.number_of_observations_since_last_state > 0 and \
               world_state.number_of_video_frames_since_last_state > 0:
                
                # Get video from agent perspective, then preprocess it
                frame_obs = self.process_frame(world_state.video_frames[0].pixels)

                #get additional observations from the agent
                agent_obs = self.process_agent_obs(world_state)
                obs = np.array(agent_obs + frame_obs, dtype=np.float32)
                # Update self.current_x, self.current_y, and self.current_z with current values
                self.update_agent_position(world_state)
                break
                
            # Still waiting for observation from agent so grab the next world_state
            world_state = self.agent_host.getWorldState()
        
        # Return the observation array
        return obs

    # Save the data for the current run as a JSON for later retrieval
    def save_data_as_json(self, location, fileName="envVariables.json"):
        envDict = dict()
        envDict["episode_step"] = self.episode_step
        envDict["episode_return"] = self.episode_return
        envDict["returns"] = self.returns
        envDict["steps"] = self.steps
        envDict["episodes"] = self.episodes
        envDict["episode_num"] = self.episode_num
        envDict["flight_distances"] = self.flight_distances
        envDict["pillars_touched"] = self.pillars_touched
        try:
            with open(location + "\\" + fileName, 'w+') as f:
                json.dump(envDict, f)
        except Exception as e:
            print("unable to save env as json")
            print(e)
            print(e.__traceback__)

    # ------------Malmo Initialization and Helper Methods----------------
    # Get the pillar locations to be used in GetMissionXML
    def get_pillar_locations(self, width=450, length=450):
        return_string = ""
        for x in range(-1 * int(width), int(width)):
            for z in range(-1 * int(length), int(length)):
                if abs(x) > 30 or abs(z) > 30:
                    if randint(1 / self.current_pillar_frequency) == 1:
                        return_string += f"<DrawLine x1='{x}' y1='2' z1='{z}' x2 = '{x}' y2 = '100' z2 = '{z}' type='coal_block'/>\n"
        return return_string

    # Get the Mission XML for any given mission
    def get_mission_XML(self):
        # <DrawCuboid x1="-20" y1="2" z1="1" x2="-10" y2="100" z2="30" type="air"/>
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
                            <DrawCuboid x1="-600" y1="2" z1="-600" x2="600" y2="100" z2="600" type="air"/>
                            ''' + \
               self.get_pillar_locations() + '''
                            <DrawBlock x="0" y="60" z="0" type="lapis_block"/>
                        </DrawingDecorator>
                        <ServerQuitWhenAnyAgentFinishes/>
                        <ServerQuitFromTimeUp timeLimitMs="45000"/>
                    </ServerHandlers>
                </ServerSection>

                <AgentSection mode="Survival">
                    <Name>elytrAI</Name>
                    <AgentStart>
                        <Placement x="0.5" y="61" z="0.5" yaw="0" pitch="5"/>
                        <Inventory>
                            <InventoryItem slot="38" type="elytra"/>
                        </Inventory>
                    </AgentStart>
                    <AgentHandlers>
                        <HumanLevelCommands/>
                        <AbsoluteMovementCommands/>
                        <ObservationFromFullStats/>
                        <VideoProducer>
                            <Width>''' + str(self.video_width) + '''</Width>
                            <Height>''' + str(self.video_height) + '''</Height>
                        </VideoProducer>
                        <RewardForTouchingBlockType>
                            ''' + \
                            f'<Block reward="{self.pillar_touch_flag}" type="coal_block"/>' + ''' 
                        </RewardForTouchingBlockType>  
                    </AgentHandlers>
                </AgentSection>
                </Mission>
                '''

    # Tells the agent to jump off the starter platform
    def agent_jump_off_starting_block(self):
        """
        Tells the agent to jump off the starting platform and open the elytra
        """
        self.agent_host.sendCommand("forward 1")
        time.sleep(.15)
        self.agent_host.sendCommand("jump 1")
        time.sleep(.1)
        self.agent_host.sendCommand("jump 0")
        self.agent_host.sendCommand("forward 0")
        time.sleep(0.7)
        self.agent_host.sendCommand("jump 1")
        time.sleep(.1)
        self.agent_host.sendCommand("jump 0")
        time.sleep(.1)

    # initialize malmo for the next run
    def init_malmo(self):
        """
        Initialize new malmo mission.
        """
        # Initialize mission specifications
        my_mission = MalmoPython.MissionSpec(self.get_mission_XML(), True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        my_mission.requestVideo(image_width, image_height)
        my_mission.setViewpoint(0)

        # Attempt to start a mission:
        max_retries = 3
        my_clients = MalmoPython.ClientPool()
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000))
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

        # Set the video policy to last frame only
        self.agent_host.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)

        # Keep checking the world_state until the mission has started
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)

        # Tell the agent to jump off the starting block and start flying
        self.agent_jump_off_starting_block()
        return world_state

    # Log the information for the given run
    def log_returns(self):
        # Log pillars hit
        try:
            with open('outputs/PillarTouched.txt', 'w') as f:
                for step, value in zip(self.episodes[1:], self.pillars_touched[1:]):
                    f.write("{}\t{}\n".format(step, value))
        except Exception as e:
            print("Unable to log pillars touched results in text")
            print(e)

        # Log flight distances
        try:
            with open('outputs/DistanceFlown.txt', 'w') as f:
                for step, value in zip(self.episodes[1:], self.flight_distances[1:]):
                    f.write("{}\t{}\n".format(step, value))
        except Exception as e:
            print("Unable to log flight distances in text")
            print(e)

        # Log rewards per step
        try:
            with open('outputs/Returns.txt', 'w') as f:
                for step, value in zip(self.steps[1:], self.returns[1:]):
                    f.write("{}\t{}\n".format(step, value))
        except Exception as e:
            print("Unable to log rewards as text")
            print(e)


# CNN Class
class TorchConvNet(TorchModelV2, nn.Module):

    def __init__(self, *args, **kwargs):
        TorchModelV2.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)

        # Convolutional Layers
        # self.conv1 = nn.Conv2d(input_layers, 16, kernel_size=(3, 3), padding=1)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)

        # Flat Dense Layers
        self.fc1 = nn.Linear(10, 30)
        self.fc2 = nn.Linear(30, 15)
        self.fc3 = nn.Linear(15,5)

        # Output Layers
        self.value_layer = nn.Linear(5, 1)
        self.policy_layer = nn.Linear(5, 4)

        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        c_vals = input_dict['obs']  # Retrieve the image
        c_vals = c_vals.type(torch.float)  # Convert the image from int to float

        # Run the image through the convoluition layers
        # c_vals = F.tanh(self.conv1(c_vals))
        # c_vals = F.tanh(self.conv2(c_vals))

        # Flatten the image while maintaining the batch size
        c_vals = c_vals.flatten(start_dim=1)

        # Run the image through the dense layers
        c_vals = F.tanh(self.fc1(c_vals))
        c_vals = F.tanh(self.fc2(c_vals))
        c_vals = F.tanh(self.fc3(c_vals))

        # Generate output and Return
        self._value_out = F.softmax(self.value_layer(c_vals))
        policy = F.softmax(self.policy_layer(c_vals))

        return policy, state

    def value_function(self):
        return self._value_out.squeeze(1)


# ------------------------------MAIN HELPER FUNCTIONS--------------------------------
# Get the model configuration settings
def get_config(loadPath = ''):
    _config = dict()
    if loadPath != '':
        jsonFilePath = loadPath.split("\\")[:-1]
        jsonFilePath.append("envVariables.json")
        jsonFilePath = "\\".join(jsonFilePath)
        try:
            with open(jsonFilePath, 'r') as f:
                _config['env_config'] = json.load(f)
        except Exception as e:
            print("could not read json file from loadPath, creating new environment")
            _config['env_config'] = {}
    else:
        _config['env_config'] = {}
    _config['num_gpus'] = 0
    _config['num_workers'] = 0
    _config['use_critic'] = True
    # _config['model'] = {'custom_model': 'custom_model',
    #                     'custom_model_config': {}}
    _config['framework'] = 'torch'
    return _config


# Register the custom model
def register_model():
    ModelCatalog.register_custom_model("custom_model", TorchConvNet)


def check_for_loading_model():
    loadPath = ''
    if len(sys.argv) > 1:
        if sys.argv[1] == '-l':
            print("loading file from path", sys.argv[2])
            loadPath = sys.argv[2]
            #reset sys.argv to not include the -l argument
            if len(sys.argv) <= 3:
                sys.argv = [sys.argv[0]]
            else:
                sys.argv = sys.argv[3:].insert(0,sys.argv[0])
    return loadPath

def save_checkpoint(trainer):
    '''
    saves a checkpoint of the trainer, as well as the environement variables into a json file
    
    '''
    saveLocation = trainer.save()
    print("Checkpoint saved, Save Location is:", saveLocation)
    try:
        folderLocation = saveLocation.split('\\')[:-1]
        folderLocation = "\\".join(folderLocation)
        trainer.workers.local_worker().env.save_data_as_json(folderLocation)
    except Exception as e:
        print("Unable to save local environment variables to json")
        print(e)

if __name__ == '__main__':
    loadPath = check_for_loading_model() #must be called first to reset argv for Rllib
    #register_model()
    ray.init()
    trainer = ppo.PPOTrainer(env=elytraFlyer, config=get_config(loadPath))
    if loadPath != '':
        trainer.restore(r"" + loadPath)
    while True:
        print(trainer.train())
        save_checkpoint(trainer)
