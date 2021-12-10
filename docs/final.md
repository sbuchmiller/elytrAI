---
layout: default
title: Final Report
---

Scott Buchmiller

Alec Grogan-Crane

Alexandria Meng

# Project Summary

ElytrAI aims to create an AI that learns how to fly and avoid obstacles using the Minecraft Elytra item. To start, our agent attempts to fly as far as possible in a given direction. Second, we teach the agent to avoid obstacles while still attempting to fly as far as possible. To accomplish this we are using project Malmo and pytorch to set up a consistent world and keep track of observations such as height, velocity, pitch, yaw, agent position, and more.

The setup for this project is to create an agent that will learn the basic strategies to fly long distances and also to avoid obstacles. To accomplish this we created a Minecraft world using the Malmo world generator. This world contains lava for the floor and a single block placed at Y: 61, meaning that the agent starts 60 blocks above the lava. The world also contains pillars placed randomly in front of the agent’s starting position in the direction of travel, so the agent must avoid the pillars to travel in the goal direction.

<img width="743" alt="Screen Shot 2021-12-10 at 2 54 00 PM" src="https://user-images.githubusercontent.com/36436765/145651331-1de5257a-f6a5-48cf-920a-81703d457d63.png">


<br>

The final goal of the agent is to travel the maximum distance in the given direction, the episode will terminate early if the agent hits an obstacle, (simulating the agent crashing) so the agent must also learn to avoid obstacles while maintaining distance.

<br>

 To solve this problem, we chose to use reinforcement learning through the rllib PPO model. This is because the mechanics to fly the elytra in minecraft are complex, they use camera movements, which can be adjusted in a continuous action space, and the optimal movements to gain the greatest distance have not been well explored. It is clear that some pattern of moving the camera up and down to maintain velocity works well, however this method isn’t super trivial to implement, especially when taking dodging pillars into consideration since the optimal distance to move is unknown. So using our model we hope to find the most optimal method to fly the greatest distance.

 <br>

## Approaches

### General Agent Setup

Our agent uses a custom class that inherits from the gym.env module in order to train our agent. Using this module, our agent’s actions are controlled by a few steps. First the agent takes an observation of the game state, this observation is fully customizable and represents the information we choose to pass to the agent in order for it to decide on a next step. Then the neural network beneath the rllib.PPO model chooses an action from the action_space that we provide it. The action chosen is based on what the agent has learned from the previous test. Then the action is completed within the malmo client, additional observations are returned, and the cycle repeats until the episode ends once the agent dies or times out.

Steps are taken every 0.1 seconds in-game time. Due to the way malmo is set up, the agent can’t move too quickly, or the observations returned at each step will not be different from the last steps. Additionally, any calculations that the agent must finish must also be done in this 0.1 second window, so that the agent can choose an action before it needs to act. 

Our observation space for this experiment is the main area that we modified to try different methodologies of solving the problem. The basic observations returned are the agent’s x, y, and z coordinates, the x, y, and z velocities, and the agent’s current pitch and yaw. All additional observations are more experimental and change per trial that we are running. Rewards are also calculated in the observation step, the general agent reward is to reward the agent based on the z distance traveled from the starting block, but we also tested other methods of rewards for different trials

For our action space, the action space of our agent is strictly mouse movements since this is the method to control the elytra within minecraft. to accomplish this we used the AgentHost.Send(moveMouse {x} {y}) command from the HumanLevelCommands module within Malmo. This allows our agent to move the mouse any amount to the left or right and up or down within each step. In order to avoid the agent moving too sporadically, we put a cap of 100 degrees in either direction.

Finally to start the agent flying, we used a pre-scripted action sequence, the agent starts each episode by moving forward while jumping, then jumping again to deploy the elytra after a short delay. We had to implement the start using this method because the malmo mechanics around deploying the elytra acted inconsistently between runs. For example, if the agent didn’t die due to losing health in the last run the elytra would not be able to be deployed, and if the delay was too short it would also not deploy. This avoided the agent overfitting to start the trail and wasting a significant amount of computation time learning to avoid malmo bugs.
     

### DISTANCE
To begin the project, we wanted to be sure our agent would be able to fly as far as possible. Due to a bug in the Minecraft version that Malmo uses (1.11.2), we would not be able to let the AI use fireworks to gain altitude as originally planned, so we instead relied simply on how far the agent was able to travel using the starting platform height of 60 blocks to gain velocity.

We rewarded our agent based on the Z distance it is able to travel from its starting position using the PPO reinforcement learning implementation. The further the agent flew, the more it would be rewarded. From the graph below, we could see that the more simulations the agent ran, the further it learned how to fly. This was a success!

<img width="837" alt="Screen Shot 2021-12-10 at 2 54 51 PM" src="https://user-images.githubusercontent.com/36436765/145651407-6206735f-4f84-42f0-bf51-00c0e1b9dfce.png">


### OBSTACLES

Now that we were confident the agent would be able to learn to fly long distances, we moved on to creating obstacles. This presented a more interesting problem than the agent simply flying in a line since it gave a reason for the agent to move left and right as well as a secondary goal. For our obstacles, we created 1 block wide pillars randomly within a 100x by 400z rectangle of the agent’s spawn point.

For our tests we used a pillar placement rate of 0.005 meaning that 0.5% of tiles had a pillar created on them, we felt this struck a good balance between presenting obstacles for the agent, and forcing the agent to only avoid pillars without focusing at all on distance due to too many obstacles.

<img width="923" alt="Screen Shot 2021-12-10 at 2 56 05 PM" src="https://user-images.githubusercontent.com/36436765/145651538-c7196d42-c01b-4b40-a529-60655282f4a3.png">


### Line of sight

Our first attempt at implementing a vision system for our agent used the line of sight feature in Malmo’s ObservationFromGrid function. This function returns the grid of all blocks surrounding the agent as well as the block that is currently in the agent’s line of sight. For this methodology we provided the type of block the agent is currently looking at as well as the x, y, and z, coordinates of that block in our observation space.

<img width="1091" alt="Screen Shot 2021-12-10 at 2 56 42 PM" src="https://user-images.githubusercontent.com/36436765/145651594-d09e1781-ba01-41fb-a26b-0985fac13ff0.png">

For rewards in this method we provided the agent a reward based on its z position relative to the original block multiplied by a constant to scale down the reward. In our case this was 0.02. For hitting a pillar we penalized the agent with a negative reward equal to 5 times the positive reward that it would have received, and additionally halved the rewards from the rest of the run.

We hoped that this reward system would show the agent that after hitting a pillar, the rest of the run was worth less total reward, forcing the agent to avoid hitting pillars.

### Rewards Field

For our second attempt, we wanted to give the agent more information pertaining to its surroundings. We did this by using the malmo ObservationFromGrid to get an array of the blocks that surrounded the agent. With the blocks available to us, we created a ‘field’ around any pillars within the agent’s field of view where the closer to any pillars, the values become closer to 0. We used a variable to keep track of this radius, and in our final tests using this method we used a radius of 12 blocks around each pillar. The current reward of the agent is then multiplied by its current value within the field as a way of punishing it for getting too close to pillars. 

We then modified the observation to include the ‘field value’ of the agent's current position, as well as the values of the positions one block to its left and right. This was in an attempt to give it information about which direction will give it a higher reward. In the hope that it could use these observations in order to make decisions preemptively about the actions to take instead of just being punished for it. 

<img width="738" alt="Screen Shot 2021-12-10 at 3 04 26 PM" src="https://user-images.githubusercontent.com/36436765/145652122-19de2e87-f5be-4f09-b6a2-907dd9cfab32.png">

### VideoProducer all Pixels

In our first attempt at using VideoProducer for pixel data, we captured the screen at every step and passed the pixels in as observations. This was less of a “cheating” way to teach the agent as instead of passing in information that real players wouldn’t be able to discern (such as pitch and yaw), we were providing visual information that a player would see. This allows our agent to make its inferences independently and is easier to create on the developer side of things. However, we found that the cost of doing this was an influx of too much data as the screen was too large.

To incorporate the entire screen as the observation, we had to change the observation space to be a 4 x 200 x 200 matrix where each value represented a pixel value for a given color. 200 x 200 was the size of the screen we were capturing and the 4 layers represented the RGB and depth values of each pixel. This, however, meant that we were now processing 160,000 input values and we knew a standard Neural Network would not work well for this. Neural networks also require flat inputs meaning we would have to flatten the matrix and we would lose the spatial properties of the image.

To avoid these problems, we implemented our own custom convolution neural network. Our model first ran the input through a convolution layer with a kernel size of 5 and padding of 2 which was then ran through a relu function. This output was then sent through a max pooling layer with a size and stride of 5. This significantly reduced the size of the image and then we sent it through another two layers set up in the same fashion. The output of the second max pooling layer was then flattened and processed by two dense layers. This model architecture allowed it to retain the spatial properties of the input image while significantly decreasing the processing time. 

Beyond the changes we made to the neural network, we also adapted the agent’s environment to suit the new observations. Having pillars randomly all over the place makes it very easy to get confused as to which direction you are looking, even as humans we struggled with this during testing. So we decided that, since we are no longer sending the agent information about it’s position and direction, to enclose the environment with walls and effectively make a hallway for the agent to follow. With the walls, pillars, lava, and sky all different colours, this gave the agent’s vision model the best chance to differentiate between the things they are looking at. 

<img width="902" alt="Screen Shot 2021-12-10 at 2 58 08 PM" src="https://user-images.githubusercontent.com/36436765/145651663-36a1ab2f-d4e5-4a78-a834-ffe3ad6a6bf1.png">

### VideoProducer Pre-Processed
In this attempt at using VideoProducer we aimed to reduce the computational load on the neural network by simplifying the observation space. One of the problems we were faced with was the time the network was taking to back-propogate between epochs when passing in all pixels as observations. So using this method we greatly simplified the observation space. This method also simplified each column into only its color feature, so calculating the action was much faster than looking at each pixel individually.

In this test we pre-processed the images created by Videoproducer using Pythons PIL library. To do this we first cropped the image into the center columns. In our tests we created 10 columns that were 20x240 pixels each. We then applied a grayscale filter to the image since we no longer needed RGB colors. Then we took the average pixel color of the pixels in the 20x240 images and passed those as the observations to the agent. The process can be seen in the image below.

<img width="1037" alt="Screen Shot 2021-12-10 at 2 58 42 PM" src="https://user-images.githubusercontent.com/36436765/145651716-2c42d586-55ad-435b-847a-5edd444d62df.png">

Using this method the average pixel color decreased when a column was visible in the region of the screen covered by each given column. With this the agent was able to “see” whether each section of the screen contained a column and we hoped this would allow it to learn to avoid regions with a low pixel color since there is a higher chance that it would contain an obstacle.

Unlike the observation space with all pixels as observations, in this space we still used some ObservationFromGrid observations. This allowed the agent to see it’s current pitch and yaw so that it would not get “lost” and begin moving a direction away from the goal direction as the other agent seemed to in our preliminary tests.

We attempted this test using both our custom model as well as the default PPO model. We saw greatly increased gains when using the PPO model, so we decided to use this as the primary model for this test. This is likely because our custom model was created with a 2 dimensional grid of observations in mind, while this observation space contained a 1 dimensional vector of values that were unrelated to one another.

For the rewards under this methodology, we kept it simple by rewarding the Z distance with a constant multiplier of 0.02 to scale the rewards. If the agent hit a pillar we would penalize it with a penalty of negative 10 times the reward it would have received as well as terminating the run early.

We tested this method with both a continuous action space similar to the other tests, where the agent could move 100 degrees horizontally and or vertically each step, we also tested with a discrete action space where the agent would choose up down left or right and would always move 40 degrees in that direction.

## Evaluation

We had the members of our group, as well as a few of our peers attempt the same test we were giving the agent. The overall results were that the human players could travel between 300 and 350 blocks. In our tests no pillars were hit, this is not a super realistic goal due to the game having a render time on our computers due to lack of computational power. In many tests the game did not render fully before the agent entered an area of the game, so some tests hit poles due to the agent not having observations. Because of this, we can call a total success a case where the agent hits pillars in about 5% of runs. These were the baselines for our agent, we expect the distance it can travel to be somewhere between those 300 and 350, and for it to hit pillars in 5% of runs or less.

### Line of sight
In this test we found that the agent was able to maintain the gains in distance that we saw in the initial tests where obstacles were not introduced. The agent did not however learn to avoid colliding with obstacles during runs.

We believe that this is because the basic line of sight observations miss many very important aspects of the problem. Firstly, the agent can only see the block that it is directly looking at. This means that it has no real awareness of its surroundings and might be very close to hitting a pillar, but not see it since it isn’t looking directly at it. Additionally, this method doesn’t allow the agent to remember the block it was looking at in previous steps, so it would often look away from a block then look back at it again later due to its lack of memory.

Finally the observation itself suffered from some inconsistencies due to Malmo. If the agent’s vision wasn’t perfectly aligned to the block, there were times where the block wouldn’t be seen. This led to situations where the agent would not be looking at a pillar, but would still be close enough to collide with it while passing.
<img width="475" alt="Screen Shot 2021-12-10 at 3 00 00 PM" src="https://user-images.githubusercontent.com/36436765/145651805-839dba21-0bf8-41d2-af2d-0eb3511392a4.png">

<img width="512" alt="Screen Shot 2021-12-10 at 3 00 30 PM" src="https://user-images.githubusercontent.com/36436765/145651817-4874c2b4-5962-40fb-8f2b-d5c9dd8271ba.png">


### Reward Field

After several runs, we found that the agent was not learning well with this implementation. The agent did not seem to be actively avoiding any pillars or responding to different penalty or reward values. We realized that penalizing the agent for flying close to the pillar but not hitting it was not “bad” behavior because the agent was avoiding the obstacles. Finding an optimal distance between the agent and the poles was not our goal, thus we realized this approach did not represent our problem well enough.

Continuing this approach would require us to fine tune the severity and range of the rewards and penalties, but testing varieties of strengths and ranges would take too much time and difficulty to optimize. From the graphs below you can see that the agent did not improve in earning rewards nor did it consistently avoid pillars.

<img width="1197" alt="Screen Shot 2021-12-10 at 3 00 55 PM" src="https://user-images.githubusercontent.com/36436765/145651867-b1d78238-0e4a-4a4e-ae2d-1bd5ab6ff74b.png">


### VideoProducer all Pixels
This model was far more promising than our previous attempts. Our first attempt  at training immediately showed improvements and made good progress and after several iterations of tweaking the model’s convolution layers and hyperparameters, the agent managed to make great progress.

The average run improved in distance traveled rapidly and began to level off after about 200,000 steps and continued to make slight improvements all the way until we terminated the test at 600,000 steps. We also saw that the agent continually reduced the number of pillars hit over the course of the test; however, it never quite reached a level we were totally happy about. 

The main issue we believe is with its inconsistency. Most of the time, the agent was able to dodge pillars completely and go as far as it could before it lost speed and fell into the lava. But every once in a while it would clip the edge of a pillar, thus ending the run. These inconsistencies ended up bringing the average results down but nevertheless, the results still show improvement.

<img width="1120" alt="Screen Shot 2021-12-10 at 3 01 46 PM" src="https://user-images.githubusercontent.com/36436765/145651912-71539bcd-bb42-4b4e-a74a-8a187a15d604.png">

### VideoProducer pre-processed
The pre-processed VideoProducer test was our most successful in terms of pillar avoidance. As previously stated, we tested this method with both a continuous and discrete action space. For the continuous test see the graphs below.

<img width="525" alt="Screen Shot 2021-12-10 at 3 02 06 PM" src="https://user-images.githubusercontent.com/36436765/145651937-09ae2e21-4b7e-42d2-8824-f24152194dac.png">

<img width="526" alt="Screen Shot 2021-12-10 at 3 02 28 PM" src="https://user-images.githubusercontent.com/36436765/145651972-f3180499-f697-44b6-a2e7-18cc949b7fae.png">


As can be seen from the graphs, using this method the agent was effectively avoiding all pillars after around 2000 episodes. This agent was able to accomplish our goal for pillar avoidance, however the agent seemed to suffer in terms of distance traveled. It topped out at around 230 units traveled, which is less than our initial distance tests and is also less than our 300-350 block goal.

This lack of distance could either be a result of overfitting on pillar avoidance where the agent cared more about avoiding a pillar hit penalty than about the distance reward. It could also be a consequence of having too much choice with a continuous action space, where the agent needed more time to learn in order to gain more distance.

We also tested the agent in a discrete action space. See graphs below.

<img width="486" alt="Screen Shot 2021-12-10 at 3 03 15 PM" src="https://user-images.githubusercontent.com/36436765/145652016-0613564f-f371-48bd-8b03-c68009eddb81.png">

<img width="583" alt="Screen Shot 2021-12-10 at 3 03 35 PM" src="https://user-images.githubusercontent.com/36436765/145652041-12a4324e-4721-46b5-a859-67cdd92a55ae.png">

In this test we see similar results to the continuous action space. The agent caps out around the 0.08 pillars hit mark, which is slightly above what our goal was of 0.05 pillar hit. This agent however was able to travel an average of around 300 which is our goal for distance and is much further than the agent using a continuous action space.

We believe that this result means that the continuous action space would have benefitted from more time to learn since the distance traveled by the discrete agent was greater than the continuous agent. Additionally, any actions the discrete agent could take, the continuous agent could also learn to take assuming it is the optimal goal, so with more training it is likely that the continuous agent could match or exceed the discrete agent’s distance while still maintaining a lower average pillar hit.
  


<br><br><br><br>

References

https://microsoft.github.io/malmo/0.21.0/Schemas/MissionHandlers.html

http://microsoft.github.io/malmo/0.30.0/Documentation/classmalmo_1_1_mission_spec.html

https://pytorch.org/docs/stable/nn.html

https://docs.ray.io/en/latest/rllib-env.html

https://docs.ray.io/en/latest/rllib-models.html#custom-pytorch-models

https://pillow.readthedocs.io/en/stable/

https://en.wikipedia.org/wiki/Convolutional_neural_network

https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53?gi=608691ea705b

https://medium.com/@BonsaiAI/deep-reinforcement-learning-models-tips-tricks-for-writing-reward-functions-a84fe525e8e0

https://github.com/Microsoft/malmo

