---
layout: default
title: Status
---

## Project Summary

ElytrAI aims to create an AI that learns how to fly using the Minecraft Elytra item. We created three distinct tasks for our AI to learn. First is flying long distances where it attempts to fly as far as possible in a given direction. Second is to avoid obstacles where the AI will still attempt to fly as far as possible, but will do so while avoiding pillars of diamond blocks in the field it is flying through. A final objective is to learn to land on targets placed throughout the field successfully. To accomplish this we are using project Malmo and pytorch to set up a consistent world and keep track of observations such as height, velocity, pitch, yaw, agent position, and more.

<br><br>

## Approach

Our approach to the problem is to use a Gym environment with the Rllib PPO reinforcement learning implementation in order to teach our AI how to fly. In minecraft the elytra works by moving the camera in different directions. Angling the camera down results in increased speed but decreased altitude while angling to the side will turn in that direction. As such the action space for our model is to pass a moveMouse command where the action will be a 2-tuple of the X and Y direction to move the camera in.

For our observation space, the agent knows its current location in X, Y, and Z, as well as its current velocity in those three directions. It also knows its current yaw and pitch since this represents the direction that the agent is currently looking. We have experimented with various vision models. The first model looks at the agent's current line of sight, for the distance agent, this observation is not necessary. For the obstacle avoiding agent, this observation will determine whether there is an obstacle in the direction the agent is facing, and for the platform landing agent this will determine whether the platform is within sight.

We are currently working to utilize Malmo’s videoProducer agentHandler, to use the visual data provided to the agent instead of grid observations. We are hoping that this change will allow our agent to have a vision model with more parity with what we see as developers, that will allow it to better complete its objective.

For reward models we have differing rewards based on the task the agent must accomplish.

-----

**Distances** - Reward is based on distance travelled in the positive Z direction multiplied by a constant to reduce the scale of the rewards. In our most successful tests, a constant of 0.02 produced the best results

-----

**Obstacles** - We kept the distance reward for moving in the positive Z direction, we also included a penalty for the agent touching a pillar. We added an additional observation for whether a pillar had been hit during the episode, if it has, the observation is passed to the agent, and all future rewards are reduced by some amount, for our best trial, we halved future rewards after the agent touched an obstacle.

We also experimented with a more advanced penalty system, where the agent would be penalized for moving closer to a pillar. To do this we used the malmo ObservationFromGrid and created a second grid which had rewards scaled down for grid cells that were close to obstacles. We did not find a ton of success with this method.

![pillarRewardVisualizer](https://user-images.githubusercontent.com/44657382/142092864-737b6de5-3a2c-48e0-ba23-cdeac06a2bef.png)

------

**Platform** - We reward the agent for movement closer to the platform's location based on X, Y and Z coordinates in minecraft. The agent is given a “score” based on its final position. This score is created by comparing the agent’s position against the platform’s position and calculating a percentage that represents how close the agent is to the platform. For example, if the agent is 20% closer to the platform than when it started, it will get a score of .20. The closer the agent ends the episode to the platform, the higher the reward -- with the highest reward for landing on the platform.

In a previous iteration, we tried rewarding the agent constantly throughout its flight. This version rewarded the AI more if it headed in the direction toward the platform, and penalized it if it headed the incorrect way. Besides a couple of successful lands, the data showed that the agent did not learn where the platform was.

<br><br>

## Evaluation

Our quantitative evaluation criteria for our agent is relatively simple, we are aiming for the agent to travel the furthest distance possible, with each run hopefully covering more distance than the previous run. So for our evaluation criteria we logged the total distance travelled in the Z direction by the agent at the end of each episode to a text file. We then used the text results in Google Sheets in order to make a graph showing the trend of the agent’s episodes.

For our graphs, we grouped the runs into groups of 10 and plotted those averages as our total episode flight distance. Finally we put a trend line on the chart in order to project whether our agent is improving as the episodes progress. We also plotted the minimum and maximum flight distance for each set of episodes along with the variance. 

For our obstacle avoidance runs, we kept track of the amount of runs within the last 20 that the agent touched an obstacle and used this as a quantitative measure for whether the agent was successfully avoiding the obstacles.

For our platform runs, we tracked the score the agent received each run to visualize how much closer the agent progressed toward the platform.

Finally we also kept track of the rewards per step for our agent, similar to how we did it in assignment 2. This allowed us as developers to view the rewards our agent was receiving for its performance and to make sure the rewards were improving over time. If the rewards do not improve it means the agent is having trouble understanding how to use the action_space we provided to increase its rewards.

if the rewards are increasing over time, but the other quantitative measures are not, it means that our rewards values are not good for the task we are attempting to solve, since the agent is maximizing rewards without completing the task.

<br>

Notable Runs: our best distance only run.

![distanceOnly](https://user-images.githubusercontent.com/44657382/142092865-fc24f018-c185-4c7e-8c5d-de5afbb0011f.png)

![distanceReturns](https://user-images.githubusercontent.com/44657382/142092867-d6d220bc-54e3-49b4-b168-37f0126d051a.png)


As can be seen from the graphs, we have a strong linear relationship between episodes and distance travelled meaning that the agent is accomplishing the task of travelling an increased distance with each run. The same trend is seen from the return vs step chart, which means that the agent is gaining increased rewards as it completes more episodes.

<br>

our best obstacle avoidance run:

![obstacleDistance](https://user-images.githubusercontent.com/44657382/142092870-c8b9b238-8af7-4eb8-9fba-77482534b441.png)

![obstacleReturns](https://user-images.githubusercontent.com/44657382/142092872-0b40ae50-9e64-4626-bd30-1c3df09a3bfe.png)

As can be seen from these graphs, our agent still increased the distance travelled with each successive flight, the return graph however shows a much weaker correlation(ignoring the dip in the middle caused by an error with malmo). This shows that the reward system is still functional, but could likely be improved to allow the agent to learn faster.

A similar problem can be seen from looking at the pillars touched graph

![obstaclePillars](https://user-images.githubusercontent.com/44657382/142092871-a4cc648d-7446-4355-adb1-89de39f91dbb.png)

We can see that the agent is not decreasing the percentage of runs with pillars touched, but almost seems to increase it over time. This shows that we must modify our reward system, or allow our agent more fine-tuned control over the camera to allow avoidance of our obstacles.

<br>

Platform Run:

![platformReturns](https://user-images.githubusercontent.com/44657382/142092862-5a5b2433-7fc2-455c-b2b6-167c87ef4bb2.png)

Our best platform run can be seen below where the agent gets closer to the platform, and a small positive linear relationship is seen. The variance in the agent’s score, and therefore the range of X, Y and Z positions it flies to, reduces greatly which signals that the AI is learning which areas the platform is not in. However, the agent still struggles to find and land on the platform and does not improve fast enough. This shows we will need to modify our vision model to give the agent more information.


<br><br>

## Remaining Goals and Challenges

For the remaining few weeks in the class we intend to replace our vision model using the videoProducer agent handler from Malmo. After consulting with the professor, we feel that using this vision model would be a more interesting problem since our agent would be observing data that is closer to what is observed as a human player. This new vision model should improve our agent’s ability to see and dodge obstacles while flying.

We also need to improve our platform landing AI since the current iteration we have has been largely unsuccessful in actually finding and landing on the platform.

We anticipate having some difficulties implementing the improved vision model, since documentation on Malmo is quite sparse. Additionally, there is a possibility that the implementation of our vision model might not improve our model, in which case we will have to find some other way to improve the agent’s ability to avoid pillars.

A final challenge is the platform landing AI which is still quite far off of consistently landing on the platform, so if we do not have time to improve both the obstacle avoidance and platform landing AI, we might have to remove the platform landing from our project scope, and instead focus our efforts on improving the obstacle avoidance.

<br><br>


## Video Summary


**EMBED VIDEO HERE**
## Resources Used

* We used the RLlib reinforcement learning library for the core of our agent. We had the most success using the PPO trainer from the library.

* We created a custom environment using Gym.env.

* We used Google sheets as a medium for creating charts once we acquired data on our agent’s performance during training. This allowed us to create charts from our data after the fact to better understand how our agent is performing.

* We used tkinter to visualize the agent’s line of sight.

