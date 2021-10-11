---
layout: default
title: Proposal
---

## project Summary
***
Our team is designing an AI that will learn to fly an eytra in Minecraft. To acomplish this we will be using project Malmo in order to set up a consistent world where our agent starts at a certain height at the beginning of each mission.

We will then take observations as the height above the ground our agent currently has, the speed our agent is traveling (calculated as the distance traveled from the last observation), as well as any blocks directly in front of the path of our agent. For output we will be focusing on camera movements since this is the primary method to control the Elytra flight.

Once our agent can conisistently fly the elytra, our group may introduce additional mechanics. Currently we plan to introduce fireworks, which can be used while flying an elytra to gain a temporary speed boost. The agent will then have to learn the best time to use these fireworks to fly further.

Possible additions for later in the project could include obstacles which the agent would have to navigate to maximize speed and distance. Also we may introduce the agent to fireworks which would allow them to gain a boost in flight speed when lit. But with only a limited number available, they would have to learn to use them sparingly and at the right time.

<br>

## algorithms
***
We currently plan to use reinforcement learning  to achieve the goal of flying the elytra. We may also attempt to use deep-learning.

<br>

## evaluation of success
Our goal for this projet is to get the agent to fly as far as possible using the Elytra, so we will reward distance travelled and negatively reward the agent taking fall damage or colliding with an obstacle. Using the quantitative metric of distance, we would classify a success as the AI being able to fly the elytra at least as far as a human playing minecraft would be able to without knowledge of any special tricks to gain more distance.

In order to verify our algorithm is working we will run many trials using Malmo's mission system. We will ensure that our algorithm is taking different approaches across iterations and that hopefully the distance travelled is increasing over time instead of decreasing. Our "moonshot" case would be our AI discovering some method that can make the elytra fly further than a normal player would be able to.
