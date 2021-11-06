# elytrAI
******

## Usages
*********
1. launch a malmo client using malmo install on your system

2. navigate to the folder containing ElytrAI.py and use python ElytrAI.py to begin training
    * while training the console should output locations that checkpoints are saved to, these paths can be used to restore models to the state they were in that checkpoint


## restoring model
********
1. when launching the script use the -l command followed by a checkpoint path to load a model

example: python Elytrai.py -l C:\Users\scott/ray_results\PPO_elytraFlyer_2021-11-05_18-40-315cb4p9wn\checkpoint_18\checkpoint-18
