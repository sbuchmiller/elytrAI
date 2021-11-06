import json

dic = {}
dic["a"] = "b"
file = open('C:\\Users\\scott/ray_results\\PPO_elytraFlyer_2021-11-05_18-16-44_m__93ws\\checkpoint_8\\envVariables.json', 'w+')
json.dump(dic,file)
