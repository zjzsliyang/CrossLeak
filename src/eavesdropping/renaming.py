#! /usr/bin/python3
import os
import yaml

with open("setup.yaml","r") as f:
    setup = yaml.load(f, Loader=yaml.FullLoader)

FROM_FOLDER = setup["raw_video_folder"]
TO_FOLDER = setup["dated_video_folder"]
os.makedirs(TO_FOLDER, exist_ok=True)

for filename in os.listdir(FROM_FOLDER):
    date = os.popen('date -r "'+FROM_FOLDER+filename+'" "+%Y_%m_%d_%H_%M_%S"').read().strip("\n")
    os.system("cp "+FROM_FOLDER+filename+" "+TO_FOLDER+date+".h264")


