#! /usr/bin/python3
import os
import yaml

with open("setup.yaml","r") as f:
    setup = yaml.load(f, Loader=yaml.FullLoader)

RV_FOLDER = setup["raw_video_folder"]
V_FOLDER = setup["dated_video_folder"]
W_FOLDER = setup["wifi_data_folder"]
A_FOLDER = setup["audio_folder"]

os.system("scp "+V_FOLDER + "*.h264 user@laptop:~/video/ > /dev/null 2>&1")
os.system("scp " + W_FOLDER + "* user@laptop:~/wifi_data/ > /dev/null 2>&1")
os.system("scp " + A_FOLDER + "*.wav user@laptop:~/audio/ > /dev/null 2>&1")

os.system("rm -rf"+RV_FOLDER+"*")
os.system("rm -rf"+V_FOLDER+"*")
os.system("rm -rf"+W_FOLDER+"*")
os.system("rm -rf"+A_FOLDER+"*")


