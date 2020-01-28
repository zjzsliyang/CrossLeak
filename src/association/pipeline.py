import os
from association import association, feature_extraction, utils

project_dir = os.path.dirname(os.getcwd())
cfg = utils.get_config(project_dir)

audio = cfg['audio']
real_world = cfg['real_world']

if not audio:
    feature_extraction.feature_extraction(real_world)

association.associate(real_world, audio)