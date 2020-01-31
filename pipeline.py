import os
import logging
from src.association import utils, association, feature_extraction


def main():
    project_dir = os.getcwd()
    cfg = utils.get_config(project_dir)

    log_level = cfg['logging_level']
    logging.basicConfig(level=getattr(logging, log_level.upper(), 10))

    audio = cfg['audio']
    real_world = cfg['real_world']

    logging.info(
        'using {} {} dataset'.format('audio' if audio else 'video', 'real_world' if real_world else 'simulation'))

    if not audio:
        feature_extraction.feature_extraction(real_world)

    association.associate(real_world, audio)


if __name__ == '__main__':
    main()
