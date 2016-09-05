import logging
from argparse import ArgumentParser as AP

import numpy as np

from luchador.env import ALEEnvironment


logger = logging.getLogger('luchador')

ap = AP()
ap.add_argument('--rom', default='breakout')
ap.add_argument('--mode', choices=['test', 'train'], default='train')
ap.add_argument('--width', type=int, default=84)
ap.add_argument('--height', type=int, default=84)

ap.add_argument('--repeat_action', type=int, default=4)
ap.add_argument('--random_start', type=int, default=0)

ap.add_argument('--buffer_frames', type=int, default=2)
ap.add_argument('--preprocess_mode', default='max')

ap.add_argument('--grayscale',
                dest='grayscale', action='store_true')
ap.add_argument('--color',
                dest='grayscale', action='store_false')

ap.add_argument('--minimal_action_set',
                dest='minimal_action_set', action='store_true')
ap.add_argument('--legal_action_set',
                dest='minimal_action_set', action='store_false')

ap.add_argument('--display_screen', '-screen', action='store_true')
ap.add_argument('--sound', action='store_true')
ap.add_argument('--record_screen_path')
ap.add_argument('--plot', action='store_true')
args = ap.parse_args()

env = ALEEnvironment(
    args.rom,
    mode=args.mode,
    width=args.width,
    height=args.height,
    repeat_action=args.repeat_action,
    random_start=args.random_start,
    buffer_frames=args.buffer_frames,
    preprocess_mode=args.preprocess_mode,
    grayscale=args.grayscale,
    minimal_action_set=args.minimal_action_set,
    display_screen=args.display_screen,
    play_sound=args.sound,
    record_screen_path=args.record_screen_path,
)

logger.info('\n{}'.format(env))

n_actions = env.n_actions
for episode in range(10):
    total_reward = 0.0
    env.reset()
    ep_frame0 = env._ale.getEpisodeFrameNumber()
    for n_steps in range(1, 10000):
        a = np.random.randint(n_actions)
        outcome = env.step(a)

        total_reward += outcome['reward']
        if outcome['terminal']:
            break

    env_state = outcome['state']
    ep_frame1 = env_state['episode_frame_number']
    logger.info('Episode {}:'.format(episode))
    logger.info('  Score : {}'.format(total_reward))
    logger.info('  Lives : {}'.format(env_state['lives']))
    logger.info('  #Steps: {}'.format(n_steps))
    logger.info('  #EpisodeFrame: {} -> {}'.format(ep_frame0, ep_frame1))
    logger.info('  #Total Frames: {}'.format(env_state['total_frame_number']))

observation = outcome['observation']
logger.info('Screen type: {}'.format(type(observation)))
logger.info('Screen shape: {}'.format(observation.shape))


if args.plot:
    import matplotlib.pyplot as plt
    plt.imshow(observation, cmap='Greys_r')
    plt.show()