#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper
from PIL import Image

from duckienet_config import inet_cfg

sys.path.append('/home/yuanbo/projects/2309/intention_net/intention_net')
from control.policy import Policy
from keras.preprocessing.image import load_img, img_to_array

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
args = parser.parse_args()

if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        seed = args.seed,
        map_name = args.map_name,
        draw_curve = args.draw_curve,
        draw_bbox = args.draw_bbox,
        domain_rand = args.domain_rand,
        frame_skip = args.frame_skip,
        distortion = args.distortion,
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()
top_down = False

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """
    global top_down

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        top_down = False
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.SPACE:
        top_down = not top_down
        if top_down:
            env.render(mode='top_down')
        else:
            env.render()
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

# Load inet model
num_intentions = 4
num_control = 2
policy = Policy(inet_cfg['intention_mode'], inet_cfg['input_frame'], num_control, inet_cfg['model_dir'], num_intentions)

next_action = np.array([0.0, 0.0])

def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    global next_action

    action = next_action

    if key_handler[key.UP]:
        action = np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action = np.array([-0.44, 0])
    if key_handler[key.LEFT]:
        action = np.array([0.35, +1])
    if key_handler[key.RIGHT]:
        action = np.array([0.35, -1])
    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info = env.step(action)
    # print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))

    # From obs, intention and speed, derive next action using IntentionNet
    # next_action = 

    # Model inputs
    img = Image.fromarray(obs).resize(size=(224, 224))
    img = img_to_array(img)
    intention = img_to_array(load_img(f"{inet_cfg['data_dir']}/intentions/I_0.png", target_size=(224, 224)))
    speed = np.array([env.get_agent_info()['Simulator']['robot_speed']])

    next_action = policy.predict_control(img, intention, speed).squeeze()
    # print("Predicted control: ", next_action)

    if key_handler[key.RETURN]:
        im = Image.fromarray(obs)

        im.save('screen.png')

    if done:
        print('done!')
        env.reset()

    if top_down:
        env.render(mode='top_down')
    else:
        env.render()


pyglet.clock.schedule_interval(func=update, interval=1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
