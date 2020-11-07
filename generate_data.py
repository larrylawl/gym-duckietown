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
from gym_duckietown.simulator import ROBOT_LENGTH, ROBOT_WIDTH
from PIL import Image
from shutil import copyfile
import glob
import os

# from experiments.utils import save_img

"""

Mobile robot motion planning sample with Dynamic Window Approach

author: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı

"""

import math
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
parser.add_argument('--save', default=False, type=str2bool, help='Saves datasets.')
parser.add_argument('--image_dir', default="./data/images/", type=str, help='Directory to save images.')
parser.add_argument('--action_dir', default="./data/actions/", type=str, help='Directory to save actions.')
parser.add_argument('--intent_dir', default="./data/intentions/", type=str, help='Directory to save intentions.')
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

print(args)
if args.save:
    # sets variables
    save = args.save
    image_dir = args.image_dir
    action_dir = args.action_dir
    intent_dir = args.intent_dir
    counter = 0
    # Ensures image and action directory are specified
    assert image_dir is not None
    assert action_dir is not None
    assert intent_dir is not None

    print(f"Save is turned on. Images will be saved in {image_dir}. Actions will be saved in {action_dir}. Intentions will be saved in {intent_dir}.")
    
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
        env.render()
        top_down = False
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)
    elif symbol == key.SPACE:
        top_down = not top_down
        if top_down:
            env.render(mode='top_down')
        else:
            env.render()

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

# def should_plan():
    # Replans after %plan_freq number of steps. Note that no steps are taken when stationary.
    # return plan_counter == plan_freq
        # action taken
        # global plan_counter
    #     if plan_counter == plan_freq:
    #         # plan_counter = 0
    #         result = True
    #     else:
    #         plan_counter += 1
    # return result
        

# calls planning every %plan_freq steps
plan_freq = 100
plan_counter = plan_freq

def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    
    global plan_counter
    action = env.get_agent_info()['Simulator']['action']
    plan = plan_counter == plan_freq
    if plan:
        plan_counter = 0
        dwa()
    elif action[0] != 0.0 or action[1] != 0.0: # only increase plan counter if you move
        plan_counter += 1

    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action = np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action = np.array([-0.44, 0])
    if key_handler[key.LEFT]:
        action = np.array([0.35, +1])
    if key_handler[key.RIGHT]:
        action = np.array([0.35, -1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info = env.step(action)
    print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))

    if key_handler[key.RETURN]:
        # TODELETE: print('key.RETURN pressed!')
        im = Image.fromarray(obs)

        im.save('screen.png')

    if done:
        print('done!')
        env.reset()
    
    if top_down:
        env.render(mode = 'top_down')
    else:
        env.render()

    if args.save:
        global counter 

        im = Image.fromarray(obs)

        # from random import randint
        image_filename = f'X_{counter}.png'
        action_filename = f'Y_{counter}.npy'
        intention_filename=f'I_{counter}.png'
        im = im.resize(size = (224, 224))
        im.save(image_dir + image_filename)
        np.save(action_dir + action_filename, action)
        if not plan:
            # take previous intention image
            # TODO: softlink instead of duplicating same image to save space
            # print(f"path: {intent_dir + intention_filename} ")
            prev_count = counter - 1
            # print(f"prev_count: {prev_count}")
            copyfile(intent_dir + f'I_{prev_count}.png', intent_dir + f'I_{counter}.png')
        counter += 1

show_animation = True #set

def dwa_control(x, config, goal, ob):
    """
    Dynamic Window Approach control
    """
    dw = calc_dynamic_window(x, config)

    u, trajectory = calc_control_and_trajectory(x, dw, config, goal, ob)

    return u, trajectory


class RobotType(Enum):
    circle = 0
    rectangle = 1


class Config:
    """
    simulation parameter class
    """

    def __init__(self):
        # robot parameter
        self.max_speed = 1.0  # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.2  # [m/ss]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.01  # [m/s]
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s] Time tick for motion prediction
        self.predict_time = 3.0  # [s]
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
        self.robot_type = RobotType.circle

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 1.0  # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.robot_width = ROBOT_WIDTH  # [m] for collision check
        self.robot_length = ROBOT_LENGTH  # [m] for collision check
        # obstacles [x(m) y(m), ....]
        ob_cood = []
        for tile in env.obstacle_tiles:
            # reflect about x-axis to fit duckietown's map
            # e.g. duckietown's (2,3) coordinate is actually (2, -3)
            (x, y) = tile['coords']
            # (x, -y) refer to starting coordinate of tile. tile is 1x1 grid from (x, -y).
            ob_cood.append((x, -y))

            # freq = 5 # I can't connect the dots. So i'm plotting %freq number of discrete dots to mimick a continuous line.
            # for i in range(freq):
            #     for j in range(freq):
            #         ob_cood.append((x + i/freq, -y - j/freq))
        self.ob = np.array(ob_cood)

    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value


config = Config()


def motion(x, u, dt):
    """
    motion model
    """

    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]

    return x


def calc_dynamic_window(x, config):
    """
    calculation dynamic window based on current state x
    """

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt, # -0.5 to reduce min speed which gives planner option to just rotate on the spot. allowing rotation mitigates the problem of ducky going out of the map.
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]

    #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
    # print(f"vs[0]: {Vs[0]}")
    # print(f"vd[0]: {Vd[0]}")
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
    # print(f"dw: {dw}")

    return dw


def predict_trajectory(x_init, v, y, config):
    """
    predict trajectory with an input
    """

    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt

    return trajectory


def calc_control_and_trajectory(x, dw, config, goal, ob):
    """
    calculation final input with dynamic window
    """

    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])

    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):

            trajectory = predict_trajectory(x_init, v, y, config)
            # calc cost
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)

            final_cost = to_goal_cost + speed_cost + ob_cost

            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = trajectory
                if abs(best_u[0]) < config.robot_stuck_flag_cons \
                        and abs(x[3]) < config.robot_stuck_flag_cons:
                    # to ensure the robot do not get stuck in
                    # best v=0 m/s (in front of an obstacle) and
                    # best omega=0 rad/s (heading to the goal with
                    # angle difference of 0)
                    best_u[1] = -config.max_delta_yaw_rate
    return best_u, best_trajectory


def calc_obstacle_cost(trajectory, ob, config):
    """
    calc obstacle cost inf: collision
    """
    ox = ob[:, 0]
    oy = ob[:, 1]
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]
    r = np.hypot(dx, dy)

    if config.robot_type == RobotType.rectangle:
        yaw = trajectory[:, 2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])
        local_ob = ob[:, None] - trajectory[:, 0:2]
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        local_ob = np.array([local_ob @ x for x in rot])
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        upper_check = local_ob[:, 0] <= config.robot_length / 2
        right_check = local_ob[:, 1] <= config.robot_width / 2
        bottom_check = local_ob[:, 0] >= -config.robot_length / 2
        left_check = local_ob[:, 1] >= -config.robot_width / 2
        if (np.logical_and(np.logical_and(upper_check, right_check),
                           np.logical_and(bottom_check, left_check))).any():
            return float("Inf")
    elif config.robot_type == RobotType.circle:
        if np.array(r <= config.robot_radius).any():
            return float("Inf")

    min_r = np.min(r)
    return 1.0 / min_r  # OK


def calc_to_goal_cost(trajectory, goal):
    """
        calc to goal cost with angle difference
    """

    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1, 2]
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    return cost


def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)


def plot_robot(x, y, yaw, config):  # pragma: no cover
    if config.robot_type == RobotType.rectangle:
        outline = np.array([[-config.robot_length / 2, config.robot_length / 2,
                             (config.robot_length / 2), -config.robot_length / 2,
                             -config.robot_length / 2],
                            [config.robot_width / 2, config.robot_width / 2,
                             - config.robot_width / 2, -config.robot_width / 2,
                             config.robot_width / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                         [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), "-k")
    elif config.robot_type == RobotType.circle:
        circle = plt.Circle((x, y), config.robot_radius, color="b")
        plt.gcf().gca().add_artist(circle)
        out_x, out_y = (np.array([x, y]) +
                        np.array([np.cos(yaw), np.sin(yaw)]) * config.robot_radius)
        plt.plot([x, out_x], [y, out_y], "-k")


def dwa(gx=3.5, gy=-3.5, robot_type=RobotType.rectangle):
    """
    Dynamic Window Approach. Plots the intention image. If successful, trajectory will be shown. Else, only map will be shown.
    
    Credits: https://github.com/larrylawl/PythonRobotics/tree/master/PathPlanning/DynamicWindowApproach
    """

    print(__file__ + " start!!")
    info = env.get_agent_info()['Simulator']
    # print(f'Agent Info: {info}')
    px, _, pz = info['cur_pos'] 
    pz *= -1 # flip y-axis to fit duckytown's coordinates 
    yaw = env.cur_angle
    v = info['robot_speed']
    # Formula for angular velocity: http://www.cs.columbia.edu/~allen/F17/NOTES/icckinematics.pdf
    Vl, Vr = info['wheel_velocities']
    l = env.wheel_dist
    w = (Vr - Vl) / l
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega/angular velocity(rad/s)]
    x = np.array([px, pz, yaw, v, w])
    # print(f"[px, pz, yaw, v, w]: {x}")
    # goal position [x(m), y(m)]
    goal = np.array([gx, gy])

    # input [forward speed, yaw_rate]

    config.robot_type = robot_type
    trajectory = np.array(x)
    ob = config.ob

    # If dist_to_goal does not improve after %early_stopping_threshold number of steps, breaks out of planning.
    early_stopping_threshold = 20
    early_stopping_counter = 0
    best_dist_to_goal = 99999
    initial_dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
    while True:
        u, predicted_trajectory = dwa_control(x, config, goal, ob)
        x = motion(x, u, config.dt)  # simulate robot
        trajectory = np.vstack((trajectory, x))  # store state history

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
            plt.plot(x[0], x[1], "xr")
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(ob[:, 0], ob[:, 1], "ok")
            plot_robot(x[0], x[1], x[2], config)
            plot_arrow(x[0], x[1], x[2])
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

        # check reaching goal
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        if dist_to_goal + 0.001 < best_dist_to_goal:
            early_stopping_counter = 0
            best_dist_to_goal = dist_to_goal
        else:
            early_stopping_counter += 1
            if early_stopping_counter == early_stopping_threshold:
                print("Failed :(")
                break

        if dist_to_goal <= config.robot_radius:
            print("Goal!!")
            break

    # if show_animation:
    #     plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
    #     plt.pause(0.0001)
    if args.save:
        if best_dist_to_goal < initial_dist_to_goal: # good plan saves trajectory
            plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
        else: # failed plan returns empty map
            plt.cla()
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(ob[:, 0], ob[:, 1], "ok")
        F = plt.gcf()
        F.set_size_inches(1.12, 1.12, forward = True) # inet img input dimensions in inches
        plt.savefig(intent_dir + f'I_{counter}.png')
    plt.show()

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
