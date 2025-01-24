#!/usr/bin/env python3

from __future__ import annotations

import gymnasium as gym
import pygame
from gymnasium import Env

from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
import torch_ac
import numpy
import torch
import cv2
import time

import matplotlib.pyplot as plt

def get_state(env):
    return env.Current_state()

def preprocess_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    X=[]
    for i in range(len(images)):
        # print(images[i]['image'].shape)
        x = cv2.resize(images[i], (300, 300))
        X.append(x)

    images = numpy.array(X)
    return torch.tensor(images, device=device, dtype=torch.float)

def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                "image": preprocess_images(obss, device=device)
            })

class ManualControl:
    def __init__(
        self,
        env: Env,
        seed=None,
    ) -> None:
        self.env = env
        self.seed = seed
        self.closed = False 
        
        self.last_state_img = None
        self.current_state_img = None
        self.current_state = None

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)

        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    break
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)

    def step(self, action: Actions):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        print(f"step={self.env.step_count}, reward={reward:.2f}")
        print(env.carrying)
        # 如果这个状态和前面状态不一样，那么就是1，2，3，4
        # 若与前面状态相同，那么就是0
        if self.current_state != get_state(self.env):
            self.last_state_img = self.current_state_img
            self.current_state_img = preprocess_obss([obs['image']], device=None).image
            self.current_state = get_state(self.env)
            image = self.current_state_img - self.last_state_img
            image = numpy.squeeze(image)
            plt.imshow(image)
            # plt.show()
            timestamp = int(time.time())
            plt.imsave(f"../../test/dataset1/{self.current_state}/image_{timestamp}.png", image.cpu().numpy().astype(numpy.uint8))
        else:
            self.last_state_img = self.current_state_img
            self.current_state_img = preprocess_obss([obs['image']], device=None).image
            self.current_state = get_state(self.env)
            image = self.current_state_img - self.last_state_img
            image = numpy.squeeze(image)
            print(image.shape)
            plt.imshow(image)
            # plt.show()
            timestamp = int(time.time())
            print(image.cpu().numpy().astype(numpy.uint8))
            plt.imsave(f"../../test/dataset1/0/image_{timestamp}.png", image.cpu().numpy().astype(numpy.uint8))

        if terminated:
            print("terminated!")
            self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)
        else:
            self.env.render()

    def reset(self, seed=None):
        obs, _ = self.env.reset(seed=seed)
        self.last_state_img = preprocess_obss([obs['image']], device=None).image
        self.current_state_img = preprocess_obss([obs['image']], device=None).image
        self.current_state = get_state(self.env)
        self.env.render()

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.forward,
            "space": Actions.toggle,
            "pageup": Actions.pickup,
            "pagedown": Actions.drop,
            "tab": Actions.pickup,
            "left shift": Actions.drop,
            "enter": Actions.done,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        type=str,
        help="gym environment to load",
        choices=gym.envs.registry.keys(),
        # default="MiniGrid-MultiRoom-N6-v0",
        default="MiniGrid-ConfigWorld-v0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=None,
    )
    parser.add_argument(
        "--tile-size", type=int, help="size at which to render tiles", default=32
    )
    parser.add_argument(
        "--agent-view",
        action="store_true",
        default=False,
        help="draw the agent sees (partially observable view)",
    )
    parser.add_argument(
        "--agent-view-size",
        type=int,
        default=7,
        help="set the number of grid spaces visible in agent-view ",
    )
    parser.add_argument(
        "--screen-size",
        type=int,
        default="640",
        help="set the resolution for pygame rendering (width and height)",
    )

    args = parser.parse_args()
    #
    # from configparser import ConfigParser
    #
    # config = ConfigParser()
    # config.read('testmap.config', encoding='UTF-8')
    # string = config['map']['map_grid']
    # lines = string.split("\n")
    # map = []
    # for i in lines:
    #     i = i.replace(" ", "")
    #     map.append(i.split(","))

    env: MiniGridEnv = gym.make(
        args.env_id,
        tile_size=args.tile_size,
        render_mode="human",
        agent_pov=args.agent_view,
        agent_view_size=args.agent_view_size,
        screen_size=args.screen_size,
    )

    # TODO: check if this can be removed
    if args.agent_view:
        print("Using agent view")
        env = RGBImgPartialObsWrapper(env, args.tile_size)
        env = ImgObsWrapper(env)
    # print(gym.envs.registry.keys())


    manual_control = ManualControl(env, seed=args.seed)
    manual_control.start()
