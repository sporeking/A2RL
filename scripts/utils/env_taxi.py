
#5*5，不与乘客重合，无随机

from contextlib import closing
from io import StringIO
from os import path
from typing import Optional

import numpy as np

import gymnasium as gym
from gymnasium import Env, spaces, utils
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import DependencyNotInstalled
import pygame
import torch



MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]


WINDOW_SIZE = (550,350)

class TaxiEnv(Env):
        
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }
        
    def __init__(self, render_mode: Optional[str] = "rgb_array", start_state: Optional[int] = None, max_steps: int = 256):
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.desc = np.asarray(MAP, dtype="c")

        # Updated location coordinates and colors
        self.locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
        self.locs_colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255)]

        # Updated grid size and number of states
        num_states = 500
        num_rows = 5
        num_columns = 5
        max_row = num_rows - 1
        max_col = num_columns - 1
        num_states = num_rows * num_columns * (len(self.locs) + 1) * len(self.locs)

        self.initial_state_distrib = np.zeros(num_states)
        num_actions = 6
        a = 1.005

        # Transition probabilities
        fixed_states = [181]
        #self.P = {state: {action: [] for action in range(num_actions)} for state in fixed_states}
        self.P = {state: {action: [] for action in range(num_actions)} for state in range(num_states)}

        for row in range(num_rows):
            for col in range(num_columns):
                for pass_idx in range(len(self.locs) + 1):
                    for dest_idx in range(len(self.locs)):
                        state = self.encode(row, col, pass_idx, dest_idx)
                        if pass_idx < len(self.locs) and pass_idx != dest_idx:
                            self.initial_state_distrib[state] += 1
                        for action in range(num_actions):
                            new_row, new_col, new_pass_idx = row, col, pass_idx
                            reward = 0
                            terminated = False
                            taxi_loc = (row, col)

                            if action == 0:  # Move South
                                new_row = min(row + 1, max_row)
                            elif action == 1:  # Move North
                                new_row = max(row - 1, 0)
                            elif action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
                                new_col = min(col + 1, max_col)
                            elif action == 3 and self.desc[1 + row, 2 * col] == b":":
                                new_col = max(col - 1, 0)
                            elif action == 4:  # Pickup
                                # print(self.locs, taxi_loc)
                                if pass_idx < len(self.locs) and abs(taxi_loc[0] - self.locs[pass_idx][0]) + abs(taxi_loc[1] - self.locs[pass_idx][1]) == 1:
                                    new_pass_idx = len(self.locs)  # Pickup passenger
                                    reward = 0
                                else:
                                    reward = 0
                            elif action == 5:  # Dropoff
                                if taxi_loc == self.locs[dest_idx] and pass_idx == len(self.locs):
                                    new_pass_idx = dest_idx
                                    terminated = True
                                    # reward = 1 - (np.power(a, self.steps) - np.power(a, -self.steps)) / (np.power(a, self.steps) + np.power(a, -self.steps))
                                    reward = 1
                                elif taxi_loc in self.locs and pass_idx == len(self.locs):
                                    new_pass_idx = self.locs.index(taxi_loc)
                                    reward = -1

                            new_state = self.encode(new_row, new_col, new_pass_idx, dest_idx)
                            self.P[state][action].append((1.0, new_state, reward, terminated))

        self.initial_state_distrib /= self.initial_state_distrib.sum()
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Box(low=0, high=255, shape=(WINDOW_SIZE[1], WINDOW_SIZE[0], 3), dtype=np.uint8)
        self.start_state = start_state if start_state is not None else categorical_sample(self.initial_state_distrib, np.random)

        self.window = None
        self.clock = None
        self.cell_size = (WINDOW_SIZE[0] / self.desc.shape[1], WINDOW_SIZE[1] / self.desc.shape[0])
        self.taxi_imgs = None
        self.passenger_img = None
        self.destination_img = None
        self.background_img = None
        self.taxi_orientation = 0
        self.median_horiz = None
        self.median_vert = None
        self.background_img = None

    def encode(self, taxi_row, taxi_col, pass_loc, dest_idx):
        # (5) 5, 5, 4
        i = taxi_row
        i *= 5
        i += taxi_col
        i *= 5
        i += pass_loc
        i *= 4
        i += dest_idx
        return i

    def decode(self, i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        return reversed(out)


    def action_mask(self, state: int): 
        """Computes an action mask for the action space using the state information."""
        mask = np.zeros(6, dtype=np.int8)
        taxi_row, taxi_col, pass_loc, dest_idx = self.decode(state)
        if taxi_row < 4:
            mask[0] = 1
        if taxi_row > 0:
            mask[1] = 1
        if taxi_col < 4 and self.desc[taxi_row + 1, 2 * taxi_col + 2] == b":":
            mask[2] = 1
        if taxi_col > 0 and self.desc[taxi_row + 1, 2 * taxi_col] == b":":
            mask[3] = 1
        if pass_loc < 4 and (taxi_row, taxi_col+1) == self.locs[pass_loc]:
            mask[4] = 1
        if pass_loc < 4 and (taxi_row, taxi_col-1) == self.locs[pass_loc]:
            mask[4] = 1
        if pass_loc < 4 and (taxi_row+1, taxi_col) == self.locs[pass_loc]:
            mask[4] = 1
        if pass_loc < 4 and (taxi_row-1, taxi_col) == self.locs[pass_loc]:
            mask[4] = 1
        if pass_loc == 4 and (
            (taxi_row, taxi_col) == self.locs[dest_idx]
            or (taxi_row, taxi_col) in self.locs
        ):
            mask[5] = 1
        return mask

    def step(self, a):
        
        state = self.s.cpu().item() if isinstance(self.s, torch.Tensor) else self.s
        transitions = self.P[state][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a
        self.step_count += 1
        if r > 0:
            r = 1 - 0.9 * (self.step_count / self.max_steps)
        truncated = False
        if self.step_count >= self.max_steps:
            truncated = True
        # self.steps += 1

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self.get_observation(), r, t, truncated, {"prob": p, "action_mask": self.action_mask(s)}
    
    def gen_obs(self):
        return self.get_observation()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        state: Optional[int] = None  
    ):
        super().reset(seed=seed)
        self.step_count = 0
        if state is not None:
            self.s = state  
        else:
            #self.s = categorical_sample(self.initial_state_distrib, self.np_random)  
            self.s=181
        self.lastaction = None
        self.taxi_orientation = 0

        if self.render_mode == "human":
            self.render()
        return self.get_observation(), {"prob": 1.0, "action_mask": self.action_mask(self.s)}

    def get_observation(self):
        return self.render(mode="rgb_array")
    
    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            return self._render_gui(mode="rgb_array")
        elif mode == "human":
            return self._render_gui(mode="human")
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
        

    def _render_gui(self, mode):
        try:
            import pygame  # dependency to pygame only if rendering with human
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[toy-text]"`'
            ) from e

        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Taxi")
            if mode == "human":
                self.window = pygame.display.set_mode(WINDOW_SIZE)
            elif mode == "rgb_array":
                self.window = pygame.Surface(WINDOW_SIZE)

        assert (
            self.window is not None
        ), "Something went wrong with pygame. This should never happen."
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.taxi_imgs is None:
            file_names = [
                path.join(path.dirname(__file__), "img/cab_front.png"),
                path.join(path.dirname(__file__), "img/cab_rear.png"),
                path.join(path.dirname(__file__), "img/cab_right.png"),
                path.join(path.dirname(__file__), "img/cab_left.png"),
            ]
            self.taxi_imgs = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]
        if self.passenger_img is None:
            file_name = path.join(path.dirname(__file__), "img/passenger.png")
            self.passenger_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.destination_img is None:
            file_name = path.join(path.dirname(__file__), "img/hotel.png")
            self.destination_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
            self.destination_img.set_alpha(170)
        if self.median_horiz is None:
            file_names = [
                path.join(path.dirname(__file__), "img/gridworld_median_left.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_horiz.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_right.png"),
            ]
            self.median_horiz = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]
        if self.median_vert is None:
            file_names = [
                path.join(path.dirname(__file__), "img/gridworld_median_top.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_vert.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_bottom.png"),
            ]
            self.median_vert = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]
        if self.background_img is None:
            file_name = path.join(path.dirname(__file__), "img/taxi_background.png")
            self.background_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        desc = self.desc

        for y in range(0, desc.shape[0]):
            for x in range(0, desc.shape[1]):
                cell = (x * self.cell_size[0], y * self.cell_size[1])
                self.window.blit(self.background_img, cell)
                if desc[y][x] == b"|" and (y == 0 or desc[y - 1][x] != b"|"):
                    self.window.blit(self.median_vert[0], cell)
                elif desc[y][x] == b"|" and (
                    y == desc.shape[0] - 1 or desc[y + 1][x] != b"|"
                ):
                    self.window.blit(self.median_vert[2], cell)
                elif desc[y][x] == b"|":
                    self.window.blit(self.median_vert[1], cell)
                elif desc[y][x] == b"-" and (x == 0 or desc[y][x - 1] != b"-"):
                    self.window.blit(self.median_horiz[0], cell)
                elif desc[y][x] == b"-" and (
                    x == desc.shape[1] - 1 or desc[y][x + 1] != b"-"
                ):
                    self.window.blit(self.median_horiz[2], cell)
                elif desc[y][x] == b"-":
                    self.window.blit(self.median_horiz[1], cell)

        for cell, color in zip(self.locs, self.locs_colors):
            color_cell = pygame.Surface(self.cell_size)
            color_cell.set_alpha(128)
            color_cell.fill(color)
            loc = self.get_surf_loc(cell)
            self.window.blit(color_cell, (loc[0], loc[1] + 10))

        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)

        if pass_idx < 4:
            self.window.blit(self.passenger_img, self.get_surf_loc(self.locs[pass_idx]))

        if self.lastaction in [0, 1, 2, 3]:
            self.taxi_orientation = self.lastaction
        dest_loc = self.get_surf_loc(self.locs[dest_idx])
        taxi_location = self.get_surf_loc((taxi_row, taxi_col))

        if dest_loc[1] <= taxi_location[1]:
            self.window.blit(
                self.destination_img,
                (dest_loc[0], dest_loc[1] - self.cell_size[1] // 2),
            )
            self.window.blit(self.taxi_imgs[self.taxi_orientation], taxi_location)
        else:  # change blit order for overlapping appearance
            self.window.blit(self.taxi_imgs[self.taxi_orientation], taxi_location)
            self.window.blit(
                self.destination_img,
                (dest_loc[0], dest_loc[1] - self.cell_size[1] // 2),
            )

        if mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )

    def get_surf_loc(self, map_loc):
        return (map_loc[1] * 2 + 1) * self.cell_size[0], (
            map_loc[0] + 1
        ) * self.cell_size[1]

    def _render_text(self):
        desc = self.desc.copy().tolist()
        outfile = StringIO()

        out = [[c.decode("utf-8") for c in line] for line in desc]
        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)
        
        def ul(x):
            return "_" if x == " " else x

        if pass_idx < 4:
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[1 + taxi_row][2 * taxi_col + 1], "yellow", highlight=True
            )
            pi, pj = self.locs[pass_idx]
            out[1 + pi][2 * pj + 1] = utils.colorize(
                out[1 + pi][2 * pj + 1], "blue", bold=True
            )
        else:  # passenger in taxi
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                ul(out[1 + taxi_row][2 * taxi_col + 1]), "green", highlight=True
            )

        di, dj = self.locs[dest_idx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], "magenta")
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write(
                f"  ({['South', 'North', 'East', 'West', 'Pickup', 'Dropoff'][self.lastaction]})\n"
            )
        else:
            outfile.write("\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()


# Taxi rider from https://franuka.itch.io/rpg-asset-pack
# All other assets by Mel Tillery http://www.cyaneus.com/
