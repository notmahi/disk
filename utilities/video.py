from os.path import join
import imageio
import os
import numpy as np
import sys
import itertools
import wandb

import utilities.utils as utils

from PIL import Image, ImageDraw, ImageFont
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

FONTPATH = "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf"


class VideoRecorder(object):
    def __init__(self, root_dir, height=252, width=252, camera_id=0, fps=5):
        self.save_dir = utils.make_dir(root_dir, "video") if root_dir else None
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []
        self.blank_frame = None

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled

    def record(self, env):
        if self.enabled:
            frame = env.render(
                mode="rgb_array",
                height=self.height,
                width=self.width,
                #    camera_id=self.camera_id
            )
            self.frames.append(frame)

            if self.blank_frame is None:
                self.blank_frame = np.zeros_like(frame)

    def record_blank(self, text=None):
        if text is not None and self.blank_frame is not None:
            # Update blank frame with the text
            self.blank_frame = self._draw_image_with_text(self.blank_frame.shape, text)
        empty_frames = self.fps
        self.frames.extend([self.blank_frame] * empty_frames)

    def save(self, file_name):
        if self.enabled:
            print(f"Saving {file_name}")
            path = os.path.join(self.save_dir, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)
            self._wandb_log_plot()

    def _wandb_log_plot(self):
        pass

    def _draw_image_with_text(self, shape_tuple, text):
        h, w = shape_tuple[:2]
        fmt = "RGBA"[: shape_tuple[2]]
        img = Image.new(fmt, (w, h), color="#000000")
        fnt = ImageFont.truetype(FONTPATH, size=36)
        canvas = ImageDraw.Draw(img)
        text_width, text_height = canvas.textsize(text, font=fnt)
        x_pos = int((w - text_width) / 2)
        y_pos = int((h - text_height) / 2)
        canvas.text((x_pos, y_pos), text, font=fnt, fill="#FFFFFF")
        return np.array(img)


class VideoRecorderWithStates(VideoRecorder):
    def __init__(
        self,
        root_dir,
        height=512,
        width=512,
        camera_id=0,
        fps=5,
        obs_transforms=None,
        num_transforms=2,
        max_ep_length=1010,
    ):
        super().__init__(root_dir, height, width, camera_id, fps)
        self.obs_transforms = obs_transforms
        num_combos = num_transforms * (num_transforms - 1)
        num_combos //= 2
        self.obses = np.empty((num_combos, max_ep_length, 2))
        self.num_combos = num_combos
        self.fig = None
        # TODO Mahi change this or figure this out
        self.size = 10

    def init(self, enabled=True):
        self.frames = []
        self.obs_frames = []
        self.current = 0
        self.old_trajectories = []
        self.old_trajectory_lengths = []
        self.old_trajectory_skill = []
        self.current_skill = 0
        self.cmap = get_cmap("rainbow")
        self.enabled = self.save_dir is not None and enabled

    def record(self, env, no_mujoco=True, make_fig=True):
        if self.enabled:
            if not no_mujoco:
                frame = env.render(
                    mode="rgb_array", height=self.height, width=self.width,
                )
                if frame.shape[-1] == 3:  # RGB image
                    frame = np.concatenate(
                        [frame, np.ones_like(frame[..., -1:])], axis=-1
                    )
                self.height = frame.shape[0]
            obs_frame = self.visualize(self.obs_transforms(env), make_fig=make_fig)

            frame_list = []
            if not no_mujoco:
                frame_list.append(frame)
            if make_fig:
                frame_list.append(obs_frame)
            assert len(frame_list), "At least one of mujoco or plot should be recorded"
            joint_frame = np.concatenate(frame_list, axis=1)
            self.frames.append(joint_frame)
            if self.blank_frame is None:
                self.blank_frame = np.zeros_like(joint_frame)

    def visualize(self, obses, make_fig=True):
        for i, (obs_1, obs_2) in enumerate(itertools.combinations(obses, r=2)):
            self.obses[i, self.current, 0] = obs_1
            self.obses[i, self.current, 1] = obs_2

        if not make_fig:
            return
        if self.current == 0:
            if self.fig is not None:
                plt.close()
            self.lineplots = []
            self.fig, self.axes = plt.subplots(nrows=1, ncols=self.num_combos)
            if self.num_combos == 1:
                # matplotlib has weird API, so need to wrap this in an array
                # for consistency
                self.axes = [self.axes]
            fig = self.fig
            self._inch_size = 2.56
            fig.set_size_inches(
                (self._inch_size * self.num_combos, self._inch_size)
            )  # width * height
            # plt.axis('off')
            # plt.xlim([-self.size, self.size])
            # plt.ylim([-self.size, self.size])

            # First, draw all the previous averages.
            # And make a color wheel.
            colors = [self.cmap(x) for x in np.linspace(0, 1, self.current_skill + 1)]
            for len_tr, traj, skill in zip(
                self.old_trajectory_lengths,
                self.old_trajectories,
                self.old_trajectory_skill,
            ):
                for i in range(self.num_combos):
                    self.axes[i].plot(
                        traj[i, :len_tr, 0],
                        traj[i, :len_tr, 1],
                        alpha=0.5,
                        linewidth=0.5,
                        color=colors[skill],
                    )
            # Then, draw all the current lines
            for i in range(self.num_combos):
                self.lineplots.append(
                    self.axes[i].plot(
                        self.obses[i, :1, 0],
                        self.obses[i, :1, 1],
                        linewidth=0.5,
                        color=colors[self.current_skill],
                    )[0]
                )

        else:
            # keep updating the lines with new data
            for i in range(self.num_combos):
                self.lineplots[i].set_data(
                    self.obses[i, : self.current + 1, 0],
                    self.obses[i, : self.current + 1, 1],
                )
        self.current += 1
        self.fig.set_dpi(self.height / self._inch_size)
        self.fig.canvas.draw()
        return np.array(self.fig.canvas.renderer._renderer)

    def record_blank(self, text=None, make_fig=True):
        if self.enabled:
            if self.current > 0:
                # time to move the previous obses to old_trajectories
                self.old_trajectories.append(self.obses.copy())
                self.old_trajectory_lengths.append(self.current)
                self.old_trajectory_skill.append(self.current_skill)

            # Now, figure out the saving of the old trajectories.
            self.current = 0

            if text is not None and self.blank_frame is not None:
                # Update blank frame with the text
                self.blank_frame = self._draw_image_with_text(
                    self.blank_frame.shape, text
                )
            empty_frames = self.fps // 2
            if make_fig and self.blank_frame is not None:
                self.frames.extend([self.blank_frame] * empty_frames)

    def init_new_skill(self):
        self.current_skill += 1
        self.current = 0

    def _wandb_log_plot(self):
        if self.fig:
            wandb.log(
                {
                    "all_trajectories": plt,
                    "all_traj_figure": [
                        wandb.Image(self.frames[-1], caption="Trajectories")
                    ],
                }
            )
