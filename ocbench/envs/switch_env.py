import mujoco
import numpy as np
from dm_control import mjcf

from ocbench.envs.manipulation_env import ManipulationEnv


class SwitchEnv(ManipulationEnv):
    def __init__(
        self,
        env_type='5x5',
        task_id=1,
        **kwargs,
    ):
        self._env_type = env_type
        self._num_button_colors = 2
        self._dropped_button_state = 2
        self._num_button_states = 3
        self._button_spacing = 0.1
        self._button_center = np.array([0.425, 0.0])

        if env_type in {'3x3', '4x4', '5x5'}:
            self._num_rows, self._num_cols = [int(x) for x in env_type.split('x')]
        else:
            raise ValueError(f'Unknown env_type: {env_type}')

        self._num_buttons = self._num_rows * self._num_cols
        self._cur_button_states = np.zeros(self._num_buttons, dtype=np.int64)

        super().__init__(
            task_id=task_id,
            **kwargs,
        )

        half_x = self._button_spacing * (self._num_rows - 1) / 2
        half_y = self._button_spacing * (self._num_cols - 1) / 2
        self._target_sampling_bounds = np.asarray(
            [
                [self._button_center[0] - half_x, self._button_center[1] - half_y],
                [self._button_center[0] + half_x, self._button_center[1] + half_y],
            ]
        )

    def set_state(self, qpos, qvel, button_states):
        self._cur_button_states = np.rint(button_states).astype(np.int64)
        self._apply_button_states()
        super().set_state(qpos, qvel)
        self._prev_button_states = self._cur_button_states.copy()

    def set_tasks(self):
        self.task_infos = [
            dict(
                task_name='monochrome',
                max_dropped_buttons=0,
            ),
            dict(
                task_name='monochrome_drop',
                max_dropped_buttons=1 if self._env_type == '3x3' else 2,
            ),
        ]

    def _toggle_neighbors(self, button_states, button):
        if button_states[button] == self._dropped_button_state:
            return
        x, y = button // self._num_cols, button % self._num_cols
        for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self._num_rows and 0 <= ny < self._num_cols:
                idx = nx * self._num_cols + ny
                if button_states[idx] != self._dropped_button_state:
                    button_states[idx] = (button_states[idx] + 1) % self._num_button_colors

    def _sample_dropped_buttons(self):
        dropped_buttons = np.zeros(self._num_buttons, dtype=bool)
        max_dropped_buttons = self.cur_task_info['max_dropped_buttons']
        if max_dropped_buttons == 0:
            return dropped_buttons

        num_singletons = self._num_buttons
        if max_dropped_buttons == 1:
            choice = int(self.np_random.integers(1 + num_singletons))
            if choice > 0:
                dropped_buttons[choice - 1] = True
            return dropped_buttons

        num_pairs = self._num_buttons * (self._num_buttons - 1) // 2
        choice = int(self.np_random.integers(1 + num_singletons + num_pairs))
        if choice == 0:
            return dropped_buttons
        if choice <= num_singletons:
            dropped_buttons[choice - 1] = True
            return dropped_buttons

        pair_idx = choice - num_singletons - 1
        rows, cols = np.triu_indices(self._num_buttons, k=1)
        dropped_buttons[rows[pair_idx]] = True
        dropped_buttons[cols[pair_idx]] = True
        return dropped_buttons

    def _sample_initial_button_states(self, dropped_buttons):
        while True:
            button_states = np.full(
                self._num_buttons,
                self.np_random.integers(self._num_button_colors),
                dtype=np.int64,
            )
            button_states[dropped_buttons] = self._dropped_button_state
            press_mask = self.np_random.integers(self._num_button_colors, size=self._num_buttons)
            press_mask[dropped_buttons] = 0
            for button, pressed in enumerate(press_mask):
                if pressed:
                    self._toggle_neighbors(button_states, button)
            if not self._is_monochrome(button_states):
                return button_states

    def _is_monochrome(self, button_states=None):
        if button_states is None:
            button_states = self._cur_button_states
        active_button_states = button_states[button_states != self._dropped_button_state]
        return bool(np.all(active_button_states == 0) or np.all(active_button_states == 1))

    def add_objects(self, arena_mjcf):
        # Add button scene.
        button_outer_mjcf = mjcf.from_path((self._desc_dir / 'button_outer.xml').as_posix())
        arena_mjcf.include_copy(button_outer_mjcf)

        # Add buttons to the scene.
        distance = self._button_spacing / 2
        for i in range(self._num_rows):
            for j in range(self._num_cols):
                button_mjcf = mjcf.from_path((self._desc_dir / 'button_inner.xml').as_posix())
                button_mjcf.find('body', 'buttonbox_0').mocap = True
                pos_x = self._button_center[0] - distance * (self._num_rows - 1) + 2 * distance * i
                pos_y = self._button_center[1] - distance * (self._num_cols - 1) + 2 * distance * j
                button_mjcf.find('body', 'buttonbox_0').pos[:2] = np.array([pos_x, pos_y])
                for tag in ['body', 'joint', 'geom', 'site']:
                    for item in button_mjcf.find_all(tag):
                        if hasattr(item, 'name') and item.name is not None and item.name.endswith('_0'):
                            item.name = item.name[:-2] + f'_{i * self._num_cols + j}'
                arena_mjcf.include_copy(button_mjcf)
                # Exclude button-base/floor collisions.
                arena_mjcf.contact.add('exclude', body1='world', body2=f'buttonbox_{i * self._num_cols + j}')

        # Add cameras.
        cameras = {
            'front': {
                'pos': (1.139, 0.000, 0.821),
                'xyaxes': (0.000, 1.000, 0.000, -0.643, 0.000, 0.766),
            },
            'front_pixels': {
                'pos': (0.905, 0.000, 0.762),
                'xyaxes': (0.000, 1.000, 0.000, -0.771, 0.000, 0.637),
            },
            'side': {
                'pos': (0.721, -0.725, 0.821),
                'xyaxes': (0.866, 0.500, -0.000, -0.314, 0.543, 0.779),
            },
        }
        for camera_name, camera_kwargs in cameras.items():
            arena_mjcf.worldbody.add('camera', name=camera_name, **camera_kwargs)

    def post_compilation_objects(self):
        # Button geom IDs.
        self._button_geom_ids_list = []
        self._button_base_body_ids = []
        for i in range(self._num_buttons):
            self._button_base_body_ids.append(self._model.body(f'buttonbox_{i}').id)
            body_ids = np.array(
                [
                    self._button_base_body_ids[-1],
                    self._model.body(f'button_{i}').id,
                ]
            )
            geom_ids = np.flatnonzero(np.isin(self._model.geom_bodyid, body_ids)).astype(int).tolist()
            self._button_geom_ids_list.append(geom_ids)
        self._button_top_geom_ids = [self._model.geom(f'btngeom_{i}').id for i in range(self._num_buttons)]
        self._button_geom_rgba_list = [
            [self._model.geom_rgba[gid].copy() for gid in button_geom_ids]
            for button_geom_ids in self._button_geom_ids_list
        ]
        self._button_geom_contype_list = [
            [self._model.geom_contype[gid].copy() for gid in button_geom_ids]
            for button_geom_ids in self._button_geom_ids_list
        ]
        self._button_geom_conaffinity_list = [
            [self._model.geom_conaffinity[gid].copy() for gid in button_geom_ids]
            for button_geom_ids in self._button_geom_ids_list
        ]
        self._button_site_ids = [self._model.site(f'btntop_{i}').id for i in range(self._num_buttons)]
        self._button_base_mocap_ids = np.asarray(
            [self._model.body_mocapid[body_id] for body_id in self._button_base_body_ids],
            dtype=np.int32,
        )
        self._nominal_button_base_poss = self._data.mocap_pos[self._button_base_mocap_ids].copy()
        self._nominal_button_base_quats = self._data.mocap_quat[self._button_base_mocap_ids].copy()

    def _apply_button_states(self):
        # Adjust button colors based on the current state.
        for i in range(self._num_buttons):
            for j, gid in enumerate(self._button_geom_ids_list[i]):
                self._model.geom_rgba[gid] = self._button_geom_rgba_list[i][j]
                self._model.geom_contype[gid] = self._button_geom_contype_list[i][j]
                self._model.geom_conaffinity[gid] = self._button_geom_conaffinity_list[i][j]

            if self._cur_button_states[i] == self._dropped_button_state:
                for gid in self._button_geom_ids_list[i]:
                    self._model.geom_rgba[gid, 3] = 0.0
                    self._model.geom_contype[gid] = 0
                    self._model.geom_conaffinity[gid] = 0
            else:
                color_zero = self._colors['blue']
                color_one = self._colors['darkgray']
                self._model.geom_rgba[self._button_top_geom_ids[i]] = (
                    color_zero if self._cur_button_states[i] == 0 else color_one
                )

        mujoco.mj_forward(self._model, self._data)

    def initialize_episode(self):
        self._data.qpos[self._arm_joint_ids] = self._home_qpos
        mujoco.mj_kinematics(self._model, self._data)

        dropped_buttons = self._sample_dropped_buttons()
        goal_button_states = np.zeros(self._num_buttons, dtype=np.int64)
        goal_button_states[dropped_buttons] = self._dropped_button_state

        self.initialize_arm()
        self._cur_button_states = self._sample_initial_button_states(dropped_buttons)
        self._apply_button_states()

        # Forward kinematics to update site positions.
        self.pre_step()
        mujoco.mj_forward(self._model, self._data)
        self.post_step()

        self._success = False

    def pre_step(self):
        self._prev_button_states = self._cur_button_states.copy()
        super().pre_step()

    def post_step(self):
        # Update button states.
        for i in range(self._num_buttons):
            if self._cur_button_states[i] == self._dropped_button_state:
                continue
            prev_joint_pos = self._prev_ob_info[f'privileged/button_{i}_pos'][0]
            cur_joint_pos = self._data.joint(f'buttonbox_joint_{i}').qpos.copy()[0]
            if prev_joint_pos > -0.02 and cur_joint_pos <= -0.02:
                self._toggle_neighbors(self._cur_button_states, i)
        self._apply_button_states()

        # Evaluate success.
        self._success = self._is_monochrome()

    def add_object_info(self, ob_info):
        # Button states.
        for i in range(self._num_buttons):
            button_pos = self._data.joint(f'buttonbox_joint_{i}').qpos.copy()
            button_vel = self._data.joint(f'buttonbox_joint_{i}').qvel.copy()
            if self._cur_button_states[i] == self._dropped_button_state:
                button_pos[:] = 0.0
                button_vel[:] = 0.0
            ob_info[f'privileged/button_{i}_state'] = self._cur_button_states[i]
            ob_info[f'privileged/button_{i}_pos'] = button_pos
            ob_info[f'privileged/button_{i}_vel'] = button_vel

        ob_info['prev_button_states'] = self._prev_button_states.copy()
        ob_info['button_states'] = self._cur_button_states.copy()

    def compute_observation(self):
        if self._ob_type == 'pixels':
            return self.get_pixel_observation()
        else:
            xyz_center = np.array([0.425, 0.0, 0.0])
            xyz_scaler = 10.0
            gripper_scaler = 3.0
            button_scaler = 120.0

            ob_info = self.compute_ob_info()
            ob = [
                ob_info['proprio/joint_pos'],
                ob_info['proprio/joint_vel'],
                (ob_info['proprio/effector_pos'] - xyz_center) * xyz_scaler,
                np.cos(ob_info['proprio/effector_yaw']),
                np.sin(ob_info['proprio/effector_yaw']),
                ob_info['proprio/gripper_opening'] * gripper_scaler,
                ob_info['proprio/gripper_contact'],
            ]
            for i in range(self._num_buttons):
                button_state = np.eye(self._num_button_states)[self._cur_button_states[i]]
                ob.extend(
                    [
                        button_state,
                        ob_info[f'privileged/button_{i}_pos'] * button_scaler,
                        ob_info[f'privileged/button_{i}_vel'],
                    ]
                )

            return np.concatenate(ob)
