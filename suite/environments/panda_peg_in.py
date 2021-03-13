from collections import OrderedDict
import random
import numpy as np

import suite.utils.transform_utils as T
from suite.utils.mjcf_utils import string_to_array
from suite.environments.panda import PandaEnv

from suite.models.arenas import Table
from suite.models.objects import  PlateWithRoundHoleObject

from suite.models.robots import Panda
from suite.models.tasks import PegInTask, UniformRandomSampler

import hjson
import os
import math

class PandaPegIn(PandaEnv):
    def __init__(
            self,
            gripper_type=None,
            table_full_size=(0.39, 0.49, 0.82),
            table_friction=(1, 0.005, 0.0001),
            use_camera_obs=True,
            use_object_obs=False,
            reward_shaping=False,
            placement_initializer=None,
            gripper_visualization=False,
            use_indicator_object=False,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_collision_mesh=False,
            render_visual_mesh=True,
            control_freq=10,
            horizon=1000,
            ignore_done=False,
            camera_name="frontview",
            camera_height=256,
            camera_width=256,
            camera_depth=False,
            controller='position',
            **kwargs
    ):
        """
        Args:

            gripper_type (str): type of gripper, used to instantiate
                gripper models from gripper factory.

            table_full_size (3-tuple): x, y, and z dimensions of the table.

            table_friction (3-tuple): the three mujoco friction parameters for
                the table.

            use_camera_obs (bool): if True, every observation includes a
                rendered image.

            use_object_obs (bool): if True, include object (cube) information in
                the observation.

            reward_shaping (bool): if True, use dense rewards.

            placement_initializer (ObjectPositionSampler instance): if provided, will
                be used to place objects on every reset, else a UniformRandomSampler
                is used by default.

            single_object_mode (int): specifies which version of the task to do. Note that
                the observations change accordingly.

                0: corresponds to the full task with all types of objects.

                1: corresponds to an easier task with only one type of object initialized
                   on the table with every reset. The type is randomized on every reset.

                2: corresponds to an easier task with only one type of object initialized
                   on the table with every reset. The type is kept constant and will not
                   change between resets.

            gripper_visualization (bool): True if using gripper visualization.
                Useful for teleoperation.

            use_indicator_object (bool): if True, sets up an indicator object that
                is useful for debugging.

            has_renderer (bool): If true, render the simulation state in
                a viewer instead of headless mode.

            has_offscreen_renderer (bool): True if using off-screen rendering.

            render_collision_mesh (bool): True if rendering collision meshes
                in camera. False otherwise.

            render_visual_mesh (bool): True if rendering visual meshes
                in camera. False otherwise.

            control_freq (float): how many control signals to receive
                in every second. This sets the amount of simulation time
                that passes between every action input.

            horizon (int): Every episode lasts for exactly @horizon timesteps.

            ignore_done (bool): True if never terminating the environment (ignore @horizon).

            camera_name (str): name of camera to be rendered. Must be
                set if @use_camera_obs is True.

            camera_height (int): height of camera frame.

            camera_width (int): width of camera frame.

            camera_depth (bool): True if rendering RGB-D, and RGB otherwise.

            use_default_task_config (bool): True if using default configuration file
                for remaining environment parameters. Default is true

            task_config_file (str): filepath to configuration file to be
                used for remaining environment parameters (taken relative to head of robosuite repo).

            use_default_controller_config (bool): True if using default configuration file
                for remaining environment parameters. Default is true

            controller_config_file (str): filepath to configuration file to be
                used for remaining environment parameters (taken relative to head of robosuite repo).

            controller (str): Can be 'position', 'position_orientation', 'joint_velocity', 'joint_impedance', or
                'joint_torque'. Specifies the type of controller to be used for dynamic trajectories

            controller_config_file (str): filepath to the corresponding controller config file that contains the
                associated controller parameters

            #########
            **kwargs includes additional params that may be specified and will override values found in
            the configuration files
        """

        # Load the parameter configuration files
        controller_filepath = os.path.join(os.path.dirname(__file__), '..',
                                               'scripts/config/controller_config.hjson')
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        # whether to show visual aid about where is the gripper
        self.gripper_visualization = gripper_visualization

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        super().__init__(
            gripper_type=gripper_type,
            gripper_visualization=gripper_visualization,
            use_indicator_object=use_indicator_object,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            use_camera_obs=use_camera_obs,
            camera_name=camera_name,
            camera_height=camera_height,
            camera_width=camera_width,
            camera_depth=camera_depth,
            controller_config_file=controller_filepath,
            controller=controller,
            **kwargs
        )

        # reward configuration
        self.reward_shaping = reward_shaping

        # # information of objects
        # self.object_names = list(self.mujoco_objects.keys())
        # self.object_site_ids = [
        #     self.sim.model.site_name2id(ob_name) for ob_name in self.object_names
        # ]

        # id of grippers for contact checking
        # self.finger_names = self.gripper.contact_geoms()

        # self.sim.data.contact # list, geom1, geom2
        self.collision_check_geom_names = self.sim.model._geom_name2id.keys()
        self.collision_check_geom_ids = [
            self.sim.model._geom_name2id[k] for k in self.collision_check_geom_names
        ]

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # load model for table top workspace
        self.mujoco_arena = Table(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )

        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # The panda robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.5, -0.15, 0])

        # task includes arena, robot, and objects of interest
        self.model = PegInTask(
            self.mujoco_arena,
            self.mujoco_robot,
        )
        # self.model.place_objects()
        self.table_size = self.model.table_size

    def _get_reference(self):
        super()._get_reference()
        self.us_pos = self.sim.data.site_xpos[self.sim.model.site_name2id("underside")]


        # self.obj_body_id = {}
        # self.obj_geom_id = {}

        # self.l_finger_geom_ids = [
        #     self.sim.model.geom_name2id(x) for x in self.gripper.left_finger_geoms
        # ]
        # self.r_finger_geom_ids = [
        #     self.sim.model.geom_name2id(x) for x in self.gripper.right_finger_geoms
        # ]

        # #得到object标识
        # for i in range(len(self.ob_inits)):
        #     obj_str = str(self.item_names[i]) + "0"
        #     self.obj_body_id[obj_str] = self.sim.model.body_name2id(obj_str)
        #     self.obj_geom_id[obj_str] = self.sim.model.geom_name2id(obj_str)

        # # for checking distance to / contact with objects we want to pick up
        # self.target_object_body_ids = list(map(int, self.obj_body_id.values()))
        # self.contact_with_object_geom_ids = list(map(int, self.obj_geom_id.values()))

        # # keep track of which objects are in their corresponding bins
        # self.objects_in_bins = np.zeros(len(self.ob_inits))

        # # target locations in bin for each object type
        # self.target_bin_placements = np.zeros((len(self.ob_inits), 3))
        # for j in range(len(self.ob_inits)):
        #     bin_id = j
        #     bin_x_low = self.bin_pos[0]
        #     bin_y_low = self.bin_pos[1]
        #     if bin_id == 0 or bin_id == 2:
        #         bin_x_low -= self.bin_size[0] / 2.
        #     if bin_id < 2:
        #         bin_y_low -= self.bin_size[1] / 2.
        #     bin_x_low += self.bin_size[0] / 4.
        #     bin_y_low += self.bin_size[1] / 4.
        #     self.target_bin_placements[j, :] = [bin_x_low, bin_y_low, self.bin_pos[2]]

    def _reset_internal(self):
        super()._reset_internal()

        # reset positions of objects, and move objects out of the scene depending on the mode
        # self.model.place_objects()

    def reward(self, action=None):
        # compute rewards
        # self.pre_reward = self.curr_reward #与return 连用
        s_max = 0.06
        sxy_max = 0.005
        delta_z = 0.005

        self.ep_pos = self.sim.data.site_xpos[self.sim.model.site_name2id("endpoint")]
        s = np.linalg.norm(self.ep_pos - (self.us_pos + [0, 0, 0.06]))#柱塞末端与洞口的三维空间距离
        sxy = np.linalg.norm(self.ep_pos[0: 2] - self.us_pos[0: 2])#柱塞末端与洞口的水平距离
        sz =self.ep_pos[2] - (self.us_pos[2] + 0.06)#柱塞末端与洞口的垂直距离
        # print(self.ep_pos - (self.us_pos + [0, 0, 0.06]))
        #分阶回报
        # if s > s_max:#靠近阶段
        #     self.curr_reward = 2 - math.tanh(10 * s) - math.tanh(10 * sxy)
        # elif sxy > sxy_max or sz > 3 * delta_z:#对齐阶段
        #     self.curr_reward = 2 - 5 * sxy - 5 * sz
        # elif self.ep_pos[2] > (self.us_pos[2] + delta_z):#插入阶段
        #     # print("完成对齐")
        #     self.curr_reward = 4 - 2 * (sz / 0.06)
        # else:#完成阶段
        #     print("完成")
        #     self.curr_reward = 10

        if sxy < sxy_max:   #对齐阶段
            if sz < 0:    #插入阶段
                print("finish alignment")
                if self.ep_pos[2] < (self.us_pos[2] + delta_z):   #完成阶段
                    print("finish")
                    self.curr_reward = 10
                else:
                    self.curr_reward = 4 - 2 * (sz / 0.06)
            else:
                self.curr_reward = 2 - 5 * sxy - 5 * sz
        else:   #靠近阶段
            self.curr_reward = 2 - math.tanh(10 * s) - math.tanh(10 * sxy)

        # return (self.curr_reward - self.pre_reward)
        return self.curr_reward

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:
            robot-state: contains robot-centric information.
            object-state: requires @self.use_object_obs to be True.
                contains object-centric information.
            image: requires @self.use_camera_obs to be True.
                contains a rendered frame from the simulation.
            depth: requires @self.use_camera_obs and @self.camera_depth to be True.
                contains a rendered depth map from the simulation
        """
        di = super()._get_observation()
        if self.use_camera_obs:
            camera_obs = self.sim.render(
                camera_name=self.camera_name,
                width=self.camera_width,
                height=self.camera_height,
                depth=self.camera_depth,
            )
            if self.camera_depth:
                di["image"], di["depth"] = camera_obs
            else:
                di["image"] = camera_obs

        # # low-level object information
        # if self.use_object_obs:

        #     # remember the keys to collect into object info
        #     object_state_keys = []

        #     # for conversion to relative gripper frame
        #     gripper_pose = T.pose2mat((di["eef_pos"], di["eef_quat"]))
        #     world_pose_in_gripper = T.pose_inv(gripper_pose)

        #     for i in range(len(self.item_names_org)):

        #         obj_str = str(self.item_names_org[i]) + "0"
        #         obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj_str]])
        #         obj_quat = T.convert_quat(
        #             self.sim.data.body_xquat[self.obj_body_id[obj_str]], to="xyzw"
        #         )
        #         di["{}_pos".format(obj_str)] = obj_pos
        #         di["{}_quat".format(obj_str)] = obj_quat

        #         # get relative pose of object in gripper frame
        #         object_pose = T.pose2mat((obj_pos, obj_quat))
        #         rel_pose = T.pose_in_A_to_pose_in_B(object_pose, world_pose_in_gripper)
        #         rel_pos, rel_quat = T.mat2pose(rel_pose)
        #         di["{}_to_eef_pos".format(obj_str)] = rel_pos
        #         di["{}_to_eef_quat".format(obj_str)] = rel_quat

        #         object_state_keys.append("{}_pos".format(obj_str))
        #         object_state_keys.append("{}_quat".format(obj_str))
        #         object_state_keys.append("{}_to_eef_pos".format(obj_str))
        #         object_state_keys.append("{}_to_eef_quat".format(obj_str))

        #     di["object-state"] = np.concatenate([di[k] for k in object_state_keys])

        return di

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            if (
                    self.sim.model.geom_id2name(contact.geom1) in self.finger_names
                    or self.sim.model.geom_id2name(contact.geom2) in self.finger_names
            ):
                collision = True
                break
        return collision

    def _gripper_visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """
        # color the gripper site appropriately based on distance to nearest object
        if self.gripper_visualization:
            # find closest object
            square_dist = lambda x: np.sum(
                np.square(x - self.sim.data.get_site_xpos("grip_site"))
            )
            dists = np.array(list(map(square_dist, self.sim.data.site_xpos)))
            dists[self.eef_site_id] = np.inf  # make sure we don't pick the same site
            dists[self.eef_cylinder_id] = np.inf
            ob_dists = dists[
                self.object_site_ids
            ]  # filter out object sites we care about
            min_dist = np.min(ob_dists)
            ob_id = np.argmin(ob_dists)
            ob_name = self.object_names[ob_id]

            # set RGBA for the EEF site here
            max_dist = 0.1
            scaled = (1.0 - min(min_dist / max_dist, 1.)) ** 15
            rgba = np.zeros(4)
            rgba[0] = 1 - scaled
            rgba[1] = scaled
            rgba[3] = 0.5

            self.sim.model.site_rgba[self.eef_site_id] = rgba

