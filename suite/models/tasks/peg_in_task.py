from collections import OrderedDict
import numpy as np

from suite.models.tasks import Task
from suite.utils import RandomizationError
from suite.utils.mjcf_utils import new_joint, array_to_string, string_to_array


class PegInTask(Task):
    """
    Creates MJCF model of a pick-and-place task.

    A pick-and-place task consists of one robot picking objects from a bin
    and placing them into another bin. This class combines the robot, the
    arena, and the objects into a single MJCF model of the task.
    """

    def __init__(self, mujoco_arena, mujoco_robot):
        """
        Args:
            mujoco_arena: MJCF model of robot workspace
            mujoco_robot: MJCF model of robot model
            mujoco_objects: a list of MJCF models of physical objects
            visual_objects: a list of MJCF models of visual objects. Visual
                objects are excluded from physical computation, we use them to
                indicate the target destinations of the objects.
        """
        super().__init__()

        # temp: z-rotation
        self.z_rotation = True

        self.merge_arena(mujoco_arena)
        self.merge_robot(mujoco_robot)
        # self.merge_objects(mujoco_objects)

    def merge_robot(self, mujoco_robot):
        """Adds robot model to the MJCF model."""
        self.robot = mujoco_robot
        self.merge(mujoco_robot)

    def merge_arena(self, mujoco_arena):
        """Adds arena model to the MJCF model."""
        self.arena = mujoco_arena
        self.table_offset = mujoco_arena.table_top_abs
        self.table_size = mujoco_arena.table_full_size
        self.merge(mujoco_arena)

    def merge_objects(self, mujoco_objects):
        """Adds physical objects to the MJCF model."""
        self.n_objects = len(mujoco_objects)
        self.mujoco_objects = mujoco_objects
        self.objects = []  # xml manifestation
        self.max_horizontal_radius = 0
        for obj_name, obj_mjcf in mujoco_objects.items():
            self.merge_asset(obj_mjcf)
            # Load object
            obj = obj_mjcf.get_collision(name=obj_name, site=True)
            obj.append(new_joint(name=obj_name, type="free", damping="0.0005"))
            self.objects.append(obj)
            self.worldbody.append(obj)

            self.max_horizontal_radius = max(
                self.max_horizontal_radius, obj_mjcf.get_horizontal_radius()
            )

            
