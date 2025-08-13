import numpy as np
import pygame
import math
import string
from .node_class import Node
from gymnasium.utils import EzPickle
from gymnasium.core import ObsType
from typing import Any
from simple_pid import PID

from pettingzoo.mpe._mpe_utils.core import World as BaseWorld
from pettingzoo.mpe._mpe_utils.core import Landmark as BaseLandmark
from pettingzoo.mpe._mpe_utils.core import Agent as BaseAgent
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn


def raw_print_state(grid):
    """Print a 2D integer grid to stdout.

    Prints rows from top to bottom for quick inspection of the state before state aggregation.

    Args:
        grid (list[list[int]]): 2D grid of cells. Each cell is an integer representing the state of that cell.
    """
    print("The state is:")
    for r in range(len(grid)-1, -1, -1):
        row = grid[r]
        print(" ".join(f"{cell:2}" for cell in row))
        
def print_state(state):
    """Print the compact Q-state representation.

    The layout shows a 3x3 center region and four aggregated directional regions
    in the order top, left, right, bottom.
    The center cell represents the A* action direction.

    Args:
        state (Sequence[int]): Encoded state. First 9 entries are the center 3×3 region.
            Entries 9..12 encode the four aggregated regions.
    """
    print("=== Q-State Representation ===")
    
    # Extract segments
    center = state[:9]  # 3x3 fine grid
    top = state[9]
    left = state[10]
    right = state[11]
    bottom = state[12]
    
    # Print layout:
    #     [ T ]
    # [L] [Center 3x3] [R]
    #     [ B ]
    
    print(f"   Top: {top}")
    print()
    
    print("Center Grid:")
    for r in range(2, -1, -1):
        row = center[r*3:(r+1)*3]
        print("        " + " ".join(f"{v:2}" for v in row))
        
    print()
    print(f"Left: {left}       Right: {right}")
    print()
    print(f"  Bottom: {bottom}")
    print("==============================\n")
        
        
class Agent(BaseAgent):
    """Mobile agent with A* guidance, Q-learning state, and simple PID tracking.

    Attributes:
        a_star_old (list[tuple[float, float]]): Already reached A* waypoints.
        a_star_new (list[tuple[float, float]]): Remaining A* waypoints.
        goal_point (list[float, float]): Goal position in world coordinates.
        q_state (tuple[int, ...]): Current Q-learning state encoding.
        controller_x (PID): PID controller for x-axis velocity command.
        controller_y (PID): PID controller for y-axis velocity command.
        terminated (bool): Set when the agent encountered a terminal event.
        a_star_action (int | None): Last A*-aligned discrete action {0,1,2,3}.
        action_history (list[int]): Last few executed discrete actions.
        reward (float): Last assigned scalar reward.

    Args:
        Kp (float): Proportional gain for both PID controllers.
        Ki (float): Integral gain for both PID controllers.
        Kd (float): Derivative gain for both PID controllers.
    """
    def __init__(self, Kp = 10, Ki = 0, Kd = 0):
        super().__init__()
        self.a_star_old = []
        self.a_star_new = []
        self.goal_point = []
        size = 9
        self.q_state = [1] * (size * size)
        self.controller_x = PID(Kp, Ki, Kd, setpoint=0)
        self.controller_y = PID(Kp, Ki, Kd, setpoint=0)
        self.terminated =  False
        self.a_star_action = None
        self.action_history = []
        self.reward = 0
        self.trajectory = []


class Landmark(BaseLandmark):
    """Goal landmark.

    The landmark is non‑movable and used to check goal proximity.
    """
    def __init__(self):
        super().__init__()

    def is_collision(self, agent):
        """Return True if the agent overlaps the landmark.

        Uses Euclidean distance and the agent radius.

        Args:
            agent (Agent): Agent instance.

        Returns:
            bool: True if the agent is within its radius of the landmark center.
        """
        euclidian_dis = math.sqrt((self.state.p_pos[0] - agent.state.p_pos[0]) ** 2 + (agent.state.p_pos[1] - agent.state.p_pos[1]) ** 2)
        if euclidian_dis <= agent.size:
            return True
        else:
            return False

class RandomLandmark(BaseLandmark):
    """Rectangular landmark that can be used as a random obstacle in training. Currently not used in the environment.

    The rectangle is represented by center position and width × height in meters.
    """
    def __init__(self):
        """Initialize rectangle size."""
        super().__init__()
        self.size = np.array([0.2, 0.2])
    
    def is_collision(self, agent):
        """Return True if the agent is inside the rectangle.

        Args:
            agent (Agent): Agent instance.

        Returns:
            bool: True if the agent center lies within the rectangle bounds.
        """
        x_min = self.state.p_pos[0] - self.size[0] / 2
        x_max = self.state.p_pos[0] + self.size[0] / 2
        y_min = self.state.p_pos[1] - self.size[1] / 2
        y_max = self.state.p_pos[1] + self.size[1] / 2
    
        agent_x = agent.state.p_pos[0]
        agent_y = agent.state.p_pos[1]
    
        return x_min <= agent_x <= x_max and y_min <= agent_y <= y_max
    
    def get_distance(self, agent_x, agent_y):
        """Return absolute x and y offsets from the rectangle center.

        Args:
            agent_x (float): Agent x coordinate.
            agent_y (float): Agent y coordinate.

        Returns:
            tuple[float, float]: Absolute delta in x and y.
        """
        dis_x = self.state.p_pos[0]-agent_x
        dis_y = self.state.p_pos[1]-agent_y
        return abs(dis_x), abs(dis_y)
    
class RectLandmark(BaseLandmark):
    """Rectangular landmark used as a static obstacle in planning and runtime."""
    def __init__(self):
        super().__init__()
        self.size = np.array([0.2, 0.4])
    
    def is_collision(self, agent):
        """Return True if the agent overlaps the rectangle.

        Collision uses rectangle bounds expanded by circles around the rectanlgle corners.

        Args:
            agent (Agent): Agent instance.

        Returns:
            bool: True if the agent center intersects the expanded rectangle.
        """
        buffer = agent.size
        agent_x, agent_y = agent.state.p_pos
        center_x, center_y = self.state.p_pos

        half_width = self.size[0] / 2 + buffer
        half_height = self.size[1] / 2 + buffer

        # Bounding box check
        if center_x - half_width <= agent_x <= center_x + half_width and \
        center_y - half_height <= agent_y <= center_y + half_height:
            
            # Corner circles for leniency
            corner_radius = 0.01
            corners = [
                (center_x - half_width, center_y - half_height),
                (center_x + half_width, center_y - half_height),  
                (center_x - half_width, center_y + half_height),  
                (center_x + half_width, center_y + half_height),  
            ]
            
            for cx, cy in corners:
                dist = np.linalg.norm([agent_x - cx, agent_y - cy])
                if dist < corner_radius:
                    return False

            return True
        return False
    
class World(BaseWorld):
    """World with custom continuous dynamics and collision handling for mixed shapes."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    
    def integrate_state(self, p_force):
        """Integrate positions using applied forces and reset velocities.

        Args:
            p_force (list[np.ndarray | None]): Per-entity force vectors in world coordinates.
        """
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            
            if p_force[i] is not None:
                entity.state.p_pos += p_force[i] * self.dt
                
            entity.state.p_vel = np.zeros_like(entity.state.p_vel)
    
    
    def apply_action_force(self, p_force):
        """Map agent actions to planar forces.

        The first two entries of the action vector are interpreted as desired planar velocities.
        A global scale is applied when converting to forces.

        Args:
            p_force (list[np.ndarray | None]): Force accumulator.

        Returns:
            list[np.ndarray | None]: Updated force accumulator.
        """
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                action = agent.action.u  # [0, -vx, 0, -vy, 0]
                vx = action[0]
                vy = action[1]
                scale = 2.0
                p_force[i] = np.array([vx, vy])* scale
        return p_force
    
    def apply_environment_force(self, p_force):
        """Apply environment forces such as contacts. Currently a no-op.

        Stub that returns the input unchanged.

        Args:
            p_force (list[np.ndarray | None]): Force accumulator.

        Returns:
            list[np.ndarray | None]: Unchanged force accumulator.
        """
        return p_force
    
    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        """Compute contact forces and termination flags between two entities.

        Handles three cases: rectangle–rectangle, rectangle–disk, and disk–disk.
        Uses a soft penetration model with contact margin and sets termination flags
        on agents upon significant penetration.

        Args:
            entity_a: First entity.
            entity_b: Second entity.

        Returns:
            list[np.ndarray | None, np.ndarray | None]: Force on A and force on B.
        """
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if entity_a is entity_b:
            return [None, None]  # don't collide against itsel
        
        # Rectangle to Rectangle collision
        if isinstance(entity_a, (RectLandmark, RandomLandmark)) and isinstance(entity_b, (RectLandmark, RandomLandmark)):
            delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
            dist_x = np.abs(delta_pos[0])  
            dist_y = np.abs(delta_pos[1])  

            half_width_a = entity_a.size[0] / 2
            half_height_a = entity_a.size[1] / 2
            half_width_b = entity_b.size[0] / 2
            half_height_b = entity_b.size[1] / 2
            
            dist_min_x = half_width_a + half_width_b
            dist_min_y = half_height_a + half_height_b
            
            # softmax penetration
            k = self.contact_margin
            
            penetration_x = np.logaddexp(0, -(dist_x - dist_min_x) / k) * k
            penetration_y = np.logaddexp(0, -(dist_y - dist_min_y) / k) * k
            
            if penetration_x > 0.01 and penetration_y > 0.01:
                if penetration_x > penetration_y:
                    force_magnitude = self.contact_force * penetration_x
                    force_direction = np.array([np.sign(delta_pos[0]), 0])
                else:
                    force_magnitude = self.contact_force * penetration_y
                    force_direction = np.array([0, np.sign(delta_pos[1])])
                    
                force = force_magnitude * force_direction
                force_a = force if entity_a.movable else None
                force_b = -force if entity_b.movable else None
                return [force_a, force_b]
            return [None, None]
        
        # Rectangle to Non-Rectangle collision
        elif isinstance(entity_a, (RectLandmark, RandomLandmark)) or isinstance(entity_b, (RectLandmark, RandomLandmark)):
            if isinstance(entity_a, (RectLandmark, RandomLandmark)):
                rect_entity = entity_a
                non_rect_entity = entity_b
            else: 
                rect_entity = entity_b
                non_rect_entity = entity_a
                
            delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
            dist_x = np.abs(delta_pos[0])  
            dist_y = np.abs(delta_pos[1])

            half_width = rect_entity.size[0] / 2
            half_height = rect_entity.size[1] / 2
            
            dist_min_x = half_width + non_rect_entity.size
            dist_min_y = half_height + non_rect_entity.size
            
            # softmax penetration
            k = self.contact_margin
            penetration_x = np.logaddexp(0, -(dist_x - dist_min_x) / k) * k
            penetration_y = np.logaddexp(0, -(dist_y - dist_min_y) / k) * k
            
            if penetration_x > 0.01 and penetration_y > 0.01:
                non_rect_entity.terminated = True
                if penetration_x > penetration_y:
                    penetration = penetration_x
                    direction = np.array([np.sign(delta_pos[0]), 0])
                else:
                    penetration = penetration_y
                    direction = np.array([0, np.sign(delta_pos[1])])

                force = self.contact_force * direction * penetration
                force_rect = -force if rect_entity.movable else None
                force_non_rect = force if non_rect_entity.movable else None
                
                if isinstance(entity_a, (RectLandmark, RandomLandmark)):
                    return [force_rect, force_non_rect]
                else: 
                    return [force_non_rect, force_rect]
                
            return [None, None]
        
        # Non-Rectangle to Non-Rectangle collision
        elif not isinstance(entity_a, (RectLandmark, RandomLandmark)) and not isinstance(entity_b, (RectLandmark, RandomLandmark)):        
            # compute actual distance between entities
            delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            # minimum allowable distance
            dist_min = entity_a.size + entity_b.size
            # softmax penetration
            k = self.contact_margin
            penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
            if penetration > 0.01:
                entity_a.terminated = True
                entity_b.terminated = True
            force = self.contact_force * delta_pos / dist * penetration
            force_a = +force if entity_a.movable else None
            force_b = -force if entity_b.movable else None
            return [force_a, force_b]


class raw_env(SimpleEnv, EzPickle):
    """PettingZoo MPE environment with A*, local state encoding, and tabular Q-Learning."""
    def __init__(
        self,
        num_good=3,
        num_obstacles=4,
        max_cycles=100,
        continuous_actions=True,
        render_mode=None,
    ):
        """Create scenario, world, and base SimpleEnv.

        Args:
            num_good (int): Number of agents.
            num_obstacles (int): Number of rectangular landmarks.
            max_cycles (int): Step limit per episode.
            continuous_actions (bool): Use continuous action vectors.
            render_mode (str | None): None or "human".
        """
        EzPickle.__init__(
            self,
            num_good=num_good,
            num_obstacles=num_obstacles,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(num_good, num_obstacles)
        
        self.epsilon_runtime = scenario.epsilon_runtime
        self.epsilon_planning = scenario.epsilon_planning
        
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        self.metadata["name"] = "custom_environment"
        self.dynamic_agents = []
        size = 9
        self.state = [1] * (size * size)
        
    def draw(self):
        """Render entities, goals, and local grid overlay with pygame.

        Assumes an initialized pygame screen and viewport. Draws agents as disks,
        goals as rings, obstacles as rectangles, and the local grid around each agent.

        Raises:
            AssertionError: If screen coordinates compute out of bounds.
        """

        # clear screen
        self.screen.fill((255, 255, 255))

        # update bounds to center around agent
        all_poses = [entity.state.p_pos for entity in self.world.entities]
        cam_range = self.fixed_cam_range

        # update geometry and text positions
        text_line = 0
        for e, entity in enumerate(self.world.entities):
            # geometry
            x, y = entity.state.p_pos
            y *= (
                -1
            )  # this makes the display mimic the old pyglet setup (ie. flips image)
            x = (
                (x / cam_range) * self.width // 2 * 0.9
            )  # the .9 is just to keep entities from appearing "too" out-of-bounds
            y = (y / cam_range) * self.height // 2 * 0.9
            x += self.width // 2
            y += self.height // 2
            
            if isinstance(entity, RectLandmark):
                width, height = entity.size
                scale_factor = (self.width / (2 * cam_range)) * 0.9  # Match the scaling of positions
                rect_width = width * scale_factor
                rect_height = height * scale_factor
                pygame.draw.rect(
                    self.screen,
                    entity.color * 200,
                    pygame.Rect(
                        x - rect_width / 2,
                        y - rect_height / 2,
                        rect_width,
                        rect_height
                    )
                )
                pygame.draw.rect(
                    self.screen,
                    (0, 0, 0),  # Randfarbe (schwarz)
                    pygame.Rect(
                        x - rect_width / 2,
                        y - rect_height / 2,
                        rect_width,
                        rect_height
                    ), 
                    1  # Randdicke
                )
            elif isinstance(entity, RandomLandmark):
                width, height = entity.size
                scale_factor = (self.width / (2 * cam_range)) * 0.9  # Match the scaling of positions
                rect_width = width * scale_factor
                rect_height = height * scale_factor
                pygame.draw.rect(
                    self.screen,
                    entity.color * 200,
                    pygame.Rect(
                        x - rect_width / 2,
                        y - rect_height / 2,
                        rect_width,
                        rect_height
                    )
                )
                pygame.draw.rect(
                    self.screen,
                    (0, 0, 0),  # Randfarbe (schwarz)
                    pygame.Rect(
                        x - rect_width / 2,
                        y - rect_height / 2,
                        rect_width,
                        rect_height
                    ), 
                    1  # Randdicke
                )
            elif isinstance(entity, Agent):
                scale_factor = (self.width / (2 * cam_range)) * 0.9
                agent_radius = entity.size * scale_factor 
                
                pygame.draw.circle(
                    self.screen, entity.color * 200, (x, y), agent_radius
                )
                pygame.draw.circle(
                    self.screen, (0, 0, 0), (x, y), agent_radius, 1
                )
                # Draw start position marker (small filled circle)
                if hasattr(entity, "start_pos"):
                    start_x = (entity.start_pos[0] / cam_range) * self.width // 2 * 0.9 + self.width // 2
                    start_y = (-entity.start_pos[1] / cam_range) * self.height // 2 * 0.9 + self.height // 2

                    pygame.draw.circle(
                        self.screen,
                        entity.color * 255,  
                        (int(start_x), int(start_y)),
                        int(agent_radius * 0.5)  
                    )
                
                # draw agent trajectory
                if hasattr(entity, "trajectory") and len(entity.trajectory) > 1:
                    scale = (self.width / (2 * cam_range)) * 0.9
                    screen_points = []
                    for pos in entity.trajectory:
                        tx = (pos[0] / cam_range) * self.width // 2 * 0.9 + self.width // 2
                        ty = (-pos[1] / cam_range) * self.height // 2 * 0.9 + self.height // 2  # flip y
                        screen_points.append((int(tx), int(ty)))

                    pygame.draw.lines(self.screen, entity.color * 255, False, screen_points, 2)
                # draw goal point
                goal_x, goal_y = entity.goal_point
                goal_y *= -1  # Flipping the y-axis

                goal_x = (goal_x / cam_range) * self.width // 2 * 0.9
                goal_y = (goal_y / cam_range) * self.height // 2 * 0.9
                goal_x += self.width // 2
                goal_y += self.height // 2
                pygame.draw.circle(
                    self.screen, entity.color * 200, (goal_x, goal_y), agent_radius, 6
                )
                
                cell_size = 0.05 * scale_factor
                # Draw center 3x3 grid
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        cell_x = x + dx * cell_size
                        cell_y = y + dy * cell_size
                        pygame.draw.rect(
                            self.screen,
                            (100, 100, 255, 100),
                            pygame.Rect(
                                cell_x - cell_size / 2,
                                cell_y - cell_size / 2,
                                cell_size,
                                cell_size
                            ),
                            1
                        )

                # Draw 3x4 regions: top, bottom, left, right
                # Top region (3 wide, 4 high)
                for dx in range(-1, 2):
                    for dy in range(1, 5):
                        cx = x + dx * cell_size
                        cy = y - (1 + dy) * cell_size
                        pygame.draw.rect(
                            self.screen,
                            (100, 100, 255, 100),
                            pygame.Rect(
                                cx - cell_size / 2,
                                cy - cell_size / 2,
                                cell_size,
                                cell_size
                            ),
                            1
                        )

                # Bottom region
                for dx in range(-1, 2):
                    for dy in range(1, 5):
                        cx = x + dx * cell_size
                        cy = y + (1 + dy) * cell_size
                        pygame.draw.rect(
                            self.screen,
                            (100, 100, 255, 100),
                            pygame.Rect(
                                cx - cell_size / 2,
                                cy - cell_size / 2,
                                cell_size,
                                cell_size
                            ),
                            1
                        )

                # Left region (4 wide, 3 high)
                for dx in range(1, 5):
                    for dy in range(-1, 2):
                        cx = x - (1 + dx) * cell_size
                        cy = y + dy * cell_size
                        pygame.draw.rect(
                            self.screen,
                            (100, 100, 255, 100),
                            pygame.Rect(
                                cx - cell_size / 2,
                                cy - cell_size / 2,
                                cell_size,
                                cell_size
                            ),
                            1
                        )

                # Right region
                for dx in range(1, 5):
                    for dy in range(-1, 2):
                        cx = x + (1 + dx) * cell_size
                        cy = y + dy * cell_size
                        pygame.draw.rect(
                            self.screen,
                            (100, 100, 255, 100),
                            pygame.Rect(
                                cx - cell_size / 2,
                                cy - cell_size / 2,
                                cell_size,
                                cell_size
                            ),
                            1
                        )
            
            assert (
                0 < x < self.width and 0 < y < self.height
            ), f"Coordinates {(x, y)} are out of bounds."
    
                
    def is_obstacle(self, x, y, q_learning = False, override_epsilon = None):
        """
        Return True if (x, y) lies inside an obstacle region.

        Semantics

        Planning / state encoding (``q_learning=True``)
            Rectangular and random rectangular landmarks are treated as obstacles
            using the planning margin (``epsilon_planning``). The outer boundary
            is not checked here because A* paths do not intersect walls.

        Runtime (``q_learning=False``)
            Only rectangular landmarks are treated as obstacles using the runtime
            margin (``epsilon_runtime``). Out-of-bounds is handled by
            :meth:`Scenario.is_out_of_bounds` and not here.

        The active epsilon can be overridden via ``override_epsilon``.

        Parameters
        ----------
        x : float
            World x coordinate.
        y : float
            World y coordinate.
        q_learning : bool, optional
            If True, use planning semantics and margin; if False, use runtime semantics and margin.
        override_epsilon : float or None, optional
            Margin that replaces the default.

        Returns
        -------
        bool
            True if the point is inside an obstacle region under the selected semantics.
        """
        
        epsilon = self.epsilon_planning if q_learning else self.epsilon_runtime
        if override_epsilon is not None:
            epsilon = override_epsilon
        # check for collision with wall
        border_limit = 0.98 
        if q_learning and (abs(x) >= border_limit or abs(y) >= border_limit):
            return True
        
        for landmark in self.world.landmarks:
            if isinstance(landmark, (RectLandmark, RandomLandmark) if q_learning else RectLandmark):
                x_min = landmark.state.p_pos[0] - landmark.size[0] / 2 - epsilon
                x_max = landmark.state.p_pos[0] + landmark.size[0] / 2 + epsilon
                y_min = landmark.state.p_pos[1] - landmark.size[1] / 2 - epsilon
                y_max = landmark.state.p_pos[1] + landmark.size[1] / 2 + epsilon
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    return True
        return False
    
    def discrete(self, agent_pos_x, agent_pos_y, goal_pos_x, goal_pos_y):
        """Discretize free space into a 4‑connected grid and find nearest nodes.

        Builds a uniform grid, removes cells inside obstacles, connects
        4‑neighborhood, and returns the closest start and goal nodes.

        Args:
            agent_pos_x (float): Agent x.
            agent_pos_y (float): Agent y.
            goal_pos_x (float): Goal x.
            goal_pos_y (float): Goal y.

        Returns:
            tuple[Node | None, Node | None]: Start and end nodes on the free grid.
        """
        cell_size = 0.05
        epsilon = 5e-2
        num_steps_x = int((1.95 // cell_size) + 1)
        num_steps_y = int((1.95 // cell_size) + 1)
        
        grid = []
        # 2D node list for adding children nodes
        node_grid = [[None for _ in range(num_steps_y)]for _ in range(num_steps_x)]
        start_node = None
        end_node = None
        min_start_dist = float('inf')
        min_end_dist = float('inf')
        
        # create nodes
        for x in range(num_steps_x):
            x_coordinate = -0.95 + x * cell_size
            for y in range(num_steps_y):
                y_coordinate = -0.95 + y * cell_size
                if not self.is_obstacle(x_coordinate, y_coordinate):
                    new_node = Node(x_coordinate, y_coordinate, 1)
                    node_grid[x][y] = new_node
                    grid.append(new_node)
                    
                    dist_start = math.hypot(x_coordinate - agent_pos_x, y_coordinate - agent_pos_y)
                    dist_end = math.hypot(x_coordinate - goal_pos_x, y_coordinate - goal_pos_y)

                    if dist_start < min_start_dist:
                        min_start_dist = dist_start
                        start_node = new_node

                    if dist_end < min_end_dist:
                        min_end_dist = dist_end
                        end_node = new_node
            
        # connect nodes
        for x in range(num_steps_x):
            for y in range(num_steps_y):
                node = node_grid[x][y]
                if node is not None:
                    # right
                    if x+1 < num_steps_x and node_grid[x + 1][y]:
                        node.add_child(node_grid[x+1][y])
                    # up
                    if y + 1 < num_steps_y and node_grid[x][y + 1]:
                        node.add_child(node_grid[x][y+1])
                    # left
                    if x - 1 >= 0 and node_grid[x - 1][y]:
                        node.add_child(node_grid[x-1][y])
                    # down
                    if y - 1 >= 0 and node_grid[x][y - 1]:
                        node.add_child(node_grid[x][y-1])

        if end_node is None:
                ("Endknoten konnte nicht gefunden werden")
        
        if start_node is None:
                ("Startknoten konnte nicht gefunden werden")
                            
        return start_node, end_node
    

    def heuristic(self, node, end_node):
        """Return heuristic for A*.

        Currently returns zero to obtain Dijkstra behavior.

        Args:
            node (Node): Current node.
            end_node (Node): Goal node.

        Returns:
            float: Heuristic estimate of remaining cost.
        """
        # euclidean distance
        #return ((node.x - end_node.x) ** 2 + (node.y - end_node.y) ** 2) ** 0.5
        return 0


    def A_star(self, start_node, end_node):
        
        """Compute a shortest path on the discrete grid.

        Uses best‑first search with total cost g + h (A*) and reconstructs the path by
        following parent pointers.

        Args:
            start_node (Node): Start node.
            end_node (Node): Goal node.

        Returns:
            list[tuple[float, float]] | None: Path as a list of (x, y) positions
                in world coordinates. None if no path exists.
        """
        Open = [start_node]
        Closed = []
        
        start_node.cost_to_come = 0
        start_node.cost_to_go = self.heuristic(start_node, end_node)
        start_node.total_cost = start_node.cost_to_come + start_node.cost_to_go
        
        while Open:
            # best first search method with total cost
            current_node = min(Open, key=lambda node: node.total_cost)
            Open.remove(current_node)
            Closed.append(current_node)
            
            # check if end_node has been reached
            if (current_node.x, current_node.y) == (end_node.x, end_node.y):
                path = []
                while current_node is not None:
                    path.append((current_node.x, current_node.y))
                    current_node = current_node.parent
                return path[::-1]
            
            # check child nodes
            for child in current_node.children:
                if child in Closed:
                    continue

                # calculate new approx. cost
                approx_cost_to_come = current_node.cost_to_come + child.cost

                if child not in Open:
                    Open.append(child)
                elif approx_cost_to_come >= child.cost_to_come:
                    continue  

                child.parent = current_node
                child.cost_to_come = approx_cost_to_come
                child.cost_to_go = self.heuristic(child, end_node)
                child.total_cost = child.cost_to_come + child.cost_to_go
                
            if not Open:
                print("ERROR: No path found")


    def is_dyn_obstacle(self, x, y, current_agent):
        """Return True if a point is inside any other agent's disk.

        Args:
            x (float): World x coordinate.
            y (float): World y coordinate.
            current_agent (Agent): Agent to exclude from the check.

        Returns:
            bool: True if the point intersects another agent.
        """
        for agent in self.world.agents:
            # agent should not detect himself as an obstacle
            if agent == current_agent:
                continue
            else:
                euclidian_dis = math.sqrt((x - agent.state.p_pos[0]) ** 2 + (y - agent.state.p_pos[1]) ** 2)
                if euclidian_dis <= agent.size:
                    return True
        return False
    
    
    def classify_region(self, grid_region, direction):
        """Classify a 3x4 region into free, passable, or impassable.

        Free if the number of obstacle cells is below a threshold.
        Passable if at least one row or column in the movement direction is fully free.
        Otherwise impassable.

        Args:
            grid_region (list[list[int]]): 3x4 binary subgrid.
            direction (str): One of "top", "bottom", "left", "right".

        Returns:
            int: 0 for free, 1 for passable, 2 for impassable.

        Raises:
            ValueError: If direction is not recognized.
        """
        flat = [cell for row in grid_region for cell in row]
        obstacle_count = sum(flat)

        if obstacle_count < 4:
            return 0  # free

        if direction in ("top", "bottom"):
            # Check if there is **any passable column** (i.e., not all 1s)
            cols = list(zip(*grid_region))  # transpose
            if any(any(cell == 0 for cell in col) for col in cols):
                return 1  # passable
            else:
                return 2  # impassable

        elif direction in ("left", "right"):
            # Check if there is **any passable row** (i.e., not all 1s)
            if any(any(cell == 0 for cell in row) for row in grid_region):
                return 1  # passable
            else:
                return 2  # impassable

        else:
            raise ValueError(f"Unknown direction: {direction}")
    
    
    def q_learning_state_space(self, agent, a_star_new):
        """Encode the local observation for tabular Q‑learning.

        Builds a binary occupancy grid around the agent, embeds the A* direction
        into the center cell, keeps the center 3×3 at full resolution, and aggregates
        four 3×4 directional regions into categorical labels.

        Args:
            agent (Agent): Agent to encode.
            a_star_new (list[tuple[float, float]]): Remaining A* path.

        Returns:
            tuple[int, ...]: Compact state tuple length 13.
        """
        size = 11
        agent_grid = 5
        cell_size = 0.05
        agent_pos_x = agent.state.p_pos[0]
        agent_pos_y = agent.state.p_pos[1]
        start_pos_x = agent_pos_x - agent_grid*cell_size
        start_pos_y = agent_pos_y - agent_grid*cell_size

        # raw grid with 0s and 1s
        raw_grid = [[0 for _ in range(size)] for _ in range(size)]
        for r in range(size):
            for c in range(size):
                x = start_pos_x + c * cell_size
                y = start_pos_y + r * cell_size
                if(r, c) == (agent_grid, agent_grid):
                    a_star_action = self.a_star_direction(agent, a_star_new)
                    raw_grid[r][c] = a_star_action
                    agent.a_star_action = a_star_action
                elif self.is_obstacle(x, y, q_learning=True, override_epsilon=0.015) or self.is_dyn_obstacle(x, y, agent):
                    raw_grid[r][c] = 1

        state = []
        
        # --- Center 3x3 block: keep full resolution ---
        for r in range(3, 6):
            for c in range(3, 6):
                state.append(raw_grid[r][c])
        
        # --- Surrounding 3x4 blocks: aggregate to 1 if >= 2 obstacles ---
        # Define regions:  top, left, right, bottom
        surrounding_blocks = {
            'top':    (6, 3, 9, 7),
            'left':   (3, 0, 6, 4),
            'right':  (3, 5, 6, 9),
            'bottom': (0, 3, 3, 7),
        }


        for direction in ['top', 'left', 'right', 'bottom']:
            r_start, c_start, r_end, c_end = surrounding_blocks[direction]
            block = [raw_grid[r][c_start:c_end] for r in range(r_start, r_end)]
            label = self.classify_region(block, direction)
            state.append(label)
        
        return tuple(state)
    
    def a_star_direction(self, agent, a_star_new):
        """Return the dominant discrete direction toward the next target.

        Chooses either the goal or the nearest remaining A* waypoint and returns
        as a discrete action code.

        Args:
            agent (Agent): Agent instance.
            a_star_new (list[tuple[float, float]]): Remaining A* waypoints.

        Returns:
            int: Discrete action code in {0 up, 1 down, 2 left, 3 right}.
        """

        agent_pos = np.array(agent.state.p_pos)
        
        if not a_star_new:
            target = np.array(agent.goal_point)
        else:
            target = min(a_star_new, key=lambda point: np.linalg.norm(np.array(point) - agent_pos))
        
        dx = target[0] - agent_pos[0]
        dy = target[1] - agent_pos[1]

        # Prioritize dominant direction
        if abs(dx) > abs(dy):
            return 3 if dx > 0 else 2
        else:
            return 0 if dy > 0 else 1


    def last(
        
        self, observe: bool = True
    ) -> tuple[ObsType | None, float, bool, bool, dict[str, Any]]:
        """Return PettingZoo‑style observation tuple for the current agent.

        Also updates the agent's Q‑state before returning.

        Args:
            observe (bool): If True, compute the observation.

        Returns:
            tuple: (observation, reward, terminated, truncated, info, q_state)
        """
        agent = self.agent_selection # current agent that is being stepped
        assert agent is not None
        observation = self.observe(agent) if observe else None
        agent_object = self.world.agents[self._index_map[agent]]
        
        agent_object.q_state = self.q_learning_state_space(agent_object, agent_object.a_star_new)
        
        return (
            observation,
            agent_object.reward,
            self.terminations[agent],
            self.truncations[agent],
            self.infos[agent],
            agent_object.q_state
        )


    def reset(self, seed=None, options=None):
        """Reset the environment and initialize per‑agent A* paths and states.

        Randomizes start and goal positions subject to obstacle and separation constraints.

        Args:
            seed (int | None): seed.
            options (dict | None): Unused, kept for API compatibility.

        Returns:
            tuple[dict[str, tuple[int, ...]], dict[str, np.ndarray]]:
                Mapping from agent name to Q‑state, and mapping to raw observation.
        """
        if seed is not None:
            self._seed(seed=seed)
        self.scenario.reset_world(self.world, self.np_random)
        
        all_poses = [entity.state.p_pos for entity in self.world.entities]
        self.fixed_cam_range = np.max(np.abs(np.array(all_poses)))

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self.agent_selection = self._agent_selector.reset()
        self.steps = 0

        self.current_actions = [None] * self.num_agents
        
        agent_states ={}
        agent_observations ={}
        
        for agent in self.agents:
            assert agent is not None
            observation = self.observe(agent)
            start_node, end_node = self.discrete(observation[0], observation[1], observation[2], observation[3])
            agent_object = self.world.agents[self._index_map[agent]]
            a_star_path = self.A_star(start_node, end_node)
            if a_star_path is None:
                print("Es konnte kein Pfad gefunden werden")
            agent_object.a_star_new = a_star_path[2:] # remove first two elements
            agent_object.a_star_old = []
            agent_object.q_state = self.q_learning_state_space(agent_object, agent_object.a_star_new)
            agent_object.reward = 0.0
            agent_object.movable = True
            agent_states[agent] = agent_object.q_state
            agent_observations[agent] = observation
            agent_object.controller_x = PID(setpoint=0)
            agent_object.controller_y = PID(setpoint=0)
            agent_object.terminated = False
            agent_object.a_star_action = None
            agent_object.start_pos = np.array(agent_object.state.p_pos)
            agent_object.trajectory = [tuple(agent_object.state.p_pos)] 
            
        return agent_states, agent_observations
    
        
    def _skip_dead_agent(self):
        agent = self.agent_selection
        agent_obj = next((a for a in self.world.agents if a.name == agent), None)
        agent_obj.movable = False
        
        agent_list = list(self.agents)
        if all(self.terminations[agent] or self.truncations[agent] for agent in agent_list):
            return 
        
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()
        
        if next_idx == 0:
            self._execute_world_step()
            self.steps += 1
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.truncations[a] = True
        
        
        
        
    def step(self, action):
        """Advance the environment by one agent action.

        Buffers the action, advances the world when the last agent has acted,
        updates rewards and terminations, and renders if enabled.

        Args:
            action (np.ndarray | int | None): Action for the selected agent. None if terminated.
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._skip_dead_agent()
            return
        cur_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()

        self.current_actions[current_idx] = action

        if next_idx == 0:
            self._execute_world_step()
            self.steps += 1
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.truncations[a] = True
        else:
            self._clear_rewards()

        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def next_agent(self):
        """Advance the internal agent selector to the next agent."""
        self.agent_selection = self._agent_selector.next()
        
    
    def update_a_star_paths(self, agent):
        """Advance and clean the remaining A* path for an agent.

        Removes already reached waypoints based on proximity. Clears the remaining path
        upon entering a goal neighborhood.

        Args:
            agent (Agent): Agent instance.

        Returns:
            bool: True if the path list or status changed.
        """
        epsilon = 25e-3
        goal_epsilon = 0.09
        agent_pos = np.array(agent.state.p_pos)
        goal_pos = np.array(agent.goal_point)

        if np.linalg.norm(agent_pos - goal_pos) < goal_epsilon:
            if agent.a_star_new:
                agent.a_star_old.extend(agent.a_star_new)
                agent.a_star_new.clear()
                print(agent.name + " reached goal proximity — cleared remaining path.")
            return True

        # Sort path points by distance (ascending) but prefer later ones in tie
        sorted_path = sorted(
            enumerate(agent.a_star_new),
            key=lambda idx_point: (np.linalg.norm(agent_pos - np.array(idx_point[1])), -idx_point[0])
        )
        
        for idx, (path_x, path_y) in sorted_path:
            if math.isclose(agent_pos[0], path_x, abs_tol=epsilon) and math.isclose(agent_pos[1], path_y, abs_tol=epsilon):
                # Reached this A* point: remove all earlier ones too
                reached_index = idx
                reached_points = agent.a_star_new[:reached_index + 1]
                agent.a_star_old.extend(reached_points)
                agent.a_star_new = agent.a_star_new[reached_index + 1:]
                print(agent.name + f" skipped to A* path point: {path_x, path_y} — cleaned up {len(reached_points)} points")
            return True

        return False
    

    def _execute_world_step(self):
        # update q-state and a* direction before stepping the agents
        for agent in self.world.agents:
            agent.q_state = self.q_learning_state_space(agent, agent.a_star_new)
        # set action for each agent
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = []
            if agent.movable:
                mdim = self.world.dim_p * 2 + 1
                if self.continuous_actions:
                    scenario_action.append(action[0:mdim])
                    action = action[mdim:]
                else:
                    scenario_action.append(action % mdim)
                    action //= mdim
            if not agent.silent:
                scenario_action.append(action)
            self._set_action(scenario_action, agent, self.action_spaces[agent.name])

        self.world.step()

        for agent in self.world.agents:
            agent.trajectory.append(tuple(agent.state.p_pos))
        global_reward = 0.0
        if self.local_ratio is not None:
            global_reward = float(self.scenario.global_reward(self.world))
        

        for agent in self.world.agents:
            agent_reward = float(self.scenario.reward(agent, self.world))
            self.update_a_star_paths(agent)
            if self.local_ratio is not None:
                reward = (
                    global_reward * (1 - self.local_ratio)
                    + agent_reward * self.local_ratio
                )
            else:
                reward = agent_reward

            self.rewards[agent.name] = reward
            self.terminations[agent.name] = self.scenario.is_termination(agent, self.world)
            
    def get_next_point(self, agent_x, agent_y, action, agent):
        """Compute the next 1‑step grid target for a discrete action.

        Args:
            agent_x (float): Current x.
            agent_y (float): Current y.
            action (int): Discrete action in {0 up, 1 down, 2 left, 3 right}.
            agent (str): Agent name, unused but kept for symmetry.

        Returns:
            list[float, float]: Target point in world coordinates.
        """
        gridsize = 0.05
        goal_point = [0.0, 0.0]
        if action == 0: #up
            goal_point[0]=agent_x
            goal_point[1]=agent_y+gridsize
        if action == 1: #down
            goal_point[0]=agent_x
            goal_point[1]=agent_y-gridsize
        if action == 2: #left
            goal_point[0]=agent_x-gridsize
            goal_point[1]=agent_y
        if action == 3: #right
            goal_point[0]=agent_x+gridsize
            goal_point[1]=agent_y   
        
        return goal_point

    def get_cont_action(self, observation, dimension, discrete_action, agent):
        """Convert a discrete move into a continuous control vector.

        Sets PID setpoints one grid cell away, computes planar velocity commands,
        and maps them into the MPE action vector layout.

        Args:
            observation (np.ndarray): Current observation [x, y, goal_x, goal_y].
            dimension (int): World dimension, expected 2.
            discrete_action (int): Discrete action in {0,1,2,3}.
            agent (str): Agent name.

        Returns:
            np.ndarray: Continuous action vector of length 2*dimension + 1.
        """
        agent_object = self.world.agents[self._index_map[agent]]
        agent_object.action_history.append(discrete_action)
        if len(agent_object.action_history) > 2:
            agent_object.action_history.pop(0)
        next_point = self.get_next_point(observation[0], observation[1], discrete_action, agent)
        
        agent_object.controller_x.setpoint = next_point[0]
        agent_object.controller_y.setpoint = next_point[1]
        
        agent_object.state.p_vel = np.zeros_like(agent_object.state.p_vel)
        
        v_x = agent_object.controller_x(observation[0])
        v_y = agent_object.controller_y(observation[1])
        
        v_x = v_x * 1.25
        v_y = v_y * 1.25
        
        action = np.zeros(dimension * 2 + 1)
        action[1] = -v_x
        action[3] = -v_y
        return action


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    """Scenario with static shelf-like landmarks, random starts and goals, and rewards."""
    
    def __init__(self):
        """Initialize default collision margins for runtime and planning."""
        super().__init__()
        self.epsilon_runtime = 5e-3
        self.epsilon_planning = 9e-2
    
    def make_world(self, num_good=2, num_obstacles=4):
        """Create world, agents, and rectangular landmarks.

        Args:
            num_good (int): Number of agents.
            num_obstacles (int): Number of rectangular obstacles.

        Returns:
            World: Configured world instance.
        """
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = num_good
        num_landmarks = num_obstacles
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            base_name =  "agent"
            base_index = i
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.027
            agent.accel = 4.0
            agent.max_speed = 1.3
            agent.damping = 0.5
        # add landmarks
        world.landmarks = [RectLandmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = np.array([0.2, 0.4])
            landmark.boundary = False
        world.state = 0
        return world


    def reset_world(self, world, np_random):
        """Randomize agent states and goal points.

        Ensures no overlap with obstacles, enforces inter‑agent spacing, and avoids
        goal conflicts.

        Args:
            world (World): World to modify.
            np_random (np.random.Generator): Random generator from PettingZoo.
        """
        colors = [
        [0.0, 1.0, 0.0],   # green
        [0.0, 0.0, 1.0],   # blue
        [1.0, 1.0, 0.0],   # yellow
        [1.0, 0.5, 0.0],   # orange
        [0.5, 0.0, 0.5],   # purple
        [0.0, 1.0, 1.0],   # cyan
        [1.0, 0.0, 1.0],   # magenta
        [0.5, 0.5, 0.0],   # olive
        [0.0, 0.5, 0.5],   # teal
        [0.5, 0.5, 1.0],   # light blue
        [1.0, 0.8, 0.8],   # pinkish
        [0.7, 0.7, 0.7],   # light gray
        [0.3, 0.3, 0.3],   # dark gray
        [1.0, 0.0, 0.0],   # red
        [0.8, 0.6, 0.7],   # mauve
        [0.6, 0.4, 0.2],   # brown
        [0.2, 0.6, 0.2],   # forest green
        [0.2, 0.2, 0.6],   # deep blue
        [0.9, 0.9, 0.1],   # lemon
        [0.6, 0.2, 0.8]   # violet
        ]
        
        # set states for landmarks
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_vel = np.zeros(world.dim_p)
                landmark.color = np.array([0.25, 0.25, 0.25])
        
        # set initial states for rectangle landmarks
        world.landmarks[0].state.p_pos= np.array([-0.25, -0.3])
        world.landmarks[1].state.p_pos= np.array([-0.25, 0.3])
        world.landmarks[2].state.p_pos= np.array([0.25, -0.3])
        world.landmarks[3].state.p_pos= np.array([0.25, 0.3])
        
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = (
                np.array(colors[i % len(colors)])
            )
        
        for agent in world.agents:
            agent.state.p_vel = np.zeros(world.dim_p)
            
            while True:
                pos = np_random.uniform(-0.8, +0.8, world.dim_p)
                if self.is_in_landmark(world, pos[0], pos[1]):
                    continue
                
                agent.state.p_pos = pos
                
                collision = any(
                    np.linalg.norm(agent.state.p_pos - other.state.p_pos) < (agent.size + other.size)
                    for other in world.agents 
                    if other is not agent and other.state.p_pos is not None
                )
                
                if not collision:
                    break
                
                agent.state.p_pos = None
                
            agent.state.c = np.zeros(world.dim_c)
            while True:
                pos = np_random.uniform(-0.6, +0.6, world.dim_p)
                if self.is_in_landmark(world, pos[0], pos[1]):
                    continue
                
                # check minimum distance
                start_pos = agent.state.p_pos
                distance = np.linalg.norm(pos - start_pos)
                if distance < 0.25:
                    continue
                
                goal_collision = any(
                    hasattr(other, 'goal_point') and isinstance(other.goal_point, list) and len(other.goal_point) == 2 and
                    np.linalg.norm(np.array(pos) - np.array(other.goal_point)) < 0.15
                    for other in world.agents 
                    if other is not agent
                )
            
                if not goal_collision:
                    agent.goal_point = pos
                    break
            

        
    def is_in_landmark(self, world, pos_x, pos_y):
        """Return True if a point is inside any rectangle with planning margin.

        Args:
            world (World): World instance.
            pos_x (float): x coordinate.
            pos_y (float): y coordinate.

        Returns:
            bool: True if inside any expanded rectangle.
        """
        for landmark in world.landmarks:
            if isinstance(landmark, (RectLandmark, RandomLandmark)):
                if landmark.state.p_pos is None:
                    continue
                x_min = landmark.state.p_pos[0] - landmark.size[0] / 2 - self.epsilon_planning
                x_max = landmark.state.p_pos[0] + landmark.size[0] / 2 + self.epsilon_planning
                y_min = landmark.state.p_pos[1] - landmark.size[1] / 2 - self.epsilon_planning
                y_max = landmark.state.p_pos[1] + landmark.size[1] / 2 + self.epsilon_planning
    
                if x_min <= pos_x <= x_max and y_min <= pos_y <= y_max:
                        return True
        return False
        
        
    def is_collision(self, agent1, agent2):
        """Return True if two agents overlap with runtime margin.

        Args:
            agent1 (Agent): First agent.
            agent2 (Agent): Second agent.

        Returns:
            bool: True if center distance is below sum of radii plus margin.
        """
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size + self.epsilon_runtime
        return True if dist < dist_min else False
    
    def is_out_of_bounds(self, agent, margin = 0.05):
        """Return True if the agent is outside the square workspace.

        Args:
            agent (Agent): Agent instance.
            margin (float): Allowed slack beyond unit box.

        Returns:
            bool: True if any coordinate exceeds the bound.
        """
        bound = 1.0 + margin
        return np.any(np.abs(agent.state.p_pos) > bound)
    
    def is_goal(self, agent):
        """Return True if the agent is within goal proximity.

        Args:
            agent (Agent): Agent instance.

        Returns:
            bool: True if distance to goal is below size plus epsilon.
        """
        epsilon_goal = 6e-3
        delta_pos = agent.state.p_pos - agent.goal_point
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent.size + epsilon_goal
        return True if dist < dist_min else False  
            
    
    def reward(self, agent, world):
        """Compute shaped reward for a single agent.

        Penalizes collisions and out-of-bounds, adds small time penalty,
        and rewards A*-aligned actions.

        Args:
            agent (Agent): Agent instance.
            world (World): World instance.

        Returns:
            float: Scalar reward for this agent and step.
        """
        reward = 0
        
        for landmark in world.landmarks:
            if landmark.is_collision(agent):
                reward -= 24    # collision
                    
        for other_agent in world.agents:
            if self.is_collision(agent, other_agent):
                if other_agent == agent:
                    continue
                else:
                    reward -= 24    # collision
        
        if self.is_out_of_bounds(agent):
            reward -= 24            # collision
            
        if agent.a_star_action is not None and agent.action_history[-1] == agent.a_star_action:
                reward += 1         # choose a-star      
    
        reward -= 1.2          # time penalty

        agent.reward = reward
        return reward

    def agents(self, world):
        """Return list of agents.

        Args:
            world (World): World instance.

        Returns:
            list[Agent]: Agents in the world.
        """
        return world.agents


    def observation(self, agent, world):
        """Return low‑level observation [x, y, goal_x, goal_y].

        Args:
            agent (Agent): Agent instance.
            world (World): World instance.

        Returns:
            np.ndarray: Concatenated agent and goal positions.
        """
        # world argument is kept because methods in the simple_env.py have it based on the original version of the observation method
        agent_pos = np.array(agent.state.p_pos)
        goal_pos = np.array(agent.goal_point)
        return np.concatenate((agent_pos, goal_pos))

    
    def is_termination(self, agent, world):
        """Return True if the agent reached a terminal condition.

        Terminal events include collision, out‑of‑bounds, or goal reached.

        Args:
            agent (Agent): Agent instance.
            world (World): World instance.

        Returns:
            bool: Termination flag.
        """
        if agent.terminated:
            return True
        
        termination = False
        for landmark in world.landmarks:
            if landmark.is_collision(agent):
                termination = True
                    
        for other_agent in world.agents:
            if self.is_collision(agent, other_agent):
                if other_agent == agent:
                    continue
                else:
                    termination = True
        
        if self.is_out_of_bounds(agent):
            termination = True
            
        if self.is_goal(agent):
            termination = True
        
        return termination
