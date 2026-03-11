import math
import random
from collections import deque
from typing import Dict, List, Optional, Tuple
import pygame

from config.settings import *
from core.models import Robot, Target, CommNode, PropagatedPacket, RobotObservation
from utils.math_utils import clamp, dist, angle_wrap, segment_intersects_circle
from utils.generators import generate_obstacles, generate_nodes, generate_target

class ForestSearchEnv:
    def __init__(self, num_nodes: int = DEFAULT_NUM_NODES, flying_robot: bool = True):
        self.num_nodes = num_nodes
        self.flying_robot = flying_robot

        self.obstacles = generate_obstacles(NUM_TREES)
        self.nodes = generate_nodes(self.obstacles, self.num_nodes)
        self.node_neighbors = self.build_node_graph()
        self.target = generate_target(self.obstacles)
        self.robot = Robot(120, 120)
        self.random_bot = Robot(180, 120)

        self.font = None
        self.big_font = None
        self.small_font = None
        self.camera = [0.0, 0.0]
        self.debug_draw_signal = False
        self.show_target_always = True
        self.target_moving_enabled = False
        self.target.moving = self.target_moving_enabled
        self.manual_camera = False
        self.autopilot_enabled = True
        self.random_baseline_enabled = True
        self.show_packets = True
        self.panel_scroll = 0
        self.panel_scroll_x = 0

        self.current_waypoint: Optional[Tuple[float, float]] = None
        self.search_mode = "sweep"
        self.sweep_points = self.build_sweep_points()
        self.sweep_index = 0

        self.best_estimate: Optional[Tuple[float, float]] = None
        self.estimated_strength: Optional[float] = None
        self.last_good_estimate: Optional[Tuple[float, float]] = None
        self.last_estimate_time = -999.0
        self.home_mode_started = 0.0
        self.last_home_target: Optional[Tuple[float, float]] = None
        self.local_probe_center: Optional[Tuple[float, float]] = None
        self.local_probe_until = 0.0
        self.local_probe_points_visited = 0

        self.stall_anchor = (self.robot.x, self.robot.y)
        self.stall_start_time = 0.0
        self.random_explore_until = 0.0
        self.recent_positions: List[Tuple[float, float, float]] = []

        self.active_packets: Dict[int, PropagatedPacket] = {}
        self.visible_packet_edges: List[Tuple[int, int, float]] = []
        self.last_robot_decodes: List[RobotObservation] = []
        self.packet_counter = 0
        self.last_emit_time = -999.0
        self.reset_stats()

    def reset_stats(self):
        self.steps = 0
        self.time_elapsed = 0.0
        self.best_estimate = None
        self.estimated_strength = None
        self.last_good_estimate = None
        self.last_estimate_time = -999.0
        self.last_distress_ping = None
        self.home_mode_started = 0.0
        self.last_home_target = None
        self.local_probe_center = None
        self.local_probe_until = 0.0
        self.local_probe_points_visited = 0
        self.stall_anchor = (self.robot.x, self.robot.y)
        self.stall_start_time = 0.0
        self.random_explore_until = 0.0
        self.recent_positions = []
        self.active_packets = {}
        self.visible_packet_edges = []
        self.last_robot_decodes = []
        self.packet_counter = 0
        self.last_emit_time = -999.0
        self.panel_scroll = 0
        self.panel_scroll_x = 0
        self.robot.found_time = None
        self.robot.found_target = False
        self.random_bot.found_time = None
        self.random_bot.found_target = False
        self.random_bot.waypoint = None
        self.random_bot.repath_at = 0.0

    def reset(self):
        moving_enabled = self.target_moving_enabled
        baseline_enabled = self.random_baseline_enabled
        self.__init__(num_nodes=self.num_nodes, flying_robot=self.flying_robot)
        self.target_moving_enabled = moving_enabled
        self.random_baseline_enabled = baseline_enabled
        self.target.moving = moving_enabled

    def random_free_point(self, margin: float = 40.0) -> Tuple[float, float]:
        for _ in range(220):
            x = random.uniform(margin, WORLD_WIDTH - margin)
            y = random.uniform(margin, WORLD_HEIGHT - margin)
            if self.flying_robot:
                return (x, y)
            if not self.collides(x, y, ROBOT_RADIUS + 2):
                return (x, y)
        return (WORLD_WIDTH / 2, WORLD_HEIGHT / 2)

    def update_target_motion(self, dt: float):
        if not self.target.moving or self.target.found:
            return
        if self.target.waypoint is None or self.time_elapsed >= self.target.repath_at:
            self.target.waypoint = self.random_free_point(margin=80.0)
            self.target.repath_at = self.time_elapsed + TARGET_REPATH_TIME
        tx, ty = self.target.x, self.target.y
        wx, wy = self.target.waypoint
        dx, dy = wx - tx, wy - ty
        d = math.hypot(dx, dy)
        if d <= TARGET_WAYPOINT_REACHED:
            self.target.waypoint = self.random_free_point(margin=80.0)
            self.target.repath_at = self.time_elapsed + TARGET_REPATH_TIME
            return
        step = min(TARGET_MOVE_SPEED * dt, d)
        nx = tx + (dx / d) * step
        ny = ty + (dy / d) * step
        if self.flying_robot or not self.collides(nx, ny, TARGET_RADIUS):
            self.target.x, self.target.y = nx, ny
        else:
            self.target.waypoint = self.random_free_point(margin=80.0)
            self.target.repath_at = self.time_elapsed + 2.0

    def build_sweep_points(self) -> List[Tuple[float, float]]:
        points = []
        margin = 150
        lane_gap = 155
        go_right = True
        y = margin
        while y <= WORLD_HEIGHT - margin:
            if go_right:
                points.append((margin, y))
                points.append((WORLD_WIDTH - margin, y))
            else:
                points.append((WORLD_WIDTH - margin, y))
                points.append((margin, y))
            y += lane_gap
            go_right = not go_right
        return points

    def build_node_graph(self) -> Dict[int, List[int]]:
        neighbors: Dict[int, List[int]] = {i: [] for i in range(len(self.nodes))}
        for i, a in enumerate(self.nodes):
            if a.failed:
                continue
            ranked = []
            for j, b in enumerate(self.nodes):
                if i == j or b.failed:
                    continue
                d = dist((a.x, a.y), (b.x, b.y))
                if d <= MAX_EDGE_DIST:
                    ranked.append((d, j))
            ranked.sort(key=lambda t: t[0])
            neighbors[i] = [j for _, j in ranked[:5]]
        for i, nbrs in list(neighbors.items()):
            for j in nbrs:
                if i not in neighbors[j]:
                    neighbors[j].append(i)
        for i in neighbors:
            uniq = sorted(set(neighbors[i]), key=lambda j: dist((self.nodes[i].x, self.nodes[i].y), (self.nodes[j].x, self.nodes[j].y)))
            neighbors[i] = uniq[:6]
        return neighbors

    def collides(self, x: float, y: float, radius: float) -> bool:
        if x - radius < 0 or y - radius < 0 or x + radius > WORLD_WIDTH or y + radius > WORLD_HEIGHT:
            return True
        if self.flying_robot:
            return False
        for ob in self.obstacles:
            if dist((x, y), (ob.x, ob.y)) < radius + ob.r:
                return True
        return False

    def path_blockers(self, a: Tuple[float, float], b: Tuple[float, float]) -> int:
        return sum(1 for ob in self.obstacles if segment_intersects_circle(a, b, ob))

    def rssi_from_distance(self, d: float) -> float:
        d = max(1.0, d)
        return -35 - 20 * math.log10(d)

    def nearest_seed_nodes(self, k: int = 3) -> List[int]:
        active = [(i, dist((self.target.x, self.target.y), (n.x, n.y))) for i, n in enumerate(self.nodes) if not n.failed]
        active.sort(key=lambda t: t[1])
        return [i for i, _ in active[:k]]

    def emit_distress_packet(self):
        seed_nodes = self.nearest_seed_nodes(k=3)
        for node_id in seed_nodes:
            node = self.nodes[node_id]
            d = dist((self.target.x, self.target.y), (node.x, node.y))
            blockers = self.path_blockers((self.target.x, self.target.y), (node.x, node.y))
            source_rssi = self.rssi_from_distance(d) - blockers * random.uniform(1.0, 2.8) + random.gauss(0, EDGE_JITTER_STD)
            pkt = PropagatedPacket(self.packet_counter, node_id, 1, source_rssi, 1.0, self.time_elapsed, [node_id])
            self.packet_counter += 1
            self.propagate_packet(pkt)

    def edge_success_probability(self, i: int, j: int) -> float:
        a = self.nodes[i]
        b = self.nodes[j]
        d = dist((a.x, a.y), (b.x, b.y))
        blockers = self.path_blockers((a.x, a.y), (b.x, b.y))
        distance_frac = clamp(d / MAX_EDGE_DIST, 0.0, 1.0)
        p = EDGE_BASE_SUCCESS - EDGE_DISTANCE_PENALTY * distance_frac - EDGE_BLOCKER_PENALTY * blockers
        return clamp(p, 0.08, 0.98)

    def propagate_packet(self, seed_packet: PropagatedPacket):
        q = deque([seed_packet])
        best_seen: Dict[int, int] = {seed_packet.node_id: seed_packet.hop_count}
        while q:
            pkt = q.popleft()
            cur = self.active_packets.get(pkt.node_id)
            if cur is None or pkt.hop_count < cur.hop_count or pkt.source_rssi > cur.source_rssi:
                self.active_packets[pkt.node_id] = pkt
            if pkt.hop_count >= MAX_PACKET_HOPS:
                continue
            for nbr in self.node_neighbors.get(pkt.node_id, []):
                if nbr in pkt.path or best_seen.get(nbr, 999) <= pkt.hop_count + 1 or self.nodes[nbr].failed:
                    continue
                if random.random() > self.edge_success_probability(pkt.node_id, nbr):
                    continue
                a = self.nodes[pkt.node_id]
                b = self.nodes[nbr]
                d = dist((a.x, a.y), (b.x, b.y))
                blockers = self.path_blockers((a.x, a.y), (b.x, b.y))
                forwarded_rssi = pkt.source_rssi - 0.14 * d / 10.0 - blockers * random.uniform(0.4, 1.2) + random.gauss(0, EDGE_JITTER_STD)
                new_pkt = PropagatedPacket(seed_packet.origin_id, nbr, pkt.hop_count + 1, forwarded_rssi, max(0.0, pkt.freshness - 0.12), self.time_elapsed, pkt.path + [nbr])
                q.append(new_pkt)
                best_seen[nbr] = new_pkt.hop_count
                self.visible_packet_edges.append((pkt.node_id, nbr, self.time_elapsed))

    def prune_packets(self):
        self.active_packets = {node_id: pkt for node_id, pkt in self.active_packets.items() if (self.time_elapsed - pkt.delivered_time) <= PACKET_TTL}
        self.visible_packet_edges = [e for e in self.visible_packet_edges if (self.time_elapsed - e[2]) <= 0.6]

    def generate_robot_observations(self) -> List[RobotObservation]:
        observations: List[RobotObservation] = []
        for node_id, pkt in self.active_packets.items():
            node = self.nodes[node_id]
            d_robot = dist((self.robot.x, self.robot.y), (node.x, node.y))
            if d_robot > SENSOR_RANGE:
                continue
            blockers = self.path_blockers((self.robot.x, self.robot.y), (node.x, node.y))
            p_success = clamp(0.96 - 0.24 * (d_robot / SENSOR_RANGE) - 0.08 * blockers, 0.08, 0.98)
            delivered = random.random() < p_success
            robot_link_rssi = pkt.source_rssi - 0.12 * d_robot / 10.0 - blockers * random.uniform(0.5, 1.3) + random.gauss(0, 1.1)
            observations.append(RobotObservation(node_id, pkt.hop_count, pkt.source_rssi, max(0.0, 1.0 - (self.time_elapsed - pkt.delivered_time) / PACKET_TTL), robot_link_rssi, delivered))
        return observations

    def update_estimate(self):
        delivered = [o for o in self.robot.last_observations if o.delivered]
        if not delivered:
            self.best_estimate = None
            self.estimated_strength = None
            return
        top = sorted(delivered, key=lambda o: o.robot_link_rssi + 12.0 * o.freshness - 2.2 * o.hop_count, reverse=True)[:5]
        weighted_x = weighted_y = total_w = 0.0
        for obs in top:
            node = self.nodes[obs.node_id]
            w = max(0.5, (obs.robot_link_rssi + 130.0) * 0.06 + 2.2 * obs.freshness - 0.35 * obs.hop_count)
            weighted_x += node.x * w
            weighted_y += node.y * w
            total_w += w
        if total_w > 0:
            self.best_estimate = (weighted_x / total_w, weighted_y / total_w)
            self.estimated_strength = max(o.robot_link_rssi for o in top)
            self.last_good_estimate = self.best_estimate
            self.last_estimate_time = self.time_elapsed

    def steer_bot_toward(self, bot: Robot, waypoint: Optional[Tuple[float, float]], dt: float, speed: float):
        if waypoint is None:
            bot.speed = 0.0
            return
        dx = waypoint[0] - bot.x
        dy = waypoint[1] - bot.y
        desired_heading = math.atan2(dy, dx)
        angle_error = angle_wrap(desired_heading - bot.heading)
        turn_amount = clamp(angle_error * AUTOPILOT_TURN_GAIN, -TURN_SPEED, TURN_SPEED)
        bot.heading += turn_amount * dt
        speed_factor = max(0.28, 1.0 - min(abs(angle_error), AUTOPILOT_SLOWDOWN_ANGLE) / AUTOPILOT_SLOWDOWN_ANGLE)
        bot.speed = speed * speed_factor
        old_x, old_y = bot.x, bot.y
        bot.forward(dt)
        if self.collides(bot.x, bot.y, ROBOT_RADIUS):
            bot.x, bot.y = old_x, old_y
            bot.heading += random.choice([-1, 1]) * 0.9

    def choose_search_waypoint(self) -> Tuple[float, float]:
        wp = self.sweep_points[self.sweep_index % len(self.sweep_points)]
        if dist((self.robot.x, self.robot.y), wp) <= AUTOPILOT_WAYPOINT_REACH:
            self.sweep_index += 1
            wp = self.sweep_points[self.sweep_index % len(self.sweep_points)]
        return wp

    def choose_random_explore_waypoint(self) -> Tuple[float, float]:
        best = None
        best_score = -1e9
        for _ in range(100):
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(RANDOM_EXPLORE_MIN_RADIUS, RANDOM_EXPLORE_MAX_RADIUS)
            x = clamp(self.robot.x + math.cos(angle) * radius, 40, WORLD_WIDTH - 40)
            y = clamp(self.robot.y + math.sin(angle) * radius, 40, WORLD_HEIGHT - 40)
            if self.collides(x, y, ROBOT_RADIUS + 4):
                continue
            revisit_penalty = 0.0
            for px, py, t in self.recent_positions:
                age = self.time_elapsed - t
                if age <= REVISIT_PENALTY_TIME:
                    d = dist((x, y), (px, py))
                    if d < REVISIT_PENALTY_RADIUS:
                        revisit_penalty += (REVISIT_PENALTY_RADIUS - d)
            score = radius - 1.8 * revisit_penalty
            if score > best_score:
                best_score = score
                best = (x, y)
        return best if best is not None else self.choose_search_waypoint()

    def choose_local_probe_waypoint(self) -> Tuple[float, float]:
        if self.local_probe_center is None:
            return self.choose_search_waypoint()
        cx, cy = self.local_probe_center
        best = None
        best_score = -1e9
        for _ in range(120):
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(LOCAL_PROBE_MIN_RADIUS, LOCAL_PROBE_MAX_RADIUS)
            x = clamp(cx + math.cos(angle) * radius, 40, WORLD_WIDTH - 40)
            y = clamp(cy + math.sin(angle) * radius, 40, WORLD_HEIGHT - 40)
            if self.collides(x, y, ROBOT_RADIUS + 4):
                continue
            revisit_penalty = 0.0
            for px, py, t in self.recent_positions:
                age = self.time_elapsed - t
                if age <= REVISIT_PENALTY_TIME:
                    d = dist((x, y), (px, py))
                    if d < REVISIT_PENALTY_RADIUS:
                        revisit_penalty += (REVISIT_PENALTY_RADIUS - d)
            dist_from_center = dist((x, y), (cx, cy))
            perimeter_bonus = abs(dist_from_center - ((LOCAL_PROBE_MIN_RADIUS + LOCAL_PROBE_MAX_RADIUS) / 2.0))
            score = 140.0 - perimeter_bonus - 1.5 * revisit_penalty
            if score > best_score:
                best_score = score
                best = (x, y)
        return best if best is not None else (cx, cy)

    def update_stall_logic(self):
        pos = (self.robot.x, self.robot.y)
        self.recent_positions.append((pos[0], pos[1], self.time_elapsed))
        self.recent_positions = [p for p in self.recent_positions if (self.time_elapsed - p[2]) <= REVISIT_PENALTY_TIME]
        if dist(pos, self.stall_anchor) > STALL_RADIUS:
            self.stall_anchor = pos
            self.stall_start_time = self.time_elapsed
            return
        stalled_too_long = (self.time_elapsed - self.stall_start_time) >= STALL_TIME
        if stalled_too_long and self.time_elapsed >= self.random_explore_until:
            self.current_waypoint = self.choose_random_explore_waypoint()
            self.random_explore_until = self.time_elapsed + RANDOM_EXPLORE_TIME
            self.search_mode = "random-explore"
            self.last_good_estimate = None
            self.last_home_target = None
            self.stall_anchor = self.current_waypoint
            self.stall_start_time = self.time_elapsed

    def update_autopilot(self, dt: float):
        if self.robot.found_target:
            self.robot.speed = 0.0
            self.search_mode = "found"
            return
        if self.time_elapsed < self.random_explore_until and self.current_waypoint is not None:
            self.search_mode = "random-explore"
            self.steer_bot_toward(self.robot, self.current_waypoint, dt, ROBOT_SPEED)
            return
        fresh_estimate = self.last_good_estimate is not None and (self.time_elapsed - self.last_estimate_time) <= ESTIMATE_STICK_TIME
        if fresh_estimate:
            self.local_probe_center = self.last_good_estimate
        if fresh_estimate and self.search_mode not in ("local-probe",):
            if self.search_mode != "homing" or self.last_home_target != self.last_good_estimate:
                self.home_mode_started = self.time_elapsed
                self.last_home_target = self.last_good_estimate
            self.search_mode = "homing"
            self.current_waypoint = self.last_good_estimate
            near_home_target = dist((self.robot.x, self.robot.y), self.current_waypoint) <= HOME_REACHED_DIST
            stuck_homing = (self.time_elapsed - self.home_mode_started) > MAX_HOME_TIME
            if near_home_target or stuck_homing:
                self.search_mode = "local-probe"
                self.local_probe_center = self.last_good_estimate
                self.local_probe_until = self.time_elapsed + LOCAL_PROBE_TIME
                self.local_probe_points_visited = 0
                self.current_waypoint = self.choose_local_probe_waypoint()
        elif self.search_mode == "local-probe" and self.local_probe_center is not None:
            if fresh_estimate:
                self.local_probe_center = self.last_good_estimate
            if self.current_waypoint is None:
                self.current_waypoint = self.choose_local_probe_waypoint()
            reached_probe_point = dist((self.robot.x, self.robot.y), self.current_waypoint) <= LOCAL_PROBE_REACHED_DIST
            probe_time_expired = self.time_elapsed >= self.local_probe_until
            enough_points = self.local_probe_points_visited >= LOCAL_PROBE_POINTS_BEFORE_EXIT
            if reached_probe_point:
                self.local_probe_points_visited += 1
                self.current_waypoint = self.choose_local_probe_waypoint()
            if probe_time_expired and enough_points and not fresh_estimate:
                self.search_mode = "sweep"
                self.current_waypoint = self.choose_search_waypoint()
                self.local_probe_center = None
                self.last_home_target = None
                self.last_good_estimate = None
        else:
            self.search_mode = "sweep"
            self.current_waypoint = self.choose_search_waypoint()
        self.update_stall_logic()
        self.steer_bot_toward(self.robot, self.current_waypoint, dt, ROBOT_SPEED)

    def update_random_baseline(self, dt: float):
        if not self.random_baseline_enabled or self.random_bot.found_target:
            self.random_bot.speed = 0.0
            return
        if self.random_bot.waypoint is None or self.time_elapsed >= self.random_bot.repath_at:
            self.random_bot.waypoint = self.random_free_point(margin=60.0)
            self.random_bot.repath_at = self.time_elapsed + RANDOM_BASELINE_REPATH_TIME
        if dist((self.random_bot.x, self.random_bot.y), self.random_bot.waypoint) <= RANDOM_BASELINE_WAYPOINT_REACHED:
            self.random_bot.waypoint = self.random_free_point(margin=60.0)
            self.random_bot.repath_at = self.time_elapsed + RANDOM_BASELINE_REPATH_TIME
        self.steer_bot_toward(self.random_bot, self.random_bot.waypoint, dt, RANDOM_BOT_SPEED)

    def handle_input(self, dt: float):
        keys = pygame.key.get_pressed()
        if self.autopilot_enabled:
            self.update_autopilot(dt)
        else:
            turn = 0.0
            speed_cmd = 0.0
            if keys[pygame.K_a]: turn -= TURN_SPEED
            if keys[pygame.K_d]: turn += TURN_SPEED
            if keys[pygame.K_w]: speed_cmd += ROBOT_SPEED
            if keys[pygame.K_s]: speed_cmd -= ROBOT_SPEED * 0.6
            self.robot.heading += turn * dt
            self.robot.speed = speed_cmd
            old_x, old_y = self.robot.x, self.robot.y
            self.robot.forward(dt)
            if self.collides(self.robot.x, self.robot.y, ROBOT_RADIUS):
                self.robot.x, self.robot.y = old_x, old_y
        self.update_random_baseline(dt)
        for bot in (self.robot, self.random_bot):
            if not bot.trail or dist(bot.trail[-1], (bot.x, bot.y)) > 8:
                bot.trail.append((bot.x, bot.y))
                if len(bot.trail) > 700:
                    bot.trail.pop(0)
        if keys[pygame.K_LEFT]:
            self.camera[0] -= CAMERA_PAN_SPEED * dt
            self.manual_camera = True
        if keys[pygame.K_RIGHT]:
            self.camera[0] += CAMERA_PAN_SPEED * dt
            self.manual_camera = True
        if keys[pygame.K_UP]:
            self.camera[1] -= CAMERA_PAN_SPEED * dt
            self.manual_camera = True
        if keys[pygame.K_DOWN]:
            self.camera[1] += CAMERA_PAN_SPEED * dt
            self.manual_camera = True
        self.camera[0] = clamp(self.camera[0], 0, WORLD_WIDTH - SIM_WIDTH)
        self.camera[1] = clamp(self.camera[1], 0, WORLD_HEIGHT - SIM_HEIGHT)

    def update(self, dt: float):
        self.steps += 1
        self.time_elapsed += dt
        self.update_target_motion(dt)
        if self.time_elapsed - self.last_emit_time >= VICTIM_EMIT_PERIOD:
            self.emit_distress_packet()
            self.last_emit_time = self.time_elapsed
        self.prune_packets()
        self.handle_input(dt)
        self.robot.last_observations = self.generate_robot_observations()
        self.last_robot_decodes = [o for o in self.robot.last_observations if o.delivered]
        self.last_robot_decodes.sort(key=lambda o: (o.robot_link_rssi + 12.0 * o.freshness - 2.2 * o.hop_count), reverse=True)
        self.last_robot_decodes = self.last_robot_decodes[:10]
        self.update_estimate()
        for bot in (self.robot, self.random_bot):
            if not bot.found_target and dist((bot.x, bot.y), (self.target.x, self.target.y)) <= TARGET_REVEAL_RANGE:
                bot.found_target = True
                bot.found_time = self.time_elapsed
        self.target.found = self.robot.found_target or self.random_bot.found_target
        if not self.manual_camera:
            self.camera[0] = clamp(self.robot.x - SIM_WIDTH / 2, 0, WORLD_WIDTH - SIM_WIDTH)
            self.camera[1] = clamp(self.robot.y - SIM_HEIGHT / 2, 0, WORLD_HEIGHT - SIM_HEIGHT)

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        return int(x - self.camera[0]), int(y - self.camera[1])

    def draw_grid(self, screen):
        spacing = 100
        start_x = int(self.camera[0] // spacing) * spacing
        start_y = int(self.camera[1] // spacing) * spacing
        for x in range(start_x, int(self.camera[0] + SIM_WIDTH) + spacing, spacing):
            sx1, sy1 = self.world_to_screen(x, 0)
            sx2, sy2 = self.world_to_screen(x, WORLD_HEIGHT)
            pygame.draw.line(screen, GRID_COLOR, (sx1, sy1), (sx2, sy2), 1)
        for y in range(start_y, int(self.camera[1] + SIM_HEIGHT) + spacing, spacing):
            sx1, sy1 = self.world_to_screen(0, y)
            sx2, sy2 = self.world_to_screen(WORLD_WIDTH, y)
            pygame.draw.line(screen, GRID_COLOR, (sx1, sy1), (sx2, sy2), 1)

    def draw_signal_overlay(self, screen):
        if not self.debug_draw_signal:
            return
        cell = 60
        surface = pygame.Surface((SIM_WIDTH, SIM_HEIGHT), pygame.SRCALPHA)
        active_nodes = [(node_id, pkt) for node_id, pkt in self.active_packets.items() if pkt.freshness > 0]
        if not active_nodes:
            return
        for sy in range(0, SIM_HEIGHT, cell):
            for sx in range(0, SIM_WIDTH, cell):
                wx = sx + self.camera[0] + cell / 2
                wy = sy + self.camera[1] + cell / 2
                score = 0.0
                for node_id, pkt in active_nodes:
                    node = self.nodes[node_id]
                    d = dist((wx, wy), (node.x, node.y))
                    node_score = max(0.0, (pkt.source_rssi + 120.0)) * pkt.freshness / (1.0 + d / 180.0 + 0.9 * pkt.hop_count)
                    score += node_score
                norm = clamp(score / 55.0, 0.0, 1.0)
                alpha = int(90 * norm)
                pygame.draw.rect(surface, (220, 40, 40, alpha), (sx, sy, cell, cell))
        screen.blit(surface, (0, 0))

    def draw_node_links(self, screen):
        drawn = set()
        for i, nbrs in self.node_neighbors.items():
            if self.nodes[i].failed:
                continue
            for j in nbrs:
                edge = tuple(sorted((i, j)))
                if edge in drawn:
                    continue
                drawn.add(edge)
                a = self.nodes[i]
                b = self.nodes[j]
                ax, ay = self.world_to_screen(a.x, a.y)
                bx, by = self.world_to_screen(b.x, b.y)
                pygame.draw.line(screen, NODE_LINK_COLOR, (ax, ay), (bx, by), 1)

    def draw_packet_flow(self, screen):
        if not self.show_packets:
            return
        for i, j, t in self.visible_packet_edges:
            age = self.time_elapsed - t
            alpha_scale = clamp(1.0 - age / 0.6, 0.0, 1.0)
            a = self.nodes[i]
            b = self.nodes[j]
            ax, ay = self.world_to_screen(a.x, a.y)
            bx, by = self.world_to_screen(b.x, b.y)
            overlay = pygame.Surface((SIM_WIDTH, SIM_HEIGHT), pygame.SRCALPHA)
            pygame.draw.line(overlay, (*PACKET_COLOR, int(180 * alpha_scale)), (ax, ay), (bx, by), 3)
            screen.blit(overlay, (0, 0))
        for node_id, pkt in self.active_packets.items():
            if (self.time_elapsed - pkt.delivered_time) > 0.7:
                continue
            node = self.nodes[node_id]
            sx, sy = self.world_to_screen(node.x, node.y)
            radius = int(10 + 10 * pkt.freshness)
            pygame.draw.circle(screen, PACKET_COLOR, (sx, sy), radius, 1)

    def draw_bot(self, screen, bot: Robot, color: Tuple[int, int, int], label_text: str, trail_color: Tuple[int, int, int]):
        if len(bot.trail) > 1:
            trail_points = [self.world_to_screen(x, y) for x, y in bot.trail]
            pygame.draw.lines(screen, trail_color, False, trail_points, 2)
        rx, ry = self.world_to_screen(bot.x, bot.y)
        pygame.draw.circle(screen, color, (rx, ry), ROBOT_RADIUS)
        pygame.draw.circle(screen, (255, 255, 0), (rx, ry), ROBOT_RADIUS + 8, 2)
        bot_label = self.font.render(label_text, True, color)
        screen.blit(bot_label, (rx + 18, ry - 10))
        nose = (int(rx + math.cos(bot.heading) * 18), int(ry + math.sin(bot.heading) * 18))
        pygame.draw.line(screen, (255, 255, 255), (rx, ry), nose, 3)
        pygame.draw.circle(screen, (255, 255, 255), (rx, ry), ROBOT_RADIUS, 2)

    def draw_panel_background(self, screen, rect: pygame.Rect, title: str):
        pygame.draw.rect(screen, PANEL_BG, rect, border_radius=12)
        pygame.draw.rect(screen, PANEL_BORDER, rect, 2, border_radius=12)
        title_surf = self.big_font.render(title, True, TEXT_COLOR)
        screen.blit(title_surf, (rect.x + 14, rect.y + 10))
        pygame.draw.line(screen, PANEL_BORDER, (rect.x, rect.y + 42), (rect.x + rect.w, rect.y + 42), 2)

    def draw_section_header(self, screen, x: int, y: int, text: str):
        surf = self.font.render(text, True, MUTED_TEXT)
        screen.blit(surf, (x, y))

    def format_time(self, t: Optional[float]) -> str:
        return f"{t:0.1f}s" if t is not None else "--"

    def draw_kv_lines(self, screen, rect: pygame.Rect, rows: List[Tuple[str, str, Tuple[int, int, int]]], start_y: int = 56, line_h: int = 22):
        y = rect.y + start_y
        for k, v, color in rows:
            ks = self.font.render(k, True, TEXT_COLOR)
            vs = self.font.render(v, True, color)
            screen.blit(ks, (rect.x + 14, y))
            screen.blit(vs, (rect.x + 320, y))
            y += line_h
            if y > rect.bottom - 24:
                break

    def right_content_height(self) -> int:
        return STATS_HEIGHT + DECODE_HEIGHT + BASELINE_HEIGHT + HELP_HEIGHT + 3 * PANEL_GAP

    def right_content_width(self) -> int:
        return RIGHT_CONTENT_WIDTH

    def draw_dashboard(self, screen):
        screen.fill(BG_COLOR)

        sim_rect = pygame.Rect(SIM_X, SIM_Y, SIM_WIDTH, SIM_HEIGHT)
        pygame.draw.rect(screen, SIM_BG_COLOR, sim_rect, border_radius=12)
        pygame.draw.rect(screen, PANEL_BORDER, sim_rect, 2, border_radius=12)
        sim_surface = pygame.Surface((SIM_WIDTH, SIM_HEIGHT))
        sim_surface.fill(SIM_BG_COLOR)

        self.draw_grid(sim_surface)
        self.draw_signal_overlay(sim_surface)
        self.draw_node_links(sim_surface)
        self.draw_packet_flow(sim_surface)

        for ob in self.obstacles:
            sx, sy = self.world_to_screen(ob.x, ob.y)
            pygame.draw.circle(sim_surface, TREE_COLOR, (sx, sy), int(ob.r))
            pygame.draw.circle(sim_surface, TREE_TRUNK, (sx, sy), max(4, int(ob.r * 0.28)))

        for i, node in enumerate(self.nodes):
            sx, sy = self.world_to_screen(node.x, node.y)
            color = (120, 120, 120) if node.failed else NODE_COLOR
            pygame.draw.circle(sim_surface, color, (sx, sy), NODE_RADIUS)
            label = self.small_font.render(str(i), True, TEXT_COLOR)
            sim_surface.blit(label, (sx + 10, sy - 10))

        if self.best_estimate:
            ex, ey = self.world_to_screen(*self.best_estimate)
            pygame.draw.circle(sim_surface, ESTIMATE_COLOR, (ex, ey), 18, 2)
            pygame.draw.line(sim_surface, ESTIMATE_COLOR, self.world_to_screen(self.robot.x, self.robot.y), (ex, ey), 1)
            est_label = self.font.render("EST", True, ESTIMATE_COLOR)
            sim_surface.blit(est_label, (ex + 20, ey - 8))

        tx, ty = self.world_to_screen(self.target.x, self.target.y)
        if self.show_target_always or self.target.found:
            pygame.draw.circle(sim_surface, TARGET_COLOR, (tx, ty), TARGET_RADIUS)
            pygame.draw.circle(sim_surface, (255, 255, 255), (tx, ty), TARGET_RADIUS + 4, 2)
            pygame.draw.circle(sim_surface, (255, 230, 120), (tx, ty), int(self.target.distress_radius), 2)
            person_label = self.font.render("PERSON IN DISTRESS", True, TARGET_COLOR)
            sim_surface.blit(person_label, (tx + 18, ty - 12))
        elif dist((self.robot.x, self.robot.y), (self.target.x, self.target.y)) <= 220:
            pygame.draw.circle(sim_surface, VICTIM_HINT_COLOR, (tx, ty), 8, 1)

        self.draw_bot(sim_surface, self.robot, ROBOT_COLOR, "GUIDED", TRAIL_COLOR)
        if self.random_baseline_enabled:
            self.draw_bot(sim_surface, self.random_bot, RANDOM_BOT_COLOR, "RANDOM", RANDOM_TRAIL_COLOR)

        if self.current_waypoint is not None and not self.robot.found_target:
            wx, wy = self.world_to_screen(*self.current_waypoint)
            pygame.draw.circle(sim_surface, (40, 40, 40), (wx, wy), 10, 1)
            wp = self.font.render("WP", True, (40, 40, 40))
            sim_surface.blit(wp, (wx + 12, wy - 10))

        if self.random_baseline_enabled and self.random_bot.waypoint is not None and not self.random_bot.found_target:
            wx, wy = self.world_to_screen(*self.random_bot.waypoint)
            pygame.draw.circle(sim_surface, (90, 90, 90), (wx, wy), 8, 1)
            wp = self.small_font.render("RWP", True, (90, 90, 90))
            sim_surface.blit(wp, (wx + 10, wy - 10))

        if self.target.found:
            if self.robot.found_time is not None and (self.random_bot.found_time is None or self.robot.found_time <= self.random_bot.found_time):
                winner = "GUIDED ROBOT FOUND VICTIM"
                color = ROBOT_COLOR
            elif self.random_bot.found_time is not None:
                winner = "RANDOM ROBOT FOUND VICTIM"
                color = RANDOM_BOT_COLOR
            else:
                winner = "VICTIM FOUND"
                color = TARGET_COLOR
            msg = self.big_font.render(winner, True, color)
            rect = msg.get_rect(center=(SIM_WIDTH // 2, 30))
            sim_surface.blit(msg, rect)

        screen.blit(sim_surface, (SIM_X, SIM_Y))

        right_view_rect = pygame.Rect(RIGHT_X, TOP_Y, RIGHT_VIEW_WIDTH, RIGHT_VIEW_HEIGHT)
        pygame.draw.rect(screen, (245, 247, 244), right_view_rect, border_radius=12)
        pygame.draw.rect(screen, PANEL_BORDER, right_view_rect, 2, border_radius=12)

        content_h = self.right_content_height()
        self.panel_scroll = int(clamp(self.panel_scroll, 0, max(0, content_h - RIGHT_VIEW_HEIGHT)))
        content = pygame.Surface((RIGHT_CONTENT_WIDTH, content_h), pygame.SRCALPHA)
        content.fill((245, 247, 244, 0))

        stats_rect = pygame.Rect(0, 0, RIGHT_CONTENT_WIDTH, STATS_HEIGHT)
        decode_rect = pygame.Rect(0, STATS_HEIGHT + PANEL_GAP, RIGHT_CONTENT_WIDTH, DECODE_HEIGHT)
        baseline_rect = pygame.Rect(0, STATS_HEIGHT + DECODE_HEIGHT + 2 * PANEL_GAP, RIGHT_CONTENT_WIDTH, BASELINE_HEIGHT)
        help_rect = pygame.Rect(0, STATS_HEIGHT + DECODE_HEIGHT + BASELINE_HEIGHT + 3 * PANEL_GAP, RIGHT_CONTENT_WIDTH, HELP_HEIGHT)

        self.draw_panel_background(content, stats_rect, "Project Stats")
        delivered = [o for o in self.robot.last_observations if o.delivered]
        avg_hops = sum(pkt.hop_count for pkt in self.active_packets.values()) / max(1, len(self.active_packets))
        avg_fresh = sum(pkt.freshness for pkt in self.active_packets.values()) / max(1, len(self.active_packets))
        avg_degree = sum(len(v) for v in self.node_neighbors.values()) / max(1, len(self.node_neighbors))
        winner = "Guided" if self.robot.found_time is not None and (self.random_bot.found_time is None or self.robot.found_time <= self.random_bot.found_time) else ("Random" if self.random_bot.found_time is not None else "None")
        stats_rows = [
            ("Sim time", f"{self.time_elapsed:0.1f}s", GOOD_COLOR),
            ("Robot type", "Flying drone" if self.flying_robot else "Ground robot", TEXT_COLOR),
            ("Victim", "MOVING" if self.target.moving else "STATIC", WARN_COLOR if self.target.moving else GOOD_COLOR),
            ("Victim position", f"({int(self.target.x)}, {int(self.target.y)})", TEXT_COLOR),
            ("Victim waypoint", f"{tuple(map(int, self.target.waypoint)) if self.target.waypoint else 'None'}", TEXT_COLOR),
            ("Nodes configured", str(self.num_nodes), TEXT_COLOR),
            ("Failed nodes", str(sum(1 for n in self.nodes if n.failed)), BAD_COLOR),
            ("Relay edges", str(sum(len(v) for v in self.node_neighbors.values()) // 2), TEXT_COLOR),
            ("Avg node degree", f"{avg_degree:0.2f}", TEXT_COLOR),
            ("Active packets", str(len(self.active_packets)), GOOD_COLOR if self.active_packets else WARN_COLOR),
            ("Avg packet hops", f"{avg_hops:0.2f}", TEXT_COLOR),
            ("Avg freshness", f"{avg_fresh:0.2f}", TEXT_COLOR),
            ("Guided mode", self.search_mode, ROBOT_COLOR),
            ("Guided pos", f"({int(self.robot.x)}, {int(self.robot.y)})", ROBOT_COLOR),
            ("Guided heading", f"{math.degrees(self.robot.heading)%360:0.1f} deg", ROBOT_COLOR),
            ("Guided speed", f"{self.robot.speed:0.1f}", ROBOT_COLOR),
            ("Guided decodes", str(len(delivered)), ROBOT_COLOR),
            ("Best estimate", f"{tuple(map(int, self.best_estimate)) if self.best_estimate else 'None'}", ESTIMATE_COLOR),
            ("Estimate RSSI", f"{self.estimated_strength:0.1f} dBm" if self.estimated_strength else "None", ESTIMATE_COLOR),
            ("Current waypoint", f"{tuple(map(int, self.current_waypoint)) if self.current_waypoint else 'None'}", TEXT_COLOR),
            ("Local probe center", f"{tuple(map(int, self.local_probe_center)) if self.local_probe_center else 'None'}", TEXT_COLOR),
            ("Random explore until", f"{self.random_explore_until:0.1f}s", TEXT_COLOR),
            ("Guided found time", self.format_time(self.robot.found_time), ROBOT_COLOR),
            ("Random found time", self.format_time(self.random_bot.found_time) if self.random_baseline_enabled else "--", RANDOM_BOT_COLOR),
            ("Winner so far", winner, GOOD_COLOR if winner == "Guided" else (WARN_COLOR if winner == "Random" else TEXT_COLOR)),
        ]
        self.draw_kv_lines(content, stats_rect, stats_rows)

        self.draw_panel_background(content, decode_rect, "Guided Packet Decodes")
        hdr = self.small_font.render("node   hops   fresh   src_rssi   link_rssi", True, TEXT_COLOR)
        content.blit(hdr, (decode_rect.x + 14, decode_rect.y + 52))
        y = decode_rect.y + 76
        if self.last_robot_decodes:
            for obs in self.last_robot_decodes:
                row = f"{obs.node_id:>4}   {obs.hop_count:>4}   {obs.freshness:>5.2f}   {obs.source_rssi:>8.1f}   {obs.robot_link_rssi:>8.1f}"
                surf = self.small_font.render(row, True, TEXT_COLOR)
                content.blit(surf, (decode_rect.x + 14, y))
                y += 20
        else:
            surf = self.font.render("No successful decodes yet", True, WARN_COLOR)
            content.blit(surf, (decode_rect.x + 14, y))
        decode_extra = [
            ("Emission period", f"{VICTIM_EMIT_PERIOD:0.1f}s", TEXT_COLOR),
            ("Packet TTL", f"{PACKET_TTL:0.1f}s", TEXT_COLOR),
            ("Max hops", str(MAX_PACKET_HOPS), TEXT_COLOR),
            ("Sensor range", f"{SENSOR_RANGE:.0f}", TEXT_COLOR),
            ("Base edge success", f"{EDGE_BASE_SUCCESS:0.2f}", TEXT_COLOR),
            ("Dist penalty", f"{EDGE_DISTANCE_PENALTY:0.2f}", TEXT_COLOR),
            ("Blocker penalty", f"{EDGE_BLOCKER_PENALTY:0.2f}", TEXT_COLOR),
            ("Ranking", "link_rssi + freshness - hop penalty", MUTED_TEXT),
        ]
        self.draw_kv_lines(content, decode_rect, decode_extra, start_y=DECODE_HEIGHT - 170, line_h=20)

        self.draw_panel_background(content, baseline_rect, "Baseline Comparison")
        baseline_rows = [
            ("Guided status", "FOUND" if self.robot.found_target else "SEARCHING", ROBOT_COLOR),
            ("Guided time", self.format_time(self.robot.found_time), ROBOT_COLOR),
            ("Random baseline", "Enabled" if self.random_baseline_enabled else "Disabled", RANDOM_BOT_COLOR),
            ("Random status", ("FOUND" if self.random_bot.found_target else "SEARCHING") if self.random_baseline_enabled else "--", RANDOM_BOT_COLOR),
            ("Random time", self.format_time(self.random_bot.found_time) if self.random_baseline_enabled else "--", RANDOM_BOT_COLOR),
            ("Random pos", f"({int(self.random_bot.x)}, {int(self.random_bot.y)})" if self.random_baseline_enabled else "--", RANDOM_BOT_COLOR),
            ("Random heading", f"{math.degrees(self.random_bot.heading)%360:0.1f} deg" if self.random_baseline_enabled else "--", RANDOM_BOT_COLOR),
            ("Random speed", f"{self.random_bot.speed:0.1f}" if self.random_baseline_enabled else "--", RANDOM_BOT_COLOR),
            ("Random waypoint", f"{tuple(map(int, self.random_bot.waypoint)) if self.random_bot.waypoint else 'None'}" if self.random_baseline_enabled else "--", RANDOM_BOT_COLOR),
            ("Anti-stall", f"{STALL_TIME:0.1f}s -> jump {RANDOM_EXPLORE_MIN_RADIUS:.0f}-{RANDOM_EXPLORE_MAX_RADIUS:.0f}", TEXT_COLOR),
            ("Local probe", f"{LOCAL_PROBE_TIME:0.1f}s / {LOCAL_PROBE_POINTS_BEFORE_EXIT} pts", TEXT_COLOR),
        ]
        self.draw_kv_lines(content, baseline_rect, baseline_rows)

        self.draw_panel_background(content, help_rect, "Guide / Controls")
        help_lines = [
            "Arrow keys = pan camera",
            "Mouse wheel over right panel = vertical scroll",
            "Shift + mouse wheel over right panel = horizontal scroll",
            "Page Up / Page Down = vertical scroll",
            "Home / End = horizontal scroll left / right",
            "C = recenter on guided robot",
            "M = guided autopilot/manual",
            "B = toggle random baseline",
            "WASD = move guided robot only when autopilot is off",
            "R = reset world with current settings",
            "F = flying/ground robot",
            "T = moving/static victim",
            "[ / ] = decrease/increase node count, then press R",
            "G = heat overlay, V = always-show victim",
            "P = packet flow",
            "Purple dots = relay nodes, orange lines = active packets",
            "Guided robot uses relayed packets; random ignores them",
        ]
        y = help_rect.y + 54
        for line in help_lines:
            surf = self.small_font.render(line, True, TEXT_COLOR)
            content.blit(surf, (help_rect.x + 14, y))
            y += 14

        clip = pygame.Rect(self.panel_scroll_x, self.panel_scroll, RIGHT_VIEW_WIDTH, RIGHT_VIEW_HEIGHT)
        screen.blit(content, (RIGHT_X, TOP_Y), area=clip)

        max_v = max(0, content_h - RIGHT_VIEW_HEIGHT)
        max_h = max(0, RIGHT_CONTENT_WIDTH - RIGHT_VIEW_WIDTH)
        scroll_hint = self.small_font.render(f"Right panel scroll V:{self.panel_scroll}px/{max_v}px  H:{self.panel_scroll_x}px/{max_h}px", True, MUTED_TEXT)
        screen.blit(scroll_hint, (RIGHT_X + 10, WINDOW_HEIGHT - 24))
        footer = self.small_font.render("Single dashboard window: simulation left, scrollable stats/decodes/baseline/help right.", True, MUTED_TEXT)
        screen.blit(footer, (24, WINDOW_HEIGHT - 24))
