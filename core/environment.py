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
        self.target.trail.append((self.target.x, self.target.y))

        self.robot = Robot(120, 120)
        self.random_bot = Robot(180, 120)

        self.info_bot = Robot(240, 120)
        self.info_bot_enabled = True
        self.info_waypoint: Optional[Tuple[float, float]] = None
        self.info_replan_at = 0.0
        self.info_recent_positions: List[Tuple[float, float, float]] = []

        self.camera = [0.0, 0.0]
        self.target_moving_enabled = False
        self.target.moving = self.target_moving_enabled
        self.manual_camera = False
        self.autopilot_enabled = True
        self.random_baseline_enabled = True

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
        self.robot.found_time = None
        self.robot.found_target = False
        self.random_bot.found_time = None
        self.random_bot.found_target = False
        self.random_bot.waypoint = None
        self.random_bot.repath_at = 0.0
        self.info_bot.found_time = None
        
        self.info_bot.found_target = False
        self.info_bot.waypoint = None
        self.info_bot.repath_at = 0.0
        self.info_waypoint = None
        self.info_replan_at = 0.0
        self.info_recent_positions = []

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
            if not self.target.trail or dist(self.target.trail[-1], (self.target.x, self.target.y)) > 6:
                self.target.trail.append((self.target.x, self.target.y))
            if len(self.target.trail) > 800:
                self.target.trail.pop(0)
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
                forwarded_rssi = pkt.source_rssi - 10 * 2 * math.log10(max(d, 1.0)) - blockers * random.uniform(0.4, 1.2) + random.gauss(0, EDGE_JITTER_STD)
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
            robot_link_rssi = pkt.source_rssi - 10 * 2 * math.log10(max(d_robot, 1.0)) - blockers * random.uniform(0.5, 1.3) + random.gauss(0, 1.1)
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
            RSSI_MIN = -200.0
            RSSI_MAX = -40.0

            signal = clamp(
                (obs.robot_link_rssi - RSSI_MIN) / (RSSI_MAX - RSSI_MIN),
                0.0,
                1.0
            )
            w = max(0.5, 4.5 * signal + 2.2 * obs.freshness - 0.35 * obs.hop_count)
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

    def update_info_bot_memory(self):
        self.info_recent_positions.append((self.info_bot.x, self.info_bot.y, self.time_elapsed))
        self.info_recent_positions = [
            p for p in self.info_recent_positions
            if (self.time_elapsed - p[2]) <= INFO_VISIT_MEMORY
        ]


    def sample_info_candidates(self) -> List[Tuple[float, float]]:
        candidates = []
        for _ in range(INFO_CANDIDATE_COUNT):
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(INFO_MIN_RADIUS, INFO_MAX_RADIUS)
            x = clamp(self.info_bot.x + math.cos(angle) * radius, 40, WORLD_WIDTH - 40)
            y = clamp(self.info_bot.y + math.sin(angle) * radius, 40, WORLD_HEIGHT - 40)
            if not self.collides(x, y, ROBOT_RADIUS + 4):
                candidates.append((x, y))
        return candidates


    def score_info_candidate(self, point: Tuple[float, float]) -> float:
        packet_score = 0.0
        for node_id, pkt in self.active_packets.items():
            node = self.nodes[node_id]
            d = dist(point, (node.x, node.y))

            strength = max(0.0, pkt.source_rssi + 180.0)

            packet_score +=  strength * pkt.freshness / (1.0 + d / 120.0 + 0.8 * pkt.hop_count)

        estimate_score = 0.0
        if self.best_estimate is not None:
            d_est = dist(point, self.best_estimate)
            estimate_score = 1.0 / (1.0 + d_est / 160.0)

        min_recent_dist = float("inf")
        revisit_penalty = 0.0
        for px, py, t in self.info_recent_positions:
            d = dist(point, (px, py))
            min_recent_dist = min(min_recent_dist, d)
            if d < INFO_VISIT_RADIUS:
                revisit_penalty += (INFO_VISIT_RADIUS - d)

        explore_score = 1.5 if min_recent_dist == float("inf") else min(min_recent_dist / INFO_VISIT_RADIUS, 1.5)

        desired_heading = math.atan2(point[1] - self.info_bot.y, point[0] - self.info_bot.x)
        turn_penalty = abs(angle_wrap(desired_heading - self.info_bot.heading))

        return (
            INFO_PACKET_WEIGHT * packet_score
            + INFO_ESTIMATE_WEIGHT * estimate_score
            + INFO_EXPLORE_WEIGHT * explore_score
            - INFO_REVISIT_WEIGHT * revisit_penalty
            - INFO_TURN_WEIGHT * turn_penalty
        )


    def choose_info_waypoint(self) -> Tuple[float, float]:
        candidates = self.sample_info_candidates()
        if not candidates:
            return (self.info_bot.x, self.info_bot.y)
        return max(candidates, key=self.score_info_candidate)

    def update_info_bot(self, dt: float):
        if not self.info_bot_enabled or self.info_bot.found_target:
            self.info_bot.speed = 0.0
            return

        if self.time_elapsed >= self.info_replan_at or self.info_waypoint is None:
            self.info_waypoint = self.choose_info_waypoint()
            self.info_replan_at = self.time_elapsed + INFO_REPLAN_PERIOD

        self.steer_bot_toward(self.info_bot, self.info_waypoint, dt, INFO_BOT_SPEED)
        self.update_info_bot_memory()


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

    def handle_input(self, dt: float, keys):
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
        self.update_info_bot(dt)

        for bot in (self.robot, self.random_bot, self.info_bot):
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

    def update(self, dt: float, keys):
        self.steps += 1
        self.time_elapsed += dt
        self.update_target_motion(dt)
        if self.time_elapsed - self.last_emit_time >= VICTIM_EMIT_PERIOD:
            self.emit_distress_packet()
            self.last_emit_time = self.time_elapsed
        self.prune_packets()
        
        self.handle_input(dt, keys)
        
        self.robot.last_observations = self.generate_robot_observations()
        self.last_robot_decodes = [o for o in self.robot.last_observations if o.delivered]
        self.last_robot_decodes.sort(key=lambda o: (o.robot_link_rssi + 12.0 * o.freshness - 2.2 * o.hop_count), reverse=True)
        self.last_robot_decodes = self.last_robot_decodes[:10]
        self.update_estimate()
        
        for bot in (self.robot, self.random_bot, self.info_bot):
            if not bot.found_target and dist((bot.x, bot.y), (self.target.x, self.target.y)) <= TARGET_REVEAL_RANGE:
                bot.found_target = True
                bot.found_time = self.time_elapsed
        self.target.found = self.robot.found_target or self.random_bot.found_target or self.info_bot.found_target
        
        if not self.manual_camera:
            self.camera[0] = clamp(self.robot.x - SIM_WIDTH / 2, 0, WORLD_WIDTH - SIM_WIDTH)
            self.camera[1] = clamp(self.robot.y - SIM_HEIGHT / 2, 0, WORLD_HEIGHT - SIM_HEIGHT)