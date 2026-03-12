import math
from typing import List, Tuple, Optional
import pygame

from config.settings import *
from core.models import Robot
from utils.math_utils import clamp, dist

class DashboardRenderer:
    def __init__(self):
        self.debug_draw_signal = False
        self.show_target_always = True
        self.show_packets = True
        self.panel_scroll = 0
        self.panel_scroll_x = 0
        self.setup_fonts()

    def setup_fonts(self):
        self.font = pygame.font.SysFont("consolas", 17)
        self.big_font = pygame.font.SysFont("consolas", 23, bold=True)
        self.small_font = pygame.font.SysFont("consolas", 13)

    def world_to_screen(self, env, x: float, y: float) -> Tuple[int, int]:
        return int(x - env.camera[0]), int(y - env.camera[1])

    def draw_grid(self, env, screen):
        spacing = 100
        start_x = int(env.camera[0] // spacing) * spacing
        start_y = int(env.camera[1] // spacing) * spacing
        for x in range(start_x, int(env.camera[0] + SIM_WIDTH) + spacing, spacing):
            sx1, sy1 = self.world_to_screen(env, x, 0)
            sx2, sy2 = self.world_to_screen(env, x, WORLD_HEIGHT)
            pygame.draw.line(screen, GRID_COLOR, (sx1, sy1), (sx2, sy2), 1)
        for y in range(start_y, int(env.camera[1] + SIM_HEIGHT) + spacing, spacing):
            sx1, sy1 = self.world_to_screen(env, 0, y)
            sx2, sy2 = self.world_to_screen(env, WORLD_WIDTH, y)
            pygame.draw.line(screen, GRID_COLOR, (sx1, sy1), (sx2, sy2), 1)

    def draw_signal_overlay(self, env, screen):
        if not self.debug_draw_signal:
            return
        cell = 60
        surface = pygame.Surface((SIM_WIDTH, SIM_HEIGHT), pygame.SRCALPHA)
        active_nodes = [(node_id, pkt) for node_id, pkt in env.active_packets.items() if pkt.freshness > 0]
        if not active_nodes:
            return
        for sy in range(0, SIM_HEIGHT, cell):
            for sx in range(0, SIM_WIDTH, cell):
                wx = sx + env.camera[0] + cell / 2
                wy = sy + env.camera[1] + cell / 2
                score = 0.0
                for node_id, pkt in active_nodes:
                    node = env.nodes[node_id]
                    d = dist((wx, wy), (node.x, node.y))
                    node_score = max(0.0, (pkt.source_rssi + 120.0)) * pkt.freshness / (1.0 + d / 180.0 + 0.9 * pkt.hop_count)
                    score += node_score
                norm = clamp(score / 55.0, 0.0, 1.0)
                alpha = int(90 * norm)
                pygame.draw.rect(surface, (220, 40, 40, alpha), (sx, sy, cell, cell))
        screen.blit(surface, (0, 0))

    def draw_node_links(self, env, screen):
        drawn = set()
        for i, nbrs in env.node_neighbors.items():
            if env.nodes[i].failed:
                continue
            for j in nbrs:
                edge = tuple(sorted((i, j)))
                if edge in drawn:
                    continue
                drawn.add(edge)
                a = env.nodes[i]
                b = env.nodes[j]
                ax, ay = self.world_to_screen(env, a.x, a.y)
                bx, by = self.world_to_screen(env, b.x, b.y)
                pygame.draw.line(screen, NODE_LINK_COLOR, (ax, ay), (bx, by), 1)

    def draw_packet_flow(self, env, screen):
        if not self.show_packets:
            return
        for i, j, t in env.visible_packet_edges:
            age = env.time_elapsed - t
            alpha_scale = clamp(1.0 - age / 0.6, 0.0, 1.0)
            a = env.nodes[i]
            b = env.nodes[j]
            ax, ay = self.world_to_screen(env, a.x, a.y)
            bx, by = self.world_to_screen(env, b.x, b.y)
            overlay = pygame.Surface((SIM_WIDTH, SIM_HEIGHT), pygame.SRCALPHA)
            pygame.draw.line(overlay, (*PACKET_COLOR, int(180 * alpha_scale)), (ax, ay), (bx, by), 3)
            screen.blit(overlay, (0, 0))
        for node_id, pkt in env.active_packets.items():
            if (env.time_elapsed - pkt.delivered_time) > 0.7:
                continue
            node = env.nodes[node_id]
            sx, sy = self.world_to_screen(env, node.x, node.y)
            radius = int(10 + 10 * pkt.freshness)
            pygame.draw.circle(screen, PACKET_COLOR, (sx, sy), radius, 1)

    def draw_bot(self, env, screen, bot: Robot, color: Tuple[int, int, int], label_text: str, trail_color: Tuple[int, int, int]):
        if len(bot.trail) > 1:
            trail_points = [self.world_to_screen(env, x, y) for x, y in bot.trail]
            pygame.draw.lines(screen, trail_color, False, trail_points, 2)
        rx, ry = self.world_to_screen(env, bot.x, bot.y)
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

    def format_time(self, t: Optional[float]) -> str:
        return f"{t:0.1f}s" if t is not None else "--"

    def draw_kv_lines(self, screen, rect: pygame.Rect, rows: List[Tuple[str, str, Tuple[int, int, int]]], start_y: int = 56, line_h: int = 20):
        y = rect.y + start_y
        for k, v, color in rows:
            ks = self.font.render(k, True, TEXT_COLOR)
            vs = self.font.render(v, True, color)
            screen.blit(ks, (rect.x + 14, y))
            screen.blit(vs, (rect.x + 240, y))  
            y += line_h
            if y > rect.bottom - 24:
                break

    def right_content_height(self) -> int:
        return STATS_HEIGHT + DECODE_HEIGHT + BASELINE_HEIGHT + HELP_HEIGHT + 3 * PANEL_GAP

    def right_content_width(self) -> int:
        return RIGHT_CONTENT_WIDTH

    def render(self, env, screen):
        screen.fill(BG_COLOR)

        sim_rect = pygame.Rect(SIM_X, SIM_Y, SIM_WIDTH, SIM_HEIGHT)
        pygame.draw.rect(screen, SIM_BG_COLOR, sim_rect, border_radius=12)
        pygame.draw.rect(screen, PANEL_BORDER, sim_rect, 2, border_radius=12)
        sim_surface = pygame.Surface((SIM_WIDTH, SIM_HEIGHT))
        sim_surface.fill(SIM_BG_COLOR)

        self.draw_grid(env, sim_surface)
        self.draw_signal_overlay(env, sim_surface)
        self.draw_node_links(env, sim_surface)
        self.draw_packet_flow(env, sim_surface)

        for ob in env.obstacles:
            sx, sy = self.world_to_screen(env, ob.x, ob.y)
            pygame.draw.circle(sim_surface, TREE_COLOR, (sx, sy), int(ob.r))
            pygame.draw.circle(sim_surface, TREE_TRUNK, (sx, sy), max(4, int(ob.r * 0.28)))

        for i, node in enumerate(env.nodes):
            sx, sy = self.world_to_screen(env, node.x, node.y)
            color = (120, 120, 120) if node.failed else NODE_COLOR
            pygame.draw.circle(sim_surface, color, (sx, sy), NODE_RADIUS)
            label = self.small_font.render(str(i), True, TEXT_COLOR)
            sim_surface.blit(label, (sx + 10, sy - 10))

        if env.best_estimate:
            ex, ey = self.world_to_screen(env, *env.best_estimate)
            pygame.draw.circle(sim_surface, ESTIMATE_COLOR, (ex, ey), 18, 2)
            pygame.draw.line(sim_surface, ESTIMATE_COLOR, self.world_to_screen(env, env.robot.x, env.robot.y), (ex, ey), 1)
            est_label = self.font.render("EST", True, ESTIMATE_COLOR)
            sim_surface.blit(est_label, (ex + 20, ey - 8))

        tx, ty = self.world_to_screen(env, env.target.x, env.target.y)
        if self.show_target_always or env.target.found:
            pygame.draw.circle(sim_surface, TARGET_COLOR, (tx, ty), TARGET_RADIUS)
            pygame.draw.circle(sim_surface, (255, 255, 255), (tx, ty), TARGET_RADIUS + 4, 2)
            pygame.draw.circle(sim_surface, (255, 230, 120), (tx, ty), int(env.target.distress_radius), 2)
            person_label = self.font.render("PERSON IN DISTRESS", True, TARGET_COLOR)
            sim_surface.blit(person_label, (tx + 18, ty - 12))
        elif dist((env.robot.x, env.robot.y), (env.target.x, env.target.y)) <= 220:
            pygame.draw.circle(sim_surface, VICTIM_HINT_COLOR, (tx, ty), 8, 1)

        self.draw_bot(env, sim_surface, env.robot, ROBOT_COLOR, "GUIDED", TRAIL_COLOR)
        if env.random_baseline_enabled:
            self.draw_bot(env, sim_surface, env.random_bot, RANDOM_BOT_COLOR, "RANDOM", RANDOM_TRAIL_COLOR)

        if env.current_waypoint is not None and not env.robot.found_target:
            wx, wy = self.world_to_screen(env, *env.current_waypoint)
            pygame.draw.circle(sim_surface, (40, 40, 40), (wx, wy), 10, 1)
            wp = self.font.render("WP", True, (40, 40, 40))
            sim_surface.blit(wp, (wx + 12, wy - 10))

        if env.random_baseline_enabled and env.random_bot.waypoint is not None and not env.random_bot.found_target:
            wx, wy = self.world_to_screen(env, *env.random_bot.waypoint)
            pygame.draw.circle(sim_surface, (90, 90, 90), (wx, wy), 8, 1)
            wp = self.small_font.render("RWP", True, (90, 90, 90))
            sim_surface.blit(wp, (wx + 10, wy - 10))

        if env.target.found:
            if env.robot.found_time is not None and (env.random_bot.found_time is None or env.robot.found_time <= env.random_bot.found_time):
                winner = "GUIDED ROBOT FOUND VICTIM"
                color = ROBOT_COLOR
            elif env.random_bot.found_time is not None:
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
        delivered = [o for o in env.robot.last_observations if o.delivered]
        avg_hops = sum(pkt.hop_count for pkt in env.active_packets.values()) / max(1, len(env.active_packets))
        avg_fresh = sum(pkt.freshness for pkt in env.active_packets.values()) / max(1, len(env.active_packets))
        avg_degree = sum(len(v) for v in env.node_neighbors.values()) / max(1, len(env.node_neighbors))
        winner = "Guided" if env.robot.found_time is not None and (env.random_bot.found_time is None or env.robot.found_time <= env.random_bot.found_time) else ("Random" if env.random_bot.found_time is not None else "None")
        
        stats_rows = [
            ("Sim time", f"{env.time_elapsed:0.1f}s", GOOD_COLOR),
            ("Robot type", "Flying drone" if env.flying_robot else "Ground robot", TEXT_COLOR),
            ("Victim", "MOVING" if env.target.moving else "STATIC", WARN_COLOR if env.target.moving else GOOD_COLOR),
            ("Victim position", f"({int(env.target.x)}, {int(env.target.y)})", TEXT_COLOR),
            ("Victim waypoint", f"{tuple(map(int, env.target.waypoint)) if env.target.waypoint else 'None'}", TEXT_COLOR),
            ("Nodes configured", str(env.num_nodes), TEXT_COLOR),
            ("Failed nodes", str(sum(1 for n in env.nodes if n.failed)), BAD_COLOR),
            ("Relay edges", str(sum(len(v) for v in env.node_neighbors.values()) // 2), TEXT_COLOR),
            ("Avg node degree", f"{avg_degree:0.2f}", TEXT_COLOR),
            ("Active packets", str(len(env.active_packets)), GOOD_COLOR if env.active_packets else WARN_COLOR),
            ("Avg packet hops", f"{avg_hops:0.2f}", TEXT_COLOR),
            ("Avg freshness", f"{avg_fresh:0.2f}", TEXT_COLOR),
            ("Guided mode", env.search_mode, ROBOT_COLOR),
            ("Guided pos", f"({int(env.robot.x)}, {int(env.robot.y)})", ROBOT_COLOR),
            ("Guided heading", f"{math.degrees(env.robot.heading)%360:0.1f} deg", ROBOT_COLOR),
            ("Guided speed", f"{env.robot.speed:0.1f}", ROBOT_COLOR),
            ("Guided decodes", str(len(delivered)), ROBOT_COLOR),
            ("Best estimate", f"{tuple(map(int, env.best_estimate)) if env.best_estimate else 'None'}", ESTIMATE_COLOR),
            ("Estimate RSSI", f"{env.estimated_strength:0.1f} dBm" if env.estimated_strength else "None", ESTIMATE_COLOR),
            ("Current waypoint", f"{tuple(map(int, env.current_waypoint)) if env.current_waypoint else 'None'}", TEXT_COLOR),
            ("Local probe center", f"{tuple(map(int, env.local_probe_center)) if env.local_probe_center else 'None'}", TEXT_COLOR),
            ("Random explore until", f"{env.random_explore_until:0.1f}s", TEXT_COLOR),
            ("Guided found time", self.format_time(env.robot.found_time), ROBOT_COLOR),
            ("Random found time", self.format_time(env.random_bot.found_time) if env.random_baseline_enabled else "--", RANDOM_BOT_COLOR),
            ("Winner so far", winner, GOOD_COLOR if winner == "Guided" else (WARN_COLOR if winner == "Random" else TEXT_COLOR)),
        ]
        self.draw_kv_lines(content, stats_rect, stats_rows)

        self.draw_panel_background(content, decode_rect, "Guided Packet Decodes")
        hdr = self.small_font.render("node   hops   fresh   src_rssi   link_rssi", True, TEXT_COLOR)
        content.blit(hdr, (decode_rect.x + 14, decode_rect.y + 54))
        y = decode_rect.y + 76
        if env.last_robot_decodes:
            for obs in env.last_robot_decodes:
                row = f"{obs.node_id:>4}   {obs.hop_count:>4}   {obs.freshness:>5.2f}   {obs.source_rssi:>8.1f}   {obs.robot_link_rssi:>8.1f}"
                surf = self.small_font.render(row, True, TEXT_COLOR)
                content.blit(surf, (decode_rect.x + 14, y))
                y += 18
        else:
            surf = self.small_font.render("No successful decodes yet", True, WARN_COLOR)
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
        self.draw_kv_lines(content, decode_rect, decode_extra, start_y=DECODE_HEIGHT - 175, line_h=16)

        self.draw_panel_background(content, baseline_rect, "Baseline Comparison")
        baseline_rows = [
            ("Guided status", "FOUND" if env.robot.found_target else "SEARCHING", ROBOT_COLOR),
            ("Guided time", self.format_time(env.robot.found_time), ROBOT_COLOR),
            ("Random baseline", "Enabled" if env.random_baseline_enabled else "Disabled", RANDOM_BOT_COLOR),
            ("Random status", ("FOUND" if env.random_bot.found_target else "SEARCHING") if env.random_baseline_enabled else "--", RANDOM_BOT_COLOR),
            ("Random time", self.format_time(env.random_bot.found_time) if env.random_baseline_enabled else "--", RANDOM_BOT_COLOR),
            ("Random pos", f"({int(env.random_bot.x)}, {int(env.random_bot.y)})" if env.random_baseline_enabled else "--", RANDOM_BOT_COLOR),
            ("Random heading", f"{math.degrees(env.random_bot.heading)%360:0.1f} deg" if env.random_baseline_enabled else "--", RANDOM_BOT_COLOR),
            ("Random speed", f"{env.random_bot.speed:0.1f}" if env.random_baseline_enabled else "--", RANDOM_BOT_COLOR),
            ("Random waypoint", f"{tuple(map(int, env.random_bot.waypoint)) if env.random_bot.waypoint else 'None'}" if env.random_baseline_enabled else "--", RANDOM_BOT_COLOR),
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
        
        if max_v > 0 or max_h > 0:
            scroll_hint = self.small_font.render(f"Right panel scroll V:{self.panel_scroll}px/{max_v}px  H:{self.panel_scroll_x}px/{max_h}px", True, MUTED_TEXT)
            screen.blit(scroll_hint, (RIGHT_X + 10, WINDOW_HEIGHT - 24))
            
        footer = self.small_font.render("Single dashboard window: simulation left, scrollable stats/decodes/baseline/help right.", True, MUTED_TEXT)
        screen.blit(footer, (24, WINDOW_HEIGHT - 24))