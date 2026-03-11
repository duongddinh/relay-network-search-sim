import pygame
from config.settings import (
    WINDOW_WIDTH, WINDOW_HEIGHT, FPS, RIGHT_X, TOP_Y, 
    SIM_WIDTH, SIM_HEIGHT, WORLD_WIDTH, WORLD_HEIGHT, 
    MIN_NUM_NODES, MAX_NUM_NODES, NODE_STEP,
    TARGET_REPATH_TIME, RIGHT_VIEW_HEIGHT, RIGHT_VIEW_WIDTH
)
from core.environment import ForestSearchEnv
from utils.math_utils import clamp

def main():
    pygame.init()
    pygame.display.set_caption("Forest Search Dashboard")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    env = ForestSearchEnv()
    # Initialize fonts after pygame.init()
    env.font = pygame.font.SysFont("consolas", 17)
    env.big_font = pygame.font.SysFont("consolas", 23, bold=True)
    env.small_font = pygame.font.SysFont("consolas", 13)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                if mx >= RIGHT_X and my >= TOP_Y:
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_SHIFT:
                        env.panel_scroll_x -= event.y * 30
                    else:
                        env.panel_scroll -= event.y * 30
                        
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.reset()
                    env.font = pygame.font.SysFont("consolas", 17)
                    env.big_font = pygame.font.SysFont("consolas", 23, bold=True)
                    env.small_font = pygame.font.SysFont("consolas", 13)
                elif event.key == pygame.K_PAGEUP:
                    env.panel_scroll -= 120
                elif event.key == pygame.K_PAGEDOWN:
                    env.panel_scroll += 120
                elif event.key == pygame.K_HOME:
                    env.panel_scroll_x -= 120
                elif event.key == pygame.K_END:
                    env.panel_scroll_x += 120
                elif event.key == pygame.K_g:
                    env.debug_draw_signal = not env.debug_draw_signal
                elif event.key == pygame.K_v:
                    env.show_target_always = not env.show_target_always
                elif event.key == pygame.K_t:
                    env.target_moving_enabled = not env.target_moving_enabled
                    env.target.moving = env.target_moving_enabled
                    if env.target.moving and env.target.waypoint is None:
                        env.target.waypoint = env.random_free_point(margin=80.0)
                        env.target.repath_at = env.time_elapsed + TARGET_REPATH_TIME
                elif event.key == pygame.K_p:
                    env.show_packets = not env.show_packets
                elif event.key == pygame.K_b:
                    env.random_baseline_enabled = not env.random_baseline_enabled
                elif event.key == pygame.K_c:
                    env.manual_camera = False
                    env.camera[0] = clamp(env.robot.x - SIM_WIDTH / 2, 0, WORLD_WIDTH - SIM_WIDTH)
                    env.camera[1] = clamp(env.robot.y - SIM_HEIGHT / 2, 0, WORLD_HEIGHT - SIM_HEIGHT)
                elif event.key == pygame.K_m:
                    env.autopilot_enabled = not env.autopilot_enabled
                elif event.key == pygame.K_f:
                    env.flying_robot = not env.flying_robot
                elif event.key == pygame.K_LEFTBRACKET:
                    env.num_nodes = max(MIN_NUM_NODES, env.num_nodes - NODE_STEP)
                elif event.key == pygame.K_RIGHTBRACKET:
                    env.num_nodes = min(MAX_NUM_NODES, env.num_nodes + NODE_STEP)

        # Enforce scroll bounds
        env.panel_scroll = int(clamp(env.panel_scroll, 0, max(0, env.right_content_height() - RIGHT_VIEW_HEIGHT)))
        env.panel_scroll_x = int(clamp(env.panel_scroll_x, 0, max(0, env.right_content_width() - RIGHT_VIEW_WIDTH)))
        
        # Step the environment and render
        env.update(dt)
        env.draw_dashboard(screen)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
