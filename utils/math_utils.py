import math
from typing import Tuple
from core.models import CircleObstacle

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def angle_wrap(theta: float) -> float:
    return (theta + math.pi) % (2 * math.pi) - math.pi

def line_point_distance(a: Tuple[float, float], b: Tuple[float, float], p: Tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab_len2 = abx * abx + aby * aby
    if ab_len2 == 0:
        return math.hypot(px - ax, py - ay)
    t = clamp((apx * abx + apy * aby) / ab_len2, 0.0, 1.0)
    cx, cy = ax + abx * t, ay + aby * t
    return math.hypot(px - cx, py - cy)

def segment_intersects_circle(a: Tuple[float, float], b: Tuple[float, float], c: CircleObstacle) -> bool:
    return line_point_distance(a, b, (c.x, c.y)) <= c.r
