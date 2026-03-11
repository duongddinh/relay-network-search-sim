import random
from typing import List
from config.settings import WORLD_WIDTH, WORLD_HEIGHT, TREE_RADIUS_RANGE
from core.models import CircleObstacle, CommNode, Target
from utils.math_utils import dist

def generate_obstacles(num_trees: int) -> List[CircleObstacle]:
    obstacles: List[CircleObstacle] = []
    margin = 80
    attempts = 0
    while len(obstacles) < num_trees and attempts < num_trees * 50:
        attempts += 1
        r = random.randint(*TREE_RADIUS_RANGE)
        x = random.randint(margin, WORLD_WIDTH - margin)
        y = random.randint(margin, WORLD_HEIGHT - margin)
        ok = True
        for ob in obstacles:
            if dist((x, y), (ob.x, ob.y)) < r + ob.r + 12:
                ok = False
                break
        if dist((x, y), (120, 120)) < 140:
            ok = False
        if ok:
            obstacles.append(CircleObstacle(x, y, r))
    return obstacles

def generate_nodes(obstacles: List[CircleObstacle], num_nodes: int) -> List[CommNode]:
    nodes: List[CommNode] = []
    attempts = 0
    while len(nodes) < num_nodes and attempts < num_nodes * 220:
        attempts += 1
        x = random.randint(90, WORLD_WIDTH - 90)
        y = random.randint(90, WORLD_HEIGHT - 90)
        ok = True
        for ob in obstacles:
            if dist((x, y), (ob.x, ob.y)) < ob.r + 35:
                ok = False
                break
        for n in nodes:
            if dist((x, y), (n.x, n.y)) < 120:
                ok = False
                break
        if ok:
            nodes.append(CommNode(x, y, failed=random.random() < 0.05))
    return nodes

def generate_target(obstacles: List[CircleObstacle]) -> Target:
    while True:
        x = random.randint(220, WORLD_WIDTH - 140)
        y = random.randint(220, WORLD_HEIGHT - 140)
        if all(dist((x, y), (ob.x, ob.y)) > ob.r + 60 for ob in obstacles):
            return Target(x, y)
