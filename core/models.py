import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class CircleObstacle:
    x: float
    y: float
    r: float

@dataclass
class CommNode:
    x: float
    y: float
    tx_power: float = -28.0
    battery: float = 1.0
    failed: bool = False

@dataclass
class PropagatedPacket:
    origin_id: int
    node_id: int
    hop_count: int
    source_rssi: float
    freshness: float
    delivered_time: float
    path: List[int] = field(default_factory=list)

@dataclass
class RobotObservation:
    node_id: int
    hop_count: int
    source_rssi: float
    freshness: float
    robot_link_rssi: float
    delivered: bool

@dataclass
class Target:
    x: float
    y: float
    found: bool = False
    distress_radius: float = 24.0
    moving: bool = False
    waypoint: Optional[Tuple[float, float]] = None
    repath_at: float = 0.0

@dataclass
class Robot:
    x: float
    y: float
    heading: float = 0.0
    speed: float = 0.0
    trail: List[Tuple[float, float]] = field(default_factory=list)
    last_observations: List[RobotObservation] = field(default_factory=list)
    found_time: Optional[float] = None
    found_target: bool = False
    waypoint: Optional[Tuple[float, float]] = None
    repath_at: float = 0.0

    def forward(self, dt: float):
        self.x += math.cos(self.heading) * self.speed * dt
        self.y += math.sin(self.heading) * self.speed * dt
