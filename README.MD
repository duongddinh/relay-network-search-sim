
# Autonomous Forest Search & Rescue Simulation

A 2D Pygame simulation demonstrating an autonomous robot locating a distress beacon in a dense, obstacle-filled forest environment. The robot relies entirely on a degraded wireless mesh network to triangulate the victim's position, competing against a blind "random search" baseline robot.


## The Future: Bio-Digital Mesh Networks

While deploying physical nodes across thousands of acres of wilderness is a logistical challenge, the ultimate future of forest search and rescue may lie in infrastructure that already exists: the forest itself.

Research into the "Wood Wide Web" has proven that trees are not isolated entities. They are interconnected via vast, subterranean mycorrhizal fungal networks. Trees actively use these biochemical pathways to communicate distress signals—such as drought, disease, or insect attacks—transferring information and resources across miles of forest floor.

The next frontier of this search and rescue algorithm is to bridge the gap between digital packet routing and biological frequencies. Instead of dropping artificial relay nodes, future iterations of this concept could utilize bio-acoustic or bio-chemical sensors capable of tapping directly into this fungal frequency.

By translating biological stress signals or vibrations into digital data, we could turn the entire root system of a forest into a massive, decentralized, zero-latency mesh network. A lost hiker's physical impact or distress could trigger localized biological responses, allowing autonomous drones to triangulate their position purely by reading the data traffic of the trees themselves.

## Features

* **Wireless Mesh Network Simulation:** Models realistic signal degradation (RSSI), packet loss, network hops, and line-of-sight obstruction caused by trees.
* **Autonomous State Machine:** The guided robot dynamically switches between search patterns (Sweep, Homing, Local Probe, and Escape/Anti-stall) based on real-time sensor data.
* **Live Analytics Dashboard:** A scrollable UI panel displaying real-time metrics, network health, packet decodes, and a direct performance comparison against a random-search baseline.
* **Dynamic Environment:** Toggle a moving vs. static victim, switch between flying (drone) and ground (rover) robots, and dynamically adjust the density of the relay nodes.
* **Signal Heatmap:** A visual representation of the network's current "belief" of the victim's location based on active distress packets.

## Project Architecture

The codebase follows a clean, modular Model-View-Controller (MVC) style structure:

```
forest_search/
├── main.py                  # Application entry point & main loop
├── config/
│   └── settings.py          # Global constants, colors, and simulation parameters
├── core/
│   ├── models.py            # Dataclasses for Robot, Target, Nodes, and Packets
│   ├── environment.py       # The simulation engine, physics, and bot algorithms
│   └── renderer.py          # Pygame drawing logic and UI dashboard layout
└── utils/
    ├── math_utils.py        # Geometry, distances, and line-of-sight calculations
    └── generators.py        # World generation (trees, nodes, victim placement)

```

## Installation & Setup

1. **Ensure Python 3.x is installed.**
2. **Install the required dependencies:**
This project relies solely on Pygame.
```bash
pip install pygame

```


3. **Run the simulation:**
```bash
python main.py

```


## Controls

The simulation is highly interactive. Use the following keyboard controls to manipulate the environment in real-time:

* **Arrow Keys:** Pan the camera around the map.
* **Mouse Wheel:** Scroll the right-hand stats panel vertically.
* **Shift + Mouse Wheel:** Scroll the right-hand stats panel horizontally.
* **R:** Reset the world with the current settings (generates a new forest).
* **M:** Toggle Guided Autopilot (ON/OFF).
* **W/A/S/D:** Manually drive the guided robot (only works when Autopilot is OFF).
* **C:** Recenter the camera on the Guided Robot.
* **B:** Toggle the Random Baseline Robot (ON/OFF).
* **T:** Toggle Victim movement (Moving vs. Static).
* **F:** Toggle Robot Type (Flying Drone vs. Ground Rover). *Ground rovers must route around trees.*
* **G:** Toggle Signal Heatmap overlay.
* **P:** Toggle Packet Flow visibility (Orange lines connecting relay nodes).
* **V:** Always show the victim's exact location (cheat mode).
* **"[" or "]":** Decrease / Increase the number of relay nodes (Press **R** to apply).

## How the Algorithm Works

The simulation models a scenario where a victim's distress beacon has a very short range. To find them, the forest is scattered with communication nodes (purple dots) that act as a mesh network.

When a node hears the victim, it forwards the packet. Trees block signals, and distance degrades the RSSI (Received Signal Strength Indicator). The Guided Robot uses this imperfect data to find the target.

### 1. The Search States

The Guided Robot operates using a 4-tier state machine:

* **Sweep Mode:** If no signal is detected, the robot executes a wide lawnmower-style grid search across the map.
* **Homing Mode:** When the robot decodes a distress packet, it calculates an Estimate (**EST**) of the victim's location and drives directly toward that Waypoint (**WP**).
* **Local Probe Mode:** Once it reaches the estimated area, it initiates a tight, randomized search radius to pinpoint the exact location.
* **Anti-Stall / Escape:** If the robot gets stuck on trees for more than 4 seconds, it abandons its current path and picks a distant random waypoint to break free.

### 2. Triangulation Strategy

When the robot receives relayed packets, it filters out dead/dropped packets and looks at the top 5 strongest signals. It calculates a weighted average based on three factors:

1. **Link RSSI:** How strong is the signal from the relay node?
2. **Freshness:** How old is the packet?
3. **Hops:** How many times was the packet relayed? (Fewer hops mean the transmitting node is closer to the actual victim).

### 3. The Signal Heatmap (G Key)

Pressing `G` reveals the "Heatmap". This is *not* the literal radio wave of the victim. Instead, it is the **network's belief** of the victim's location. The heat blooms around relay nodes that currently hold fresh distress packets. It uses a simplified distance formula for performance, which is why it appears as coarse, 60x60 pixel blocks.

## The Random Baseline

To prove the efficiency of the routing algorithm, a dark gray "Random Baseline" robot explores the map simultaneously. It operates purely on luck, picking random Waypoints (**RWP**) and driving to them. The dashboard tracks which robot finds the victim first to demonstrate the value of the mesh-network triangulation approach.


Here is a detailed breakdown of the "Mesh-Assisted Triangulation" algorithm that you can drop directly into your `README.md`. It explains exactly how the math, packet routing, and state machine functions work together in the code.

---

## Detailed Algorithm: Mesh-Assisted Triangulation

The core intelligence of this simulation relies on a decentralized infrastructure. The Guided Robot acts as a dynamic client navigating a highly variable mesh topology to isolate the source of a distress beacon. The algorithm is broken down into five distinct phases: Generation, Propagation, Decoding, Triangulation, and Movement.

### Phase 1: Packet Generation (The Victim)

The victim acts as the origin point, broadcasting a distress packet every 1.6 seconds via the `emit_distress_packet()` function.

* **Seeding:** The victim finds the 3 nearest communication nodes.
* **Base RSSI Calculation:** It calculates the initial Received Signal Strength Indicator (RSSI) using the standard free-space path loss formula: `rssi = -35 - 20 * log10(distance)`.
* **Blocker Penalty:** For every tree intersecting the direct line-of-sight between the victim and the seed node, a flat signal penalty (1.0 to 2.8 dBm) is deducted, simulating physical obstruction.

### Phase 2: Mesh Propagation (The Network)

Once the seed nodes receive the packet, they flood the network using `propagate_packet()`, passing the data from node to node.

* **Hop Limits:** Packets are strictly limited to a maximum of 7 hops (`MAX_PACKET_HOPS`) and a time-to-live of 3.4 seconds (`PACKET_TTL`) to prevent infinite broadcast loops.
* **Edge Success Probability:** The `edge_success_probability()` function determines if a packet successfully jumps from Node A to Node B. It starts at a 94% success rate, subtracting 18% based on the distance between the nodes, and 8% for every tree blocking the path.
* **Signal Degradation:** As the packet hops, its `forwarded_rssi` drops based on distance and obstacles, and its `freshness` value decays over time.

### Phase 3: Observation & Decoding (The Robot's Receiver)

The Guided Robot does not hear the victim directly; it listens to the network nodes using `generate_robot_observations()`.

* **Sensor Range:** The robot only attempts to read packets from nodes within its 420-pixel sensor radius.
* **Decoding Probability:** Just like the network edges, the link between the relay node and the robot is imperfect. The robot calculates a `p_success` probability based on distance and line-of-sight. If a random check passes this probability, the packet is flagged as `delivered`.
* **Link RSSI:** The robot calculates the `robot_link_rssi`, the signal strength of the transmission from the relay node *to* the robot.

### Phase 4: The Triangulation Math (The Brains)

When the robot successfully decodes a batch of packets, it processes them in `update_estimate()` to guess the victim's location.

1. **Filtering & Scoring:** The robot filters out dropped packets and scores the successful ones using a custom heuristic formula:
`Score = robot_link_rssi + (12.0 * freshness) - (2.2 * hop_count)`
*This severely penalizes packets that took too many hops, prioritizing nodes physically closest to the origin beacon.*
2. **Top 5 Selection:** It isolates the 5 packets with the highest scores.
3. **Weighted Average:** The algorithm calculates a weighted geographic center based on the physical `(x, y)` coordinates of the nodes that transmitted those top 5 packets.
* **Weight Formula:** `w = max(0.5, (robot_link_rssi + 130.0) * 0.06 + 2.2 * freshness - 0.35 * hop_count)`
* **Center Calculation:** `Estimated X = sum(node.x * w) / sum(w)`


4. This new `(x, y)` coordinate becomes the robot's **Best Estimate (EST)**.

### Phase 5: The Autonomous State Machine & Movement

With an estimated coordinate calculated, the robot uses `update_autopilot()` to make decisions, and `steer_bot_toward()` to physically drive.

**The State Machine:**

* **Sweep Mode:** If there are no valid estimates (no signal), the robot follows a pre-calculated array of waypoints (`build_sweep_points()`), driving in a lawnmower pattern to maximize map coverage.
* **Homing Mode:** The moment an Estimate is calculated, the robot overrides the sweep pattern and sets the Estimate as its active Waypoint.
* **Local Probe Mode:** Once the robot reaches the Estimate (or if it gets stuck trying to get there for more than 7 seconds), it assumes the victim is close but obscured. It generates short, random radial waypoints around the estimate to thoroughly search the immediate area.
* **Anti-Stall (Random Explore):** The robot tracks its recent positional history. If it detects that it hasn't moved more than 85 pixels in 4 seconds (stuck on a cluster of trees), it flushes its current waypoint and targets a distant random coordinate to break free before resuming the hunt.

**Kinematics (`steer_bot_toward`):**

* **Heading Calculation:** The robot calculates the angle to its active waypoint using `atan2(dy, dx)`.
* **Turning:** It finds the shortest turn direction using `angle_wrap()` and applies a proportional turn gain to swing its nose toward the target.
* **Speed Regulation:** To prevent overshooting, the robot calculates a `speed_factor`. If it has to make a sharp turn, it automatically throttles down its speed, accelerating back to maximum velocity only when pointing directly at the target.