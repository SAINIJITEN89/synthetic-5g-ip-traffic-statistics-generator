# Low-Level Requirement \& Design Document

**5G Backhaul Traffic Time Series Statistics Generator (Counter Utility)**

## 1. Purpose

Develop a configurable utility that simulates and outputs realistic, time series IP traffic statistics for 5G backhaul (gNB-to-Core) scenarios. The utility calculates and records traffic counters based on user-defined profiles, without generating actual network packets.

## 2. Functional Requirements

### 2.1. Core Features

- Generate per-interval (e.g., per ms/sec/min) statistics for backhaul traffic.
- Accept input for realistic 5G service types: eMBB, URLLC, mMTC, and custom profiles.
- Support detailed parameterization: users, packet size, packet rate, UL/DL ratio, flows, bursts, and 5QI/QoS.
- Handle multiple concurrent simulation sessions (multi-profile, multi-QoS).
- Produce output in standard file formats (CSV, JSON).
- Optionally, expose a REST/gRPC API for integration or remote control.


### 2.2. User Inputs

| Parameter | Type/Unit | Description |
| :-- | :-- | :-- |
| Simulation duration | Int (sec/min) | Total simulation runtime |
| Output granularity | Int (ms/sec/min) | Interval for statistics aggregation |
| Traffic profile(s) | Enum/List | Pre-defined/custom profile templates (eMBB, URLLC, etc.) |
| Num. of users (UE) | Int | Simulated number of active users per profile |
| Packet size distribution | List/Model | e.g., percentage split of small/large packets |
| Packet rate/flow | Int/List | Per-user or per-flow packet rate specification |
| UL/DL ratio | Float | Uplink-to-downlink bytes or packets ratio |
| Burst characteristics | Struct/List | Burst duration, duty cycle, on-off pattern (if bursty) |
| Flow/session model | Struct | Concurrent flows, session inter-arrival, concurrency |
| 5QI / QoS config | Int/List | QoS identifier(s) per profile or flow |
| Random seed | Int | For deterministic reproducibility of randomized scenarios |

### 2.3. Output Statistics

At each reporting interval, generate:

- Total packets (UL/DL)
- Total bytes (UL/DL)
- Number of active flows, sessions, users
- Average, min, max packet size
- Packet size histogram/bins (optional)
- Throughput (UL/DL) in Mbps
- DL/UL byte and packet ratio
- Per-QoS (5QI) breakdown of packets/bytes
- Simulated latency, jitter (if modelled)
- Timestamps


## 3. Non-Functional Requirements

- High efficiency for large scale simulation (e.g., 100k users, >10Gbps aggregate rates).
- Deterministic output support (via static random seed).
- Modular, extensible codebase (configurable via CLI, file, or API).
- Minimal dependencies; OS-agnostic.


## 4. Architecture Overview

### 4.1. Main Components

- **Configuration Parser**
    - Accepts/validates parameters (CLI, file, API input)
- **Traffic Model Engine**
    - Implements 5G traffic profile logic and stochastic/deterministic models
    - Supports multiple active profiles in parallel
- **Simulation Engine**
    - Advances simulation clock, manages event queue/state, computes counters per interval
- **Statistics Collector**
    - Aggregates interval counters, maintains histories, formats output lines
- **Output Formatter/Exporter**
    - Writes final stats (CSV/JSON), optionally exposes API endpoints


### 4.2. Component Interactions

```mermaid
graph LR
  ConfigParser --> TrafficModelEngine
  TrafficModelEngine --> SimulationEngine
  SimulationEngine --> StatisticsCollector
  StatisticsCollector --> OutputFormatter
```


## 5. Detailed Module Design

### 5.1. Configuration Parser

- Accepts YAML/JSON config or CLI arguments
- Validates input types, ranges, and constraints
- Loads built-in and user-defined profiles/templates


### 5.2. Traffic Model Engine

- Maps scenario/profile to traffic parameters (following industry/3GPP profiles)
- Implements:
    - Packet size splits (e.g., 90% large, 10% small)
    - Per-flow/session modelling
    - Stochastic models: Poisson arrivals; burst behaviour
    - UL/DL split and QoS/5QI tagging


### 5.3. Simulation Engine

- Advances simulation in discrete steps (per output granularity)
- Tracks all active flows/sessions per profile
- For each interval:
    - Samples packet events (size, direction, flow, QI)
    - Updates user/flow/session states (appearing/disappearing)
    - Applies burst/on-off patterns if configured


### 5.4. Statistics Collector

- For each interval, records:
    - Packets/bytes per direction, per QoS, per user
    - Aggregate and per-flow metrics
    - Packet size, throughput stats
    - Optional: per-bucket histograms
- Maintains cumulative counters as needed


### 5.5. Output Formatter/Exporter

- Formats and writes data per user configuration (CSV or JSON)
- Each record: timestamp, metrics, profile IDs, interval annotations
- Optional: API output hooks for streaming or REST/gRPC serving


## 6. Example API/Config

### CLI Invocation

```
5gsimstat --duration=60 --interval=1 --profile=embb --users=500 --ul_dl=0.1 --packet_size_split='90:10:large:small' --output=out.csv
```


### JSON Config Example

```json
{
    "duration_sec": 120,
    "interval_ms": 100,
    "profiles": [
        {
            "name": "eMBB",
            "num_users": 1000,
            "packet_size_bytes": [1200, 200],
            "packet_size_pct": [88, 12],
            "ul_dl_ratio": 0.12,
            "flows_per_user": 2,
            "5qi": 9
        }
    ],
    "output_file": "stats.csv"
}
```


## 7. Test Case Examples

| Scenario | Users | Flows/User | Packet Split (L/S) | UL/DL Ratio | Duration | Expected Outputs |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| Dense eMBB | 500 | 3 | 90%/10% | 0.15 | 5 min | DL-dominant, high throughput |
| Mixed (eMBB+URLLC) | 200/50 | 2/1 | 80%/20%, 60%/40% | 0.1/1.0 | 2 min | URLLC: many small, equal UL/DL |
| Bursty IoT (mMTC) | 2,000 | 1 | 100% small | 0.85 | 10 min | Mostly UL, sparse intervals |

## 8. Reference Traffic Profile Table

| Profile | Packet Size (bytes) | Packet Size Split | UL/DL Ratio | Traffic Mix |
| :-- | :-- | :-- | :-- | :-- |
| eMBB | 645–1,400 | 90–95% large, 5–10% small | 1:9 | Video, app, bulk |
| URLLC | 40–250 | 100% small | 1:1 (variable) | Low-latency |
| mMTC | <100 | 100% small | UL-heavy | IoT, sensors |

## 9. Implementation Notes

- Use efficient timer/event loop (asyncio/threaded for scaling).
- Modular design: allow plugin profiles for future extensibility.
- Include random seed management for reproducibility.
- Add logging, parameter validation, and error handling.


## 10. Optional Enhancements

- Live preview of time series graphs (via REST UI or terminal).
- Real-time anomaly alerts if traffic deviates from expected.
- Integration hooks for lab/testbed orchestration tools.

**This specification provides full technical guidance for a coding agent to implement the counter-only 5G backhaul traffic statistics utility, aligned with real-world and standards-driven scenarios.**

