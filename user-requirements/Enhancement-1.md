# Enhancement Requirements \& Detailed Design

**(For 5G Backhaul Time Series Traffic Statistics Generator Utility)**

This section provides detailed requirements and design notes for three targeted enhancements: Traffic Mix Reflections, Loss/Latency/Jitter Patterns, and Time-of-Day Variation. These details are designed to extend your existing base model for greater realism and utility in telecom network simulation and analysis.

## 1. Traffic Mix Reflections

### Requirements

- Enable parameterization and modeling of realistic telecom traffic mixes as observed in field deployments.
- Support multiple concurrent traffic classes: eMBB (mobile broadband), URLLC (critical low-latency), IoT/mMTC (machine-type), control/VoNR, etc.
- Allow per-class traffic characteristics:
    - Percentage of total traffic per class.
    - Distinct packet size distributions per class (e.g., large/small percentage splits).
    - Protocol and QoS/5QI identifier assignment.
- Permit user configuration of per-class traffic shares and distributions.


### Design

#### Configuration

```yaml
traffic_classes:
  - name: eMBB
    traffic_share: 0.85     # 85% of bytes
    packet_size_distribution: [ {"size": 1400, "share": 0.80}, {"size": 200, "share": 0.20} ]
    flow_count: 25
    qos: 9
  - name: URLLC
    traffic_share: 0.10
    packet_size_distribution: [ {"size": 88, "share": 1.0} ]
    qos: 3
  - name: VoNR
    traffic_share: 0.05
    packet_size_distribution: [ {"size": 60, "share": 1.0} ]
    qos: 1
```


#### Engine Changes

- Each interval:
    - Determine per-class share of total bytes/packets.
    - For each class, sample packet sizes from configured distribution.
    - Assign protocol, flow, and QoS identifiers as configured.
- Aggregate outputs across classes for total statistics, and retain per-class breakdowns.


#### Output Statistics

| Time | Total UL Bytes | eMBB Bytes | URLLC Bytes | eMBB PacketSize | URLLC PacketSize | ... |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| ... | ... | ... | ... | ... | ... | ... |

## 2. Loss, Latency, and Jitter Patterns

### Requirements

- Model and synthesize effects of packet loss, variable latency, and jitter, as observed in operational 5G backhaul scenarios.
- Allow configuration of:
    - Average packet loss rates (overall and per-class).
    - Latency profiles: base latency, variation, distribution type (normal, uniform, etc.).
    - Jitter profiles: standard deviation, distribution, burstiness factor.
    - Correlation parameters (e.g., increased loss/jitter at peak load intervals).
- Optionally correlate impairments to time-of-day or traffic bursts.


### Design

#### Loss Model

- User-configurable as either a fixed percentage or a stochastic model (e.g., Gilbert-Elliott for bursty loss).
- Loss applied during statistics computation; record 'lost packets' counter per interval/class.


#### Latency \& Jitter Model

- For each interval or (optionally) each packet sample:
    - Draw latency from configured distribution (mean, stddev, min/max).
    - Option to add occasional outliers representing bursty congestion.
- Compute jitter as the variation of latency relative to the previous interval/packet.


#### Configuration Example

```yaml
impairments:
  - class: eMBB
    loss_rate: 0.01          # 1%
    latency_ms: { mean: 12, std: 1.2 }
    jitter_ms: { std: 0.8 }
  - class: URLLC
    loss_rate: 0.0001        # 0.01%
    latency_ms: { mean: 3, std: 0.3 }
    jitter_ms: { std: 0.1 }
```


#### Output Statistics

- For each interval/class:
    - Packets Lost, Loss Rate
    - Avg/Min/Max Latency
    - Avg/Max Jitter

| Time | eMBB Latency(ms) | eMBB Jitter(ms) | eMBB Loss(%) | URLLC Latency(ms) | ... |
| :-- | :-- | :-- | :-- | :-- | :-- |
| ... | ... | ... | ... | ... | ... |

## 3. Time-of-Day Variation

### Requirements

- Allow daily (and optionally weekly) traffic fluctuation modeling for each class and/or aggregate traffic.
- Support both pre-configured patterns (industry typical: morning/evening peaks, troughs at night[^1][^2]) and user-uploaded custom curves.
- Apply these variation patterns to modulate per-interval traffic volume, flow counts, and even loss/latency profiles if desired.


### Design

#### Configuration

- Provide default daily traffic patterns:
    - Example: Office area — peaks at noon (12PM) and evening (8PM), trough at 4AM.
    - Alternate patterns for residential, transport, entertainment functional areas.
- Accept a normalized 24-hour curve per class (resolution: 5-30min; interpolated per interval).
- Optionally link impairment patterns (loss/jitter/latency) to time-of-day (i.e., higher impairment during peaks).

```yaml
time_of_day_profile:
  eMBB: [0.2, 0.4, 0.6, 1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.2, ...]   # 24 or 48 points
  URLLC: [0.7, 0.6, ...]
  # Or accept a .csv with normalized load multipliers for each time segment
```


#### Simulation Flow

- At each interval:
    - Multiply baseline per-class traffic volume by the relevant time-of-day multiplier.
    - Optionally, adjust impairment rates proportionally during traffic peaks to reflect observed field congestion effects.


#### Output

- Add annotations/tags per interval indicating time-of-day, traffic pattern phase (peak/normal/trough).
- Statistics show effect of time-varying load (clearly visible in graphical or CSV analysis tables).

| Time | eMBB Multiplier | URLLC Multiplier | Total Traffic | eMBB Latency | eMBB Loss | ... |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 08:00 (Morning) | 0.6 | 1.2 | ... | ... | ... | ... |
| 12:00 (Noon Peak) | 1.0 | 1.1 | ... | ... | ... | ... |
| 20:00 (Evening) | 1.0 | 0.9 | ... | ... | ... | ... |
| 04:00 (Night) | 0.2 | 0.8 | ... | ... | ... | ... |

## Implementation Notes

- All enhancements must be backward-compatible with base model and their configuration should be optional/overridable.
- Data structure should enable independent and combined reporting for per-class and aggregate counters.
- Robust logging and validation of configuration/inputs should be enforced.
- Maintain full scriptability and integration/automation capabilities in line with user’s advanced network engineering workflows and interest in traffic analysis.
