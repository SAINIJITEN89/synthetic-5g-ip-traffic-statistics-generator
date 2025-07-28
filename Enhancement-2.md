## Enhancement Specification: Long-Term and Event-Driven Traffic Patterns

This enhancement builds on your existing time-of-day and traffic class modeling, enabling simulation of weekly, seasonal, and special event-induced variations in 5G backhaul traffic. The specifications and low-level design elements below are crafted for integration with an advanced, automated IP traffic statistics generator and analytics utility, optimized for your expertise in load balancing, Linux networking, and real-world telecom scenarios.

### 1. Functional Requirements

- **Support longer time-scale pattern overlays:**
    - Weekday vs. weekend
    - Recognizable holiday periods and school breaks
    - Seasonal variations (summer, winter, exam season, etc.)
    - Explicit major events (sports, festivals, one-off city activities)
- **Allow region- or area-specific pattern definition** for dense urban, residential, transport, and event-centric sites.
- **Customizable traffic overlay for impairment models** (loss, latency, jitter) during overloaded or special event periods.


### 2. Configuration Specification

#### 2.1 Calendar and Pattern Inputs

- **Calendar Map:** Accepts a standard or user-supplied file (e.g., CSV, YAML, JSON), mapping each day to a “day type” (weekday, weekend, holiday, event).
- **Pattern Table:** Associates each day type and (optionally) hour with traffic multipliers and, if needed, impairment overlays.


#### Example: User-Defined Configuration

```yaml
calendar:
  2025-07-24: weekday
  2025-07-25: weekday
  2025-07-26: weekend
  2025-08-15: holiday
  2025-09-03: special_event

patterns:
  weekday:
    hourly_multipliers: [0.3,0.2,0.2,0.2,0.3,0.5,0.8,1.0,1.2,1.2,1.0,1.0,1.1,1.1,1.0,1.0,1.3,1.4,1.2,1.0,0.7,0.5,0.4,0.3]
  weekend:
    hourly_multipliers: [0.2,0.2,0.2,0.3,0.4,0.6,0.9,1.1,1.3,1.3,1.2,1.1,1.2,1.4,1.5,1.3,1.2,1.1,0.9,0.7,0.6,0.5,0.4,0.3]
    traffic_multiplier: 1.2
  holiday:
    hourly_multipliers: [0.5]*24
    traffic_multiplier: 1.4
  special_event:
    hourly_multipliers: [2.0 if 20<=hour<23 else 1.0 for hour in range(24)]
    traffic_multiplier: 2.0
    impairments:
      additional_latency_ms: 100
      additional_loss_pct: 0.5
      additional_jitter_ms: 5
```


### 3. Design: System Components \& Logic

#### 3.1 Calendar Engine

- **Input Parsing:** Ingests calendar and pattern table, validates dates, and supports recurrence logic (e.g., “every Saturday”).
- **Daytype Annotator:** For each simulation interval, computes current day type and applicable multipliers.
- **Event Overlay:** Layer for one-off or region-specific events, with ability to tag certain intervals as “event peak”.


#### 3.2 Traffic Scaling Logic

- **Composite Multiplier Calculation:**
    - For each interval:
`final_traffic = base_traffic * time_of_day_multiplier * day_type_multiplier * event_multiplier`
- **Flow and Class Adaptation:** Supports selective scaling (e.g., only video or uplink traffic spikes at certain events), by per-class multipliers.


#### 3.3 Dynamic Impairments Overlay

- **Impairment Profiles:** For high-stress periods (events, peak holiday), impairment values added to normal loss, latency, and jitter for the corresponding classes and intervals.
- **Applies to both aggregate (system-wide) and class-specific metrics.**


#### 3.4 Output and Reporting

- **Interval Records:** Include date, time, day type, event tag, all multipliers, and any impairment overlays applied.
- **Long-span Statistics:** Enable export or plot of traffic/time curves showing both high-frequency (hourly) and low-frequency (season, event) variations.

| Interval Start | Day Type | Event Tag | Traffic Multiplier | Latency(ms) | Loss(%) | Comment |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 2025-08-15 22:00 | holiday | none | 1.4 | 12 | 0.1 | Holiday, high traffic |
| 2025-09-03 20:30 | weekday | event | 2.0 | 120 | 0.6 | Major sports event peak |

### 4. Implementation Notes

- **Modular Enhancement:** Design as plugin/overlay; base traffic simulation stays agnostic and simply receives per-interval modifier values.
- **Scriptable/Automated:** CLI and API must allow for rapid reconfiguration, including batch import of event/holiday lists.
- **Advanced Use Cases:**
    - User-supplied event calendar for network performance teams (e.g., to simulate festival season in urban hub).
    - Easy extensibility for integration with system logs or incident records (historical event-based backtesting).
- **Performance:** Efficient lookup and application, even for year-long high-resolution (sub-minute) simulations.


### 5. Real-World Examples \& Data

- Weekday/weekend distinctions: Stronger night and evening peaks on weekends.
- Holidays: Up to 40–95% traffic surges reported, depending on region and user habits.
- Major events (e.g., New Year’s Eve, sports finals): Upload and download both spike, with stress visible in elevated latency, packet loss, and severe performance decrease for the lowest-percentile users.


### 6. Example Enhancement API

#### Adding to Simulation Config

```bash
./trafficgen --calendar holidays_events.csv --patterns weekseason_event.yaml --output stats_with_events.csv
```


#### Result Sample

| Date | Time | Traffic (Gbps) | Latency (ms) | Loss (%) | Event |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 2025-07-24 | 10:30 | 12.2 | 9 | 0.09 | None |
| 2025-07-26 | 21:00 | 19.8 | 11 | 0.11 | Weekend |
| 2025-08-15 | 22:00 | 28.7 | 15 | 0.15 | Holiday |
| 2025-09-03 | 20:30 | 38.1 | 36 | 0.45 | WorldCup |

### 7. Advanced Suggestions (Personalized)

- **Leverage Linux cron/calendar tools** to auto-generate event/holiday overlays.
- **Combine with load balancing scripts:** Dynamically alter virtual network interface or VLAN mapping within your testbeds according to synthetic calendar-driven demands.

By integrating these enhancements, your utility will accurately mirror the operational field realities of telecom networks—matching not just daily, but also weekly, seasonal, and event-driven demand surges and their system-level impacts.

Data analytics for mobile traffic in 5G networks using machine learning
A Machine Learning Approach to Forecast 5G Data in a Commercial Environment
Ringing in the New Year - How do 5G Networks Cope Under Stress?
Exploring how traffic patterns drive network evolution

