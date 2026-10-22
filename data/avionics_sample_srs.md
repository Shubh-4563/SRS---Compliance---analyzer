# Software Requirements Specification (SRS)
## Subsystem: UAV Flight Display and Autopilot Telemetry Module (FD-ATM)

### 1. Scope and System Overview
This specification details the software requirements governing the Flight Display and Autopilot Telemetry Module (FD-ATM) deployed in light uncrewed aerial vehicles. The subsystem receives sensory data, computes state estimates, and renders pilot telemetry visuals.

### 2. Functional Requirements
REQ-FDM-001: The flight display unit shall refresh the barometric altitude reading at a frequency of at least 20 Hz with a maximum end-to-end latency of 50 ms.

REQ-FDM-002: The system should update the ground speed display quickly whenever wind gust changes occur.

REQ-FDM-003: The autopilot mode indicator shall display current navigation state, and the unit shall transmit GPS coordinates to the ground control station and emit an audio alarm if link latency exceeds 500 ms.

The primary flight display shall present pitch and roll attitude within +/- 0.5 degrees accuracy.

REQ-FDM-004: If the primary Pitot-static airspeed sensor reports an electrical fault or timeout exceeding 100 ms, the software shall transition into secondary synthetic airspeed estimation mode and flag a Level-2 caution on the annunciator.

### 3. Interface Requirements
REQ-INT-001: The software shall communicate with the Air Data Computer across the ARINC 429 dual-redundant bus at a baud rate of 100 kbps.

REQ-INT-002: The pilot UI shall be modern, intuitive, robust, and easy to use during high-turbulence maneuvers.
