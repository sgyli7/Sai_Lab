# 12 held-FF age audit — four actual states, zero integration

Exactly four initial LP calls plus one Owner-authorized replacement of the non-drive6.433 sample. The rejected original fourth remains saved. No model, source, gain, phase, contact, cap or frozen run was changed. Every LP uses original12 actual-native modes and the same current C2 planned q/qd/qdd and actual held front channel. Three exact drive-tick held-command reproductions match actual total torque within1.7e−11 Nm.

At6.433 the last real drive is6.430. The corrected comparison holds the recorded target and qd, without reissuing/advancing the segment. Actual6.430 PD+FF reconstructs the held total torque exactly. It separately reports current-state PD feedback with those held targets, then newFF with the same current PD. Neither is falsely called a command that actually ran.

|Actual t|LP ms|FF age ms|Goal Y / roll / yaw|Old ACTUAL response|Fresh response|
|---|---:|---:|---|---|---|
|6.265|10.66|5.0|+0.2326 / +1.5052 / -2.9783|+1.0541 / +4.8447 / -2.3980|+0.2491 / +1.5221 / -5.4291|
|6.290|8.74|10.0|-0.1520 / -0.3249 / -3.8449|-1.1419 / -2.5487 / -6.6143|-0.2852 / -0.3253 / -5.8941|
|6.355|7.73|15.0|-0.2073 / +0.2481 / -1.9737|-1.3623 / -5.7293 / -4.8000|-0.2073 / -2.4844 / -3.2300|
|6.433|11.77|13.0|-0.3922 / +2.7896 / +4.4787|-0.7673 / -0.3542 / -3.3648|+0.0997 / +2.7896 / +0.0001|

Linear Y acceleration is m/s²; roll/yaw are native freejoint local angular acceleration rad/s², compared against the same correctly framed root task. They are not COM acceleration. The original saved centroidal/wrench record is included verbatim in final_comparison; no new COM estimate is presented as measurement.

Fresh FF strongly changes root/foot response and removes the first LAR clipping at6.265 (2.000→1.994 Nm), while increasing leftHR4.098→6.012 Nm. First-state yaw worsens−2.398→−5.429 against goal−2.978. At6.355 right knee remains at−6.4 Nm. At6.433 freshY+.0997 is still the wrong sign against goal−.3922; roll/pitch improve but yaw remains far below its+4.4787 target. Thus reduced error in selected axes is not general balance success.

Recommendation: heldFF staleness is large enough to justify one independently declared **non-real-time simulation** FF-refresh comparison, subject to Owner choice. It must retain the50Hz plan/segments, original200Hz PD and all physical/source limits, and explicitly versionFF200Hz. The current solver takes7.7–11.8ms, exceeding5ms: it cannot be claimed board-real-time. Native source legality or an optimal scratch LP does not prove real air, stable landing, support exchange or transport. Existing yaw tradeoff, mixed soft objectives, contact/mode dependence, and foot-clearance failure remain unresolved. No implementation or actual run is authorized by this audit.
