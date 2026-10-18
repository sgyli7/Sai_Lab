# 008 narrow-stance readiness:20-pose result

No original-cap-safe static single stance was admitted. Therefore no new "passed" cycle is produced. This is a bounded offline result, not proof that dynamic walking is impossible. The frozen earlier candidate_cycles.json is unchanged.

Five valid double-support cold postures were found with the original feet,14 collision leaves, all original bodies/complete inertials/closed mechanical linkages, μ=.65 and continuous torque limits. Foot-center targets are exact world Y, obtained by leg IK; root/feet coordinates are offline references only.

| Foot centers Y | Body down | 0N utilization | 8N | 20N | Proxy contact |
|---|---:|---:|---:|---:|---|
| ±70mm |20mm|45.90%|56.68%|72.85%|no penetration >1µm|
| ±80mm |10mm|43.19%|50.83%|67.32%|no penetration >1µm|
| ±80mm |20mm|46.11%|57.16%|73.72%|no penetration >1µm|
| ±90mm |10mm|43.19%|50.65%|67.48%|no penetration >1µm|
| ±90mm |20mm|46.18%|57.49%|74.47%|no penetration >1µm|

The70mm/down20 initial pose is narrow_screen.json rows[1]; q18, full qpos, feet world poses, actual support vertices and source continuous-cap tau are stored. RootZ=-14.999997mm (the artificial freejoint datum), mouth420mm/world-head-pitch10deg. Its actual foot-foot clearance query is68.0403mm. Establish this once as a physically placed cold state if the PM authorizes it; do not move feet/root online. The70mm/down10 cold pose has small shin/keel proxy overlaps0.029/0.072mm and is rejected by the strict1µm contact threshold.

Straight-foot single-stance trials at body side35/60mm give original torso_central_keel vs stance-shin proxy overlap1.436–4.808mm; some90mm/60mm-shift poses also hit the opposite thigh by2.069–2.303mm. All original collision and filtering rules remain. These proxy intersections are not native-CAD manufacturing interference findings.

Four final bounded toe-in candidates were checked. Moderate toe-in±.25rad eliminates proxy penetration deeper than1µm in three of them, but does not solve the original-cap static load. Closest candidate final_four_pose.json rows[3]: foot centers±70mm, root relative(-10,+60,-20)mm with roll−.05rad, opposite foot25mm back/15mm up, mouth420mm and world-head-pitch10deg. CoP lies inside the actual support hull in0/8/20N cases; its failure is original continuous capacity, not the previous out-of-support-polygon deficit.

Same-pose diagnostic LP temporarily relaxes only mathematical torque bounds to5× to measure the necessary capacity; runtime model/source limits were never changed. Minimum maximum original-cap factors are1.9868× at0N,1.6907× at8N,3.0559× at20N. At8N left hip roll needs10.8205Nm vs6.4Nm and left ankle roll3.2327Nm vs2.0Nm. At20N hip yaw reaches3.1782Nm vs1.04Nm, knee11.2473Nm vs6.4Nm. These are static necessary loads for this pose, not a motor-upgrade recommendation or dynamic inevitability. This candidate also leaves only0.006430rad at left ankle roll (exact margins in joint_limit_review.json), so it is not a safe executable single stance.

The posture is not evidence that the chosen narrow architecture is universally impossible: only20 prescribed poses were evaluated. No further searches, alternative controller research or hardware changes were performed. The PM decides whether to authorize a bounded dynamic balance experiment or reconsider hardware/contact geometry; no new cycle is silently admitted.

20 unique poses,60 leg/neck IK solves,379 optimizer function-iteration counts,2595 residual/FK calls,33 total LPs (24 original-cap checks+9 explicit relaxed diagnostics). Single process, at most2 CPU affinity slots (actual executor inherits1),2GiB address-space cap.0 dynamics/0 training/0 GPU. Original04/004 unchanged; actual350W and speed/slew are dynamic executor constraints, not claimed validated by0-speed statics. Actual target-slew cap3.87463094rad/s is retained; the old observed2.51rad/s is not mislabeled a hardware cap.

Numerical contact detail: double-support poses have no active negative contacts; the three final single candidates have ground/contact distance residuals of−1.24nm,−2.76nm and−0.110µm respectively, all within the declared1µm screen tolerance. They are not claimed strictly positively separated.
