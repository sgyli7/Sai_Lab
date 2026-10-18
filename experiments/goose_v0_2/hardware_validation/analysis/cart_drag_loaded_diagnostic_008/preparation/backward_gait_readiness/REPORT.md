# 008 backward gait readiness — offline bounded result

No alternating loaded gait was admitted. No continuous0.5m retreat was executed by this review. A complete experimental40mm-cycle joint-reference interface and a geometry-clean single-stance reference are delivered to the sole dynamic Teacher; these are not passed walking results.

The best single-stance reference has no active14-leaf collision deeper than0.5mm, leg/neck IK errors below1e-8m, mouth420mm/world-head-pitch10deg, root(-20,+40,-20)mm absolute (rootZ=-19.999997mm) and jaw0.196825rad. Full q18/qpos, actual feet, equality residual and source-based forces are recorded. Internal yaw rotor positions were explicitly synchronized to their original1:1 belt equalities. Maximum equality residual2.78e-17.

Its COM is(-9.860,+33.299,+293.664)mm. At +8N mouth reaction, required static CoP lies61.622mm beyond a sole-support hull halfspace. The single-support LP is infeasible at0,8,20N. This is a static support equilibrium deficit even before torque-cap relaxation, not evidence that stronger motors alone solve it. A simplified COM/ZMP necessary lateral acceleration toward the opposite side is at least1.488m/s² using the global inner-edge bound; full multibody angular momentum and actual dynamic actuator feasibility remain untested. This is not a prescribed controller or passed dynamic acceleration.

An adjacent double-support pose with the same root, left foot and orientation, right foot lowered to the actual plane, passes the0.5mm proxy-overlap screening threshold and is statically feasible at0/8/20N using43.186/49.644/79.925% maximum original continuous caps. Mouth TCP stays420mm. No external root force or mouth vertical-support credit is used.

## Teacher interface

`candidate_cycles.json`, cycles[1]: period2s, nominal40mm backward per cycle (.02m/s), original18-axis q targets; frames contain u/stance/phase/root diagnostic/feet diagnostics/static tau when feasible. Each foot has a40mm backward landing and15mm swing clearance. Targets repeat relative to the next cycle's40mm translation, so13cycles would request.52m; this is a requested trajectory, not measured travel. Initial toe-in±.25rad/crouch must be established physically, never by online feet/root repositioning.

This cyclic candidate is NOT admitted: right lift and left transfer/swing have original torso_upper_envelope/thigh proxy intersections1.415–1.464mm; one geometry-clean single-support frame still violates static support by66.953mm. Double-support cycle frames are approximately47.8–51.3%cap. `lean_candidate_cycle.json` is a finite roll±.07rad variant and also not admitted. No collision leaves were deleted or filtered. These are proxy conflicts requiring native/trajectory interpretation, not established manufacturing interferences.

`best_single_stance_reference.json` plus `adjacent_double_support_reference.json` provide the requested complete best single/adjacent-double q18/feet/source static limits. Teacher may use them only as separately authorized bounded dynamic exploration, recording actual support transfer, both feet, root and cart travel, jaw slip and true adapter output. Final exact negative active distances are separately recorded in final_contact_threshold_receipt.json; the0.5mm screen never certifies strict separation. A failed static pose does not prove dynamic walking impossible. No robust whole-body balance controller is asserted.

## Scope and corrections

The staged finite screen explored shallow/deeper crouch, shift/roll, moderate toe-in and narrowed stance, then constructed two explicit cycles and one local lean variant. Early stage2 toe-in poses retained zero yaw-rotor initial coordinates; those old rows are not closure-valid release candidates. Final cycles and the final best single/adjacent double explicitly correct the original belt coordinates. Historical raw rows are retained; no frozen file changes.

All LPs use actual04MCI, actual sole lowest-face vertices, original18 continuous caps, original mimic/equality Jacobians and conservative8-plane μ=.65 cone. Static0-speed output is not a torque-speed/350W dynamic qualification. No cart mouth redesign, simulator parameter adjustment, live root/foot write, dynamics, PPO or GPU occurred. The input is frozen04; actual cart004/20kg exists only in the sole Teacher experiment, while this robot-side static analysis applies declared0/8/20N horizontal mouth reactions.

Execution counters (IK samples means solved leg/neck target problems; function and iteration counts exposed separately):
```json
{
  "poses": 157,
  "IK_solves": 471,
  "IK_nfev": 3351,
  "FK_calls": 22856,
  "LP_solves": 36,
  "forward_calls": 201
}
```
