# Bounded ingress geometry review

Result: no complete 50 mm fixed-torso path has been proved. The observed upper-mouth/shaft collision does not establish a hardware redesign requirement. Repair the Teacher ingress and coordinate body retreat before a new dynamic experiment. No frozen hardware was modified.

- Original006 nominal phi=0.4 rad, TCP 6 mm high: upper_bill_envelope/cart_rotating_shaft intersects by 0.6304 mm near final seating. This is a concrete invalid nominal path. ROM margin remains 0.6085 rad; joint reach is not its cause. Actual adaptive drift from006 is not replayed.
- World-head-pitch10deg sloped approach: TCP (270,0,426) mm -> (300,0,422) mm -> (300,0,420) mm; jaw0.29rad, then close to0.19682523rad. All sampled approach/seating/closure positions have no negative non-target or active-self contact. Minimum reported clearance is zero at lower-bill/yoke with overlapping AABBs, so positive safety clearance is NOT certified. Seating phase minimum is0.21885mm. Treat this as a promising geometric corridor, not released ingress.
- Holding pitch10deg and pulling the cart50mm with fixed torso avoids shaft interference (minimum3.7701mm), but neck_pitch_envelope hits upper_bill_envelope from the33.333mm pull sample onward:2.1135mm overlap at33.333mm,12.7117mm at50mm. No such active self contact was found in sampled pull points through30mm. Full50mm therefore FAILS. Coordinate body retreat or alter head pitch; do not infer the mouth itself must change.
- Pure vertical30mm approach at pitch10deg crosses the lower mouth through the rod by8.3653mm and is rejected.

Next minimal action: Teacher should use the supplied true rod-derived head-local TCP, start with the10deg sloped candidate, establish strictly positive non-target clearance, and combine retreating body motion with the mouth rather than retracting the entire50mm through the neck. These samples grant neither continuous sweep safety nor physical grip/traction success. Whole-body Teacher remains a separate static readiness task.

Inputs and exact qpos are in summary.json and sampled_paths.json. Original04 and004 hashes remained unchanged. Root z0.005000002806m; rod-derived head-local TCP[0.13750001972253326,0,-0.03849996489687674]m. No5.5mm reference offset used.

Budget: 3 saved paths x50 samples; an earlier three-path pass failed only final JSON serialization, then one local analytic ingress revision was made. Total300 offline IK/FK/collision sample evaluations,4 distinct ingress variants. 0 integrations,0 training,0 LP,0 GPU. No new dynamic attempts. Distance zero with overlapping AABBs is unresolved grazing, never described as positive clearance. Full runtime source and phase samples preserved.
