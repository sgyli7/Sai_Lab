# Closed006 archival reproduction boundary

source_snapshot keeps original numerical sources/configs/declarations/coldstates and actual sceneXML byte-for-byte. Do not execute archived prepare/run/analyze/render/finish in place: they target the old raw directory. All large MJB, trajectories, prestep and classified-contact logs remain at the indexed ProjectBackups location. Producer85-record manifest and independent author files stay separate.

The standard-library-only reproduction/prepare_fresh_run.py can later prepare one selected attempt in a non-existing new directory. It does not run MuJoCo,forward,IK,LP or dynamics. It relocates only scene mesh URIs and runner I/O paths, keeps source planning/physical/actuator data, and prints the future command. Neither preparer nor derived runner was executed here; only AST/XML/hash validation was performed. A new bounded experiment order is needed for any actual replay. Runtime environment remains the producer’s Python/MuJoCo3.10/numpy/scipy setup.

Attempts1/2 use originalPD,pitch.25 and are stopped near limit.3/4 use declared gravity position-target bias,pitch.4;4 is the open-jaw countercheck for3, though adaptive rod-following creates different worldpaths. All use experimental k20 passivepitch fixture and nominal.125Nm wheel Coulombcaps; none changes04/004hardware. See reproduction_contract.json for exact sources, plans, mesh dependencies and videos. Pending ingress_geometry_review/full_body_readiness are excluded.

The local trial3 result35.76mm/5.18s/~6.024N is finite contact evidence, not a whole-robot0.5m/20kg qualification. Existing approach shaft contact,prepush,edge-band contact and release closure limit are preserved. Mouth hardware stays unchanged. See independent_review.md/json for final interpretation.
