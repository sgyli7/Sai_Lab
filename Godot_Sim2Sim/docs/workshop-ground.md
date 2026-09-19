# Workshop ground surfaces

The zero-height entry sections of the three stair lanes previously rendered white box tops at exactly the same height as the workshop floor. They competed for the same depth samples as the paving when the camera moved (4.35 m² in ascent layouts, 4.91 m² in descent layouts).

`hub.gd` now renders only the exposed portion of each task box. Zero-height entries use the existing floor; partially buried obstacles have no underground visual faces or coplanar underside ink pass. Raised treads retain their original heights, widths and materials. All contact boxes remain unchanged.

The regression exercises the real task builder across all nine selections. It checks that no task triangles lie on the floor, the visible obstacles retain their dimensions, and all 21 collision boxes remain present. Before/after collision descriptors are identical. Task geometry decreases from 252 to 180 triangles per layout.

```bash
# Run after the project's Godot resources have been imported.
HUB_SURFACE_REPORT=/tmp/hub-surfaces.json godot --headless --path godot \
  --script res://tests/hub_task_surfaces.gd
```

[Measured results and source hash](workshop-ground-validation.json).
