shader_type spatial;
render_mode specular_disabled;
uniform vec4 pigment : source_color = vec4(0.6, 0.58, 0.54, 1.0);
uniform float shade_softness = 0.05;
uniform float hatch_strength = 0.075;
varying vec3 surface_position;
varying vec3 surface_normal;
varying vec3 world_normal;
void vertex() {
    surface_position = VERTEX;
    surface_normal = NORMAL;
    world_normal = MODEL_NORMAL_MATRIX * NORMAL;
}
void fragment() {
    vec3 an = abs(surface_normal);
    vec2 p = surface_position.xy;
    if (an.y > an.x && an.y > an.z) { p = surface_position.xz; }
    else if (an.x > an.z) { p = surface_position.zy; }
    float coordinate = (p.x + p.y * 0.62) * 330.0;
    float footprint = fwidth(coordinate);
    float distance_to_line = abs(fract(coordinate) - 0.5);
    float line = 1.0 - smoothstep(0.045, 0.045 + footprint, distance_to_line);
    float fade = 1.0 - smoothstep(0.16, 0.50, footprint);
    float shade = 1.0 - smoothstep(0.10,0.65,dot(normalize(world_normal), normalize(vec3(-0.4,0.8,0.3))));
    ALBEDO = pigment.rgb * (1.0 - line * fade * shade * hatch_strength);
    ROUGHNESS = 0.95;
}
void light() {
    float n = dot(NORMAL, LIGHT);
    float middle = smoothstep(0.08 - shade_softness, 0.08 + shade_softness, n);
    float bright = smoothstep(0.60 - shade_softness, 0.60 + shade_softness, n);
    float band = 0.12 + middle * 0.48 + bright * 0.26;
    DIFFUSE_LIGHT += LIGHT_COLOR * band * ATTENUATION / 3.14159265;
}
