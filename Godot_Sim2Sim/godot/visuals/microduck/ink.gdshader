shader_type spatial;
render_mode cull_front, unshaded, shadows_disabled;
uniform vec4 ink_color : source_color = vec4(0.075, 0.071, 0.085, 1.0);
uniform float line_pixels = 0.75;
void vertex() {
    vec4 p = MODELVIEW_MATRIX * vec4(VERTEX, 1.0);
    vec3 n = normalize(MODELVIEW_NORMAL_MATRIX * NORMAL);
    vec2 direction = (PROJECTION_MATRIX * vec4(n, 0.0)).xy;
    direction /= max(length(direction), 0.0001);
    vec4 clip = PROJECTION_MATRIX * p;
    float fade = 1.0 - smoothstep(3.0, 8.0, -p.z);
    clip.xy += direction * line_pixels * fade * 2.0 / VIEWPORT_SIZE * clip.w;
    POSITION = clip;
}
void fragment() { ALBEDO = ink_color.rgb; }
