shader_type spatial;
render_mode specular_schlick_ggx;
uniform bool diagnostic_surface = false;
uniform float aperture_radius = 0.00222;
uniform float reflection_strength = 0.23;
varying vec3 local_position;
varying vec3 local_normal;
void vertex() {
    local_position=VERTEX;
    local_normal=NORMAL;
}
void fragment() {
    float rim = pow(1.0 - abs(dot(NORMAL, VIEW)), 3.0);
    ALBEDO = mix(vec3(0.018,0.016,0.023), vec3(0.15,0.105,0.20),rim);
    // Compiled lens coordinates are identical for the two snapshot robots.
    // A restrained coated-glass ring and reflected streak clarify the aperture.
    vec2 q=(local_position.xy-vec2(0.00921642,-0.00300745))/aperture_radius;
    float radial=length(q);
    float ring=1.0-smoothstep(0.015,0.050,abs(radial-0.75));
    vec2 glint=(q-vec2(-0.30,0.30))/vec2(0.21,0.085);
    float reflected=exp(-dot(glint,glint)*1.5);
    float face=smoothstep(0.78,0.98,abs(normalize(local_normal).z));
    EMISSION=(vec3(0.045,0.03,0.06)*ring+vec3(1.0,1.0,0.87)*reflection_strength*reflected)*face;
    METALLIC = 0.35;
    ROUGHNESS = 0.19;
    SPECULAR = 0.75;
    if (diagnostic_surface) { ALBEDO=vec3(1.0,0.0,0.0);EMISSION=vec3(1.0,0.0,0.0); }
}
