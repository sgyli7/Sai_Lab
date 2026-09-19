shader_type spatial;
render_mode specular_disabled;
uniform sampler2D printing : source_color, filter_linear_mipmap_anisotropic, repeat_disable;
void fragment() {
    ALBEDO = texture(printing,UV).rgb;
    ROUGHNESS = 0.95;
}
void light() {
    float n=dot(NORMAL,LIGHT);
    float band=0.12+smoothstep(0.03,0.13,n)*0.48+smoothstep(0.55,0.65,n)*0.26;
    DIFFUSE_LIGHT+=LIGHT_COLOR*band*ATTENUATION/3.14159265;
}
