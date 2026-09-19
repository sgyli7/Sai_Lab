shader_type sky;
float hash(vec2 p) { return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453); }
float noise(vec2 p) {
    vec2 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);
    return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x),f.y);
}
void sky() {
    float h=max(EYEDIR.y,0.0);
    vec3 base=mix(vec3(.58,.62,.59),vec3(.31,.38,.40),pow(h,.55));
    vec2 p=EYEDIR.xz/max(.12,EYEDIR.y+.14)*vec2(1.9,3.4);
    float n=noise(p)*.60+noise(p*2.1)*.28+noise(p*4.3)*.12;
    float cloud=smoothstep(.48,.70,n)*smoothstep(.01,.14,h);
    COLOR=mix(base,vec3(0.70,0.71,0.65),cloud);
}
