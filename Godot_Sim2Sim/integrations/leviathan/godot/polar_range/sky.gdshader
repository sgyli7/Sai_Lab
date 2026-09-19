shader_type sky;
void sky(){
 float h=max(EYEDIR.y,0.);
 vec3 blue=mix(vec3(.58,.70,.79),vec3(.18,.37,.57),pow(h,.42));
 vec2 p=EYEDIR.xz/(h+.22);
 float band=sin(p.y*3.+sin(p.x*1.7)*.65)+.22*sin(p.y*9.+p.x*2.);
 float cloud=smoothstep(.7,1.2,band)*smoothstep(.08,.35,h)*.23;
 COLOR=mix(blue,vec3(.85,.88,.88),cloud);
}
