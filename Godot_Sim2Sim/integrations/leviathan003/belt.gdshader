shader_type spatial;
render_mode cull_back;
uniform vec4 paint:source_color=vec4(0.4,0.4,0.4,1.);
uniform float metal=.35;
uniform float travel_m=0.;
const vec2 path[29]=vec2[](vec2(-4.15000000,3.37000000),vec2(4.15000000,3.37000000),vec2(4.40105450,3.33694810),vec2(4.63500000,3.24004460),vec2(4.83589360,3.08589360),vec2(4.99004460,2.88500000),vec2(5.08694810,2.65105450),vec2(5.12000000,2.40000000),vec2(5.08694810,2.14894550),vec2(4.99004460,1.91500000),vec2(4.83589360,1.71410640),vec2(4.63500000,1.55995540),vec2(2.99500000,0.37923740),vec2(2.78034900,0.29032600),vec2(2.55000000,0.26000000),vec2(-2.55000000,0.26000000),vec2(-2.78034900,0.29032600),vec2(-2.99500000,0.37923740),vec2(-4.63500000,1.55995540),vec2(-4.83589360,1.71410640),vec2(-4.99004460,1.91500000),vec2(-5.08694810,2.14894550),vec2(-5.12000000,2.40000000),vec2(-5.08694810,2.65105450),vec2(-4.99004460,2.88500000),vec2(-4.83589360,3.08589360),vec2(-4.63500000,3.24004460),vec2(-4.40105450,3.33694810),vec2(-4.15000000,3.37000000));
const vec3 wheels[5]=vec3[](vec3(-4.15000000,-0.60000000,0.88000000),vec3(-2.55000000,-1.85000000,0.80000000),vec3(0.00000000,-1.82000000,0.83000000),vec3(2.55000000,-1.85000000,0.80000000),vec3(4.15000000,-0.60000000,0.88000000));
vec4 belt(float phase){
    float total=0.;for(int i=0;i<28;i++){total+=length(path[i+1]-path[i]);}
    float s=mod(phase,total);
    for(int i=0;i<28;i++){
        vec2 d=path[i+1]-path[i];float len=length(d);
        if(s<=len){return vec4(path[i]+d*s/len-vec2(0.,3.00000000),d/len);}s-=len;
    }
    return vec4(path[0]-vec2(0.,3.00000000),1.,0.);
}
void vertex(){
    // The glTF exporter flips V: static=1, belt=0, rolling wheel=-1.
    if(UV.y<-.5){
        int index=clamp(int(round(UV.x)),0,4);
        vec3 wheel=wheels[index];float a=-travel_m/wheel.z;float c=cos(a);float s=sin(a);
        vec2 d=VERTEX.xz-wheel.xy;
        VERTEX.xz=wheel.xy+vec2(c*d.x-s*d.y,s*d.x+c*d.y);
        vec2 n=NORMAL.xz;NORMAL.xz=vec2(c*n.x-s*n.y,s*n.x+c*n.y);
    }else if(UV.y<.5){
        vec4 old=belt(UV.x);vec4 next=belt(UV.x+travel_m);
        vec2 oldn=vec2(-old.w,old.z);vec2 newn=vec2(-next.w,next.z);
        vec2 delta=VERTEX.xz-old.xy;
        VERTEX.xz=next.xy+next.zw*dot(delta,old.zw)+newn*dot(delta,oldn);
        NORMAL.xz=next.zw*dot(NORMAL.xz,old.zw)+newn*dot(NORMAL.xz,oldn);
    }
}
void fragment(){ALBEDO=paint.rgb;METALLIC=metal;ROUGHNESS=.62;}
