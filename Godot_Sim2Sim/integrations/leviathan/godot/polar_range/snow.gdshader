shader_type spatial;
render_mode cull_back;
varying vec3 world_pos;
varying vec3 world_normal;
float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
float noise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),mix(hash(i+vec2(0,1)),hash(i+vec2(1)),f.x),f.y);}
void vertex(){world_pos=(MODEL_MATRIX*vec4(VERTEX,1.)).xyz;world_normal=normalize(MODEL_NORMAL_MATRIX*NORMAL);}
void fragment(){
 vec2 p=world_pos.xz;
 float broad=.5+.18*sin(p.x*.012+sin(p.y*.017))+.10*sin(p.y*.036+p.x*.009);
 float footprint=max(length(dFdx(p)),length(dFdy(p)));
 broad=mix(broad,.5,smoothstep(12.,80.,footprint));
 vec3 snow=mix(vec3(.45,.55,.64),vec3(.68,.74,.78),smoothstep(.12,.85,broad));
 float steep=1.-smoothstep(.52,.82,world_normal.y);
 float strata=smoothstep(.45,.62,noise(p*.012+world_pos.y*.022));
 vec3 rock=mix(vec3(.28,.38,.49),vec3(.46,.56,.64),strata);
 snow=mix(snow,rock,steep*.86);
 float lee=1.-smoothstep(-.18,.18,dot(world_normal,normalize(vec3(-.7,.15,.45))));
 snow=mix(snow,vec3(.37,.48,.60),lee*.18);
 ALBEDO=snow;ROUGHNESS=.94;
}
void light(){
 float band=.27+.73*smoothstep(.05,.6,dot(NORMAL,LIGHT));
 DIFFUSE_LIGHT+=LIGHT_COLOR*band*ATTENUATION/3.14159265;
}
