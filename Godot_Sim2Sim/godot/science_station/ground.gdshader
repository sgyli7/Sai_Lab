shader_type spatial;
render_mode specular_disabled;
uniform float station_influence = 1.0;
varying vec3 world;
void vertex(){world=(MODEL_MATRIX*vec4(VERTEX,1.)).xyz;}
float hash(vec2 p){
 uint h=uint(int(p.x))*1597334677u ^ uint(int(p.y))*3812015801u;
 h=(h^(h>>16u))*2246822519u;h=(h^(h>>13u))*3266489917u;h=h^(h>>16u);
 return float(h & 16777215u)/16777215.;
}
float noise(vec2 p){
 vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);
 return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x),f.y);
}
float field(vec2 p){return .66*noise(p)+.24*noise(p*2.07+3.2)+.10*noise(p*4.13-1.7);}
float ink(float d,float w){return 1.-smoothstep(w,w+max(fwidth(d),.003),abs(d));}
// Only a pixel-wide blend: material boundaries never use world-space blur.
float edge(float d){float w=max(fwidth(d)*.7,.001);return smoothstep(-w,w,d);}
float rect(vec2 p,vec2 c,vec2 h,float r){vec2 q=abs(p-c)-h+r;return length(max(q,vec2(0)))+min(max(q.x,q.y),0.)-r;}
float segment(vec2 p,vec2 a,vec2 b){vec2 d=b-a;return length(p-a-d*clamp(dot(p-a,d)/dot(d,d),0.,1.));}
float pavement(vec2 p){
 float d=min(rect(p,vec2(0,7),vec2(3,2.6),.36),rect(p,vec2(-6,.1),vec2(3.6,3.3),.55));
 d=min(d,rect(p,vec2(-3.5,-16.4),vec2(9.8,7.4),.9));
 d=min(d,length((p-vec2(-7.5,-.5))*vec2(1,1.1))-3.9);
 return min(d,rect(p,vec2(13,-2),vec2(10,6),.45));
}
void fragment(){
 vec2 p=world.xz;
 vec2 bend=vec2(field(p*.115+11.),field(p*.115-7.))-.5;
 vec2 q=p+bend*2.4;
 float broad=field(q*.19);
 // Apricot dunes and blue-lilac rock are painted as large, quiet pigment fields.
 vec3 amber=vec3(.90,.255,.035),apricot=vec3(1.,.46,.12);
 vec3 basalt=vec3(.19,.225,.33),mineral=vec3(.34,.39,.50);
 vec3 slab=vec3(.90,.865,.745),blue=vec3(.075,.34,.58);
 vec3 color=mix(amber,apricot,.56+.10*edge(broad-.51));
 color*=.99+.015*broad;
 // Bedrock follows the actual relief and slope. Broad mineral strata emerge
 // from the terrain itself; this is not a field of circular painted islands.
 vec3 world_normal=normalize((INV_VIEW_MATRIX*vec4(NORMAL,0.)).xyz);
 float slope=1.-clamp(world_normal.y,0.,1.);
 float exposure=max(smoothstep(mix(.035,.002,station_influence),mix(.16,.025,station_influence),slope),
  smoothstep(mix(3.5,.10,station_influence),mix(7.,.26,station_influence),world.y)*.40);
 float wind=field(p*.073+vec2(8,-4));
 float rock_signal=min(exposure+wind*.30-.62,field(p*.30+vec2(14,-3))-.38);
 float rock_width=max(fwidth(rock_signal),.004);
 float rock=smoothstep(-rock_width,rock_width,rock_signal);
 vec3 rock_color=mix(basalt,mineral,.25+.55*edge(field(p*.11+vec2(2.5,-8.))-.51));
 color=mix(color,rock_color,rock);
 color=mix(color,vec3(.22,.255,.32),ink(rock_signal,.002)*.19);
 // A few discontinuous bedding lines follow elevation around the slopes.
 float spacing=mix(1.45,.095,station_influence);
 float bedding=ink(mod(world.y+.035*sin(p.x*.8+p.y*.4)+spacing*30.,spacing)-spacing*.5,spacing*.016);
 bedding*=smoothstep(.001,.006,slope)*step(.49,field(p*.31));
 color=mix(color,vec3(.15,.19,.28),bedding*rock*.36);
 // Thin fissures and wind-scoured facets belong to exposed rock, with gaps.
 vec2 crack_p=p*.93+vec2(noise(p*.65),noise(p*.65+19.))*.34;
 vec2 ci=floor(crack_p),cf=fract(crack_p);float first=10.,second=10.;
 for(int ix=-1;ix<=1;ix++){for(int iz=-1;iz<=1;iz++){
  vec2 cell=vec2(float(ix),float(iz));vec2 key=ci+cell;
  vec2 jitter=vec2(hash(key),hash(key+vec2(71,29)));
  float v=length(cell+.15+.70*jitter-cf);
  if(v<first){second=first;first=v;}else{second=min(second,v);}
 }}
 float fissure=1.-smoothstep(.008,.015+fwidth(second-first),second-first);
 fissure*=smoothstep(.38,.62,field(p*.41))*rock;
 color=mix(color,vec3(.12,.17,.25),fissure*.38);
 // Sand physically reads as a thin deposit over the enamel slab: sharp,
 // chipped contact edges and a narrow exposed lip, never a sprayed halo.
 float d=pavement(p);
 float slab_mask=(1.-edge(d))*station_influence;
 float drift=field(p*vec2(.60,1.10)+4.);
 float grains=(noise(p*9.7)-.5)*.027+(noise(p*23.)-.5)*.009;
 float dust_depth=.012+.46*pow(max(0.,(drift-.38)/.62),2.);
 float dust_front=d+dust_depth+grains;
 float dust=edge(dust_front)*slab_mask;
 vec3 sand_color=color;
 color=mix(color,slab,slab_mask);
 color=mix(color,sand_color,dust);
 float deposit_lip=ink(dust_front+.011,.004)*slab_mask*(1.-dust);
 deposit_lip*=step(.46,noise(p*2.7));
 color=mix(color,vec3(.52,.38,.25),deposit_lip*.32);
 color=mix(color,vec3(.48,.49,.49),ink(d+.048,.010)*slab_mask*(1.-dust)*.24);
 // Pale compacted tracks have broken but well-defined aggregate shoulders.
 float road_d=abs(p.x+.7*sin(p.y*.19))-1.8+.065*sin(p.y*1.9)+.09*(noise(p*3.2)-.5);
 road_d=max(road_d,max(-25.4-p.y,p.y-12.));
 float road=(1.-edge(road_d))*station_influence*(1.-slab_mask);
 float branch=min(segment(p,vec2(0,7),vec2(-2,3.8)),segment(p,vec2(-2,3.8),vec2(-5,1)));
 branch=min(branch,segment(p,vec2(-5,1),vec2(5,5)));
 branch=min(branch,segment(p,vec2(5,5),vec2(0,7)));
 float path_d=branch-.54+.055*sin(p.x*3.+p.y)+.06*(noise(p*5.)-.5);
 float path=(1.-edge(path_d))*station_influence*(1.-slab_mask);
 float trail_width=.55*smoothstep(-24.,-21.5,p.x)*(1.-smoothstep(-12.5,-10.,p.x));
 float trail_d=abs(p.y+3.+.15*sin(p.x*.45))-trail_width+.07*(noise(p*3.7)-.5);
 trail_d=max(trail_d,max(-23.-p.x,p.x+10.));
 float trail=(1.-edge(trail_d))*station_influence*(1.-slab_mask);
 float track=max(road,path);
 color=mix(color,vec3(.82,.56,.30),trail);
 color=mix(color,vec3(.87,.71,.46),track);
 float track_lip=max(ink(road_d,.009),ink(path_d,.009))*station_influence*(1.-slab_mask);
 color=mix(color,vec3(.64,.43,.24),track_lip*.22);
 float seam=max(ink(mod(p.x+1.,2.)-1.,.006),ink(mod(p.y+1.,2.)-1.,.006));
 color=mix(color,vec3(.52,.54,.54),seam*slab_mask*(1.-dust)*.28);
 float guide=max(ink(p.x+.7*sin(p.y*.19)-.94,.045),ink(p.x+.7*sin(p.y*.19)+.94,.045));
 guide*=smoothstep(-25.,-23.,p.y)*(1.-smoothstep(10.,12.,p.y))*station_influence;
 float bay=ink(rect(p,vec2(13,-2),vec2(9.6,5.6),.35),.025)*station_influence;
 float dotted=ink(branch,.012)*step(.57,fract((p.x+p.y*.47)*2.7))*station_influence;
 color=mix(color,blue,max(guide,max(bay,dotted*.7))*(1.-dust*.5));
 float trench=ink(p.x+11.35,.12)*(1.-smoothstep(7.9,8.,abs(p.y+12.)))*station_influence;
 color=mix(color,vec3(.21,.25,.30),trench*(.55+.45*step(.45,fract(p.y*5.))));
 float cover=1.-smoothstep(-.005,.008,rect(p,vec2(-3,-12),vec2(.7,.43),.06));
 color=mix(color,vec3(.36,.44,.49),cover*station_influence);
 // Sparse wind strokes. High-frequency grain is intentionally absent.
 vec2 cell=floor(p*1.35),local=fract(p*1.35)-.5;float seed=hash(cell);
 float stroke=ink(local.y+.03*sin(local.x*8.+seed*4.),.0012);
 stroke*=1.-smoothstep(.08+seed*.19,.11+seed*.19,abs(local.x));
 stroke*=step(.53,seed)*(1.-max(slab_mask,rock));
 stroke*=1.-smoothstep(.025,.10,length(fwidth(p)));
 color=mix(color,vec3(.54,.25,.14),stroke*.55);
 // Small angular mineral flakes interrupt the sand near natural ground.
 vec2 chip_cell=floor(p*2.4),chip_local=fract(p*2.4)-.5;
 float chip_seed=hash(chip_cell+93.);
 chip_local-=vec2(hash(chip_cell+17.),hash(chip_cell-9.))*.34-.17;
 float chip_shape=abs(chip_local.x+chip_local.y*.47)*1.1+abs(chip_local.y)*1.7;
 float chip=1.-smoothstep(.033,.033+max(fwidth(chip_shape),.008),chip_shape);
 chip*=step(.91,chip_seed)*(1.-slab_mask)*smoothstep(.20,.50,field(p*.24));
 chip*=1.-smoothstep(.015,.065,length(fwidth(p)));
 color=mix(color,vec3(.32,.34,.41),chip*.72);
 // Slightly different warm/cool planes make decimetre relief readable from
 // robot height; buildings keep their existing enamel lighting.
 float facing=dot(world_normal,normalize(vec3(-.394,.743,.541)))-.743;
 float lee=1.-edge(facing+.08);
 float sunward=edge(facing-.085);
 color=mix(color,mix(color*.64,vec3(.23,.22,.34),.25),lee*.58*(1.-slab_mask));
 color+=vec3(.025,.027,.009)*sunward*(1.-slab_mask);
 float distance_fade=smoothstep(38.,155.,length(p-vec2(0,-6)));
 color=mix(color,vec3(.28,.37,.56),distance_fade*.58);
 ALBEDO=color;ROUGHNESS=1.;
}
void light(){
 float band=.19+.67*smoothstep(.08,.42,dot(NORMAL,LIGHT));
 DIFFUSE_LIGHT+=LIGHT_COLOR*band*ATTENUATION/3.14159265;
}
