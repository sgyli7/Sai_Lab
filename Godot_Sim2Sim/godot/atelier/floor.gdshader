shader_type spatial;
render_mode specular_disabled;
uniform vec4 pigment : source_color = vec4(0.69,0.69,0.64,1.0);
uniform vec4 seam_color : source_color = vec4(0.45,0.48,0.47,1.0);
uniform vec4 backdrop : source_color = vec4(0.73,0.75,0.70,1.0);
uniform vec4 bay_paint : source_color = vec4(0.8784, 0.7412, 0.2196, 1.0);
uniform bool open_route = false;
uniform bool yard_dressing = false;
uniform bool neighbourhood = false;
varying vec3 world_position;
void vertex() {
    world_position = (MODEL_MATRIX * vec4(VERTEX,1.0)).xyz;
}
void fragment() {
    vec2 p = world_position.xz;
    vec2 cell = floor(p / 0.44);
    float variation = fract(sin(dot(cell,vec2(12.9898,78.233))) * 43758.5453);
    vec3 tile = pigment.rgb * (0.98 + variation * 0.04);
    // The courtyard is a flush surface treatment; no raised mesh or new contact plane.
    vec2 edge = abs(p - vec2(0.0,-0.11)) - vec2(1.85,1.54);
    float boundary = max(edge.x,edge.y);
    if (open_route) {
        vec2 apron = abs(p - vec2(2.45,0.55)) - vec2(0.85,0.66);
        boundary = min(boundary,max(apron.x,apron.y));
        vec2 return_lane = abs(p - vec2(-2.45,0.035)) - vec2(0.85,0.405);
        boundary = min(boundary,max(return_lane.x,return_lane.y));
    }
    if (yard_dressing) {
        vec2 rear = abs(p - vec2(0.05,-2.07)) - vec2(3.45,1.39);
        vec2 side = abs(p - vec2(3.02,-0.43)) - vec2(0.48,1.04);
        boundary = min(boundary, min(max(rear.x,rear.y),max(side.x,side.y)));
    }
    float yard = 1.0 - smoothstep(-0.005,0.005,boundary);
    vec2 dist = abs(fract((p + 0.22)/0.44)-0.5)*0.44;
    vec2 footprint = max(fwidth(p),vec2(0.00001));
    vec2 grid = 1.0 - smoothstep(vec2(0.0004),vec2(0.0004)+footprint,dist);
    float line = max(grid.x,grid.y);
    float line_fade = 1.0-smoothstep(0.01,0.035,max(footprint.x,footprint.y));
    tile = mix(tile,seam_color.rgb,line*line_fade*0.68);
    vec3 outside = pigment.rgb*0.90;
    if (neighbourhood) {
        // Complete ground surface across the district, with a curved service lane.
        float radius=length(p);
        float road=1.0-smoothstep(0.71,0.75,abs(radius-4.7));
        vec2 block=abs(fract(p/0.27)-0.5)*0.27;
        vec2 aa=max(fwidth(p),vec2(.00001));
        vec2 joint=1.0-smoothstep(vec2(.0006),vec2(.0006)+aa,block);
        float fade=1.0-smoothstep(.014,.05,max(aa.x,aa.y));
        outside=pigment.rgb*(.88+variation*.045);
        outside=mix(outside,seam_color.rgb,max(joint.x,joint.y)*fade*.32);
        outside=mix(outside,seam_color.rgb*.93,road*.56);
        float curb=1.0-smoothstep(.012,.012+length(aa),abs(abs(radius-4.7)-.74));
        outside=mix(outside,pigment.rgb*1.06,curb*.8);
    }
    vec3 color = mix(outside,tile,yard);
    // A quiet ink border gives the paved area an intentional perimeter.
    float border = 1.0-smoothstep(0.001,0.001+length(footprint),abs(boundary));
    color = mix(color,seam_color.rgb,border*0.6);
    // Flush yellow parking rectangle: one surface, no crossing decal boxes.
    vec2 bay = abs(p) - vec2(0.52, 0.41);
    float bay_distance = max(bay.x, bay.y);
    float bay_aa = max(length(footprint), 0.00001);
    float paint = 1.0 - smoothstep(0.009, 0.009 + bay_aa, abs(bay_distance));
    color = mix(color, bay_paint.rgb, paint);
    float haze = neighbourhood ? 0.0 : smoothstep(6.0,18.0,length(VERTEX));
    ALBEDO = color*(1.0-haze);
    EMISSION = backdrop.rgb*haze;
    ROUGHNESS=1.0;
}
void light() {
    float n=dot(NORMAL,LIGHT);
    float band=0.12+smoothstep(0.03,0.13,n)*0.48+smoothstep(0.55,0.65,n)*0.26;
    DIFFUSE_LIGHT+=LIGHT_COLOR*band*ATTENUATION/3.14159265;
}
