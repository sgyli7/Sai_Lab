shader_type sky;
void sky(){
 float h=clamp(EYEDIR.y,0.,1.);
 // Clear azure at the zenith, a pale blue band only close to the horizon.
 COLOR=mix(vec3(.075,.35,.70),vec3(.008,.14,.52),pow(h,.32));
}
