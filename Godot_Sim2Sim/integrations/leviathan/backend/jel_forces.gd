extends RefCounted
## Canonical +X/+Y/+Z math shared in formula with leviathan.jel.
static func v(a) -> Vector3:
	return Vector3(a[0],a[1],a[2])
static func arr(a: Vector3) -> Array:
	return [a.x,a.y,a.z]
static func orientation_state(r: Basis, omega: Vector3) -> Dictionary:
	var roll := asin(clampf(r.y.z,-1.,1.))
	var yaw := atan2(-r.y.x,r.y.y)
	var pitch := atan2(-r.x.z,r.z.z)
	var axes := Basis(Vector3(0,0,1),Vector3(cos(yaw),sin(yaw),0),Vector3(-sin(yaw)*cos(roll),cos(yaw)*cos(roll),sin(roll)))
	return {"angles":Vector3(yaw,roll,pitch),"rates":axes.inverse()*omega,"axes":axes}
static func point(pose: Transform3D, linear: Vector3, omega: Vector3, local) -> Dictionary:
	var arm := pose.basis*v(local)
	return {"position":pose.origin+arm,"velocity":linear+omega.cross(arm)}
static func evaluate(group: Dictionary, hp: Transform3D, tp: Transform3D, hv: Vector3, hw: Vector3, tv: Vector3, tw: Vector3, heave_target: float, yaw_target: float, hydraulic_state: Dictionary={}, dt: float=0.) -> Dictionary:
	var cfg: Dictionary=group.configuration
	var sim: Dictionary=group.simulation
	var relative := hp.basis.transposed()*tp.basis
	var orientation := orientation_state(relative,hp.basis.transposed()*(tw-hw))
	var angles: Vector3=orientation.angles
	var rates: Vector3=orientation.rates
	var heave: float=(hp.basis.transposed()*(tp.origin-hp.origin)-v(group.pivot_hull_local_m)).z
	var forces: Array[Dictionary]=[]
	var lengths: Array[float]=[]
	var speeds: Array[float]=[]
	var units: Array[Vector3]=[]
	var values: Array[float]=[]
	for i in range(4):
		var a:=point(hp,hv,hw,group.lift_upper_hull_local_m[i])
		var b:=point(tp,tv,tw,group.lift_lower_truck_local_m[i])
		var displacement:Vector3=b.position-a.position
		var length:=displacement.length()
		var direction:=displacement/length
		lengths.append(length);speeds.append(direction.dot(b.velocity-a.velocity));units.append(direction)
		forces.append({"hull_point":a.position,"truck_point":b.position,"force_truck":Vector3.ZERO,"kind":"lift"})
	var hydraulic:Dictionary=lift_hydraulics(group,lengths,speeds,units,heave_target,hydraulic_state,dt)
	values.assign(hydraulic["values"])
	var lift_moment:=Vector3.ZERO
	var pressures:Array[float]=[]
	var lift_vertical:=0.
	for i in range(4):
		forces[i].force_truck=units[i]*values[i]
		lift_moment+=(forces[i].truck_point-tp.origin).cross(forces[i].force_truck)
		pressures.append(values[i]/(PI*pow(float(cfg.lift.bore_m),2)/4.))
		lift_vertical-=values[i]*units[i].z
	var bounds:=Vector3(deg_to_rad(cfg.working_envelope.yaw_deg[1]),deg_to_rad(cfg.working_envelope.roll_deg[1]),deg_to_rad(cfg.working_envelope.pitch_deg[1]))
	var generalized:=Vector3.ZERO
	for i in range(3):
		var excess:=maxf(absf(angles[i])-bounds[i]*float(sim.angular_buffer_start_fraction),0.)
		generalized[i]=-signf(angles[i])*float(sim.angular_buffer_stiffness_nm_rad)*excess
		if excess>0.:generalized[i]-=float(sim.angular_buffer_damping_nms_rad)*rates[i]
		generalized[i]=clampf(generalized[i],-float(sim.angular_buffer_limit_nm),float(sim.angular_buffer_limit_nm))
	var buffer_torque: Vector3=hp.basis*(orientation.axes.transposed().inverse()*generalized)
	var requested:=clampf(float(sim.yaw_stiffness_nm_rad)*(yaw_target-angles.x)-float(sim.yaw_damping_nms_rad)*rates.x,-float(sim.net_yaw_command_limit_nm),float(sim.net_yaw_command_limit_nm))
	var residual:=requested-(lift_moment+buffer_torque).dot(hp.basis.z)
	var steering: Dictionary=cfg.steering
	var cap:=PI*pow(float(steering.bore_m),2)/4.
	var annulus:=cap-PI*pow(float(steering.rod_m),2)/4.
	var signs:Array[float]=[]
	var areas:Array[float]=[]
	var capacities:Array[float]=[]
	var steering_speeds:Array[float]=[]
	var steering_units:Array[Vector3]=[]
	var capacity:=0.
	for i in range(4):
		var a:=point(hp,hv,hw,group.steering_upper_hull_local_m[i])
		var b:=point(tp,tv,tw,group.steering_lower_truck_local_m[i])
		var displacement:Vector3=b.position-a.position
		var length:=displacement.length()
		var direction:=displacement/length
		var coefficient:float=(b.position-tp.origin).cross(direction).dot(hp.basis.z)
		var direction_sign:=signf(residual*coefficient)
		var area:=cap if direction_sign>=0. else annulus
		var maximum:=area*float(steering.pressure_limit_pa)*float(steering.torque_efficiency_assumption)
		capacity+=absf(coefficient)*maximum
		signs.append(direction_sign);areas.append(area);capacities.append(maximum)
		steering_speeds.append(direction.dot(b.velocity-a.velocity));steering_units.append(direction)
		forces.append({"hull_point":a.position,"truck_point":b.position,"force_truck":Vector3.ZERO,"kind":"steering","length_m":length})
	var steering_values:Array[float]=[]
	var positive_power:Array[float]=[]
	var power:=0.
	var flow:=0.
	for i in range(4):
		var value:=signs[i]*capacities[i]*minf(1.,absf(residual)/maxf(capacity,1.))
		var p:=maxf(value*steering_speeds[i],0.)
		steering_values.append(value);positive_power.append(p);power+=p
		if p>0.:flow+=areas[i]*absf(steering_speeds[i])
	var supply_scale:=minf(1.,minf(float(steering.power_limit_w_per_truck)*float(sim.pump_efficiency)*float(steering.torque_efficiency_assumption)/maxf(power,1.),float(steering.pump_flow_limit_l_min_per_truck)/60000./maxf(flow,1e-12)))
	for i in range(4):
		if positive_power[i]>0.:steering_values[i]*=supply_scale
		forces[4+i].force_truck=steering_values[i]*steering_units[i]
	var failures:Array[String]=[]
	for i in range(3):
		if absf(angles[i])>bounds[i]+1e-4:failures.append(["yaw","roll","pitch"][i])
	if heave<float(cfg.working_envelope.heave_m[0])-.002 or heave>float(cfg.working_envelope.heave_m[1])+.002:failures.append("heave")
	for length in lengths:
		if length<float(cfg.lift.pin_closed_length_m) or length>float(cfg.lift.pin_closed_length_m)+float(cfg.lift.stroke_m):
			failures.append("lift_stroke");break
	for i in range(4):
		if forces[4+i].length_m<float(steering.pin_closed_length_m) or forces[4+i].length_m>float(steering.pin_closed_length_m)+float(steering.stroke_m):
			failures.append("steering_stroke");break
	var total_force:=Vector3.ZERO
	var total_moment:=buffer_torque
	for item in forces:
		total_force+=item.force_truck
		total_moment+=(item.truck_point-tp.origin).cross(item.force_truck)
	return {"forces":forces,"buffer_torque_world":buffer_torque,"telemetry":{
		"angles_rad":arr(angles),"rates_rad_s":arr(rates),"heave_m":heave,"lift_lengths_m":lengths,"lift_speeds_m_s":speeds,"lift_forces_n":values,"lift_pressure_pa":pressures,
		"lift_supply_scale":hydraulic.supply_scale,"lift_power_demand_w":hydraulic.power_demand,"lift_flow_demand_l_min":hydraulic.flow_demand*60000.,"hydraulics":hydraulic.telemetry,
		"lift_vertical_n":lift_vertical,"lift_yaw_moment_nm":lift_moment.dot(hp.basis.z),"steering_forces_n":steering_values,"steering_supply_scale":supply_scale,
		"steering_power_demand_w":power,"steering_flow_demand_l_min":flow*60000.,"net_yaw_command_nm":requested,"total_force_world_n":arr(total_force),"total_moment_world_nm":arr(total_moment),
		"envelope_failures":failures,"guide_reaction_status":"native constrained reaction; Jolt exact reaction readback pending",
		"guide_friction_coefficient":sim.get("guide_friction_coefficient"),"guide_friction_applied":false,
		"hydraulic_model":"finite linearized accumulator C dp/dt = Q - A dL/dt; Q+ pump charge and Q- metered return; bounded pressure with explicit relief/cavitation telemetry"}}

static func lift_hydraulics(group:Dictionary,lengths:Array[float],speeds:Array[float],units:Array[Vector3],heave_target:float,storage:Dictionary,dt:float) -> Dictionary:
	var cfg:Dictionary=group.configuration
	var sim:Dictionary=group.simulation
	var area:=PI*pow(float(cfg.lift.bore_m),2)/4.
	var limit:=float(cfg.lift.pressure_limit_pa)
	var compliance:=4.*area*area/float(sim.linear_stiffness_n_m)
	var previous:Array[float]=[]
	var accumulator:Array[float]=[]
	var mean_pressure:Array[float]=[]
	var targets:Array[float]=[]
	var flow_values:Array[float]=[]
	var values:Array[float]=[]
	var old_energy:=0.
	var power_demand:=0.
	var flow_demand:=0.
	var return_demand:=0.
	for i in range(4):
		var target:=clampf((float(group.nominal_support_n)/4.+float(sim.linear_stiffness_n_m)/4.*(float(group.lift_rest_lengths_m[i])-heave_target-lengths[i]))/area,0.,limit)
		var p:float=storage.accumulator_pressure_pa[i] if dt>0. and storage.has("accumulator_pressure_pa") else target
		targets.append(target);previous.append(p);old_energy+=.5*compliance*p*p
		var q:=compliance*(target-p)/float(sim.pressure_response_time_s) if dt>0. else 0.
		flow_values.append(q);flow_demand+=maxf(q,0.);return_demand-=minf(q,0.)
		power_demand+=maxf(p,target)*maxf(q,0.)
	var supply_scale:=1.
	var return_scale:=1.
	if dt>0.:
		if not storage.has("initial_stored_energy_j"):storage.initial_stored_energy_j=old_energy
		var flow_limit:=float(sim.lift_flow_limit_l_min_per_truck)/60000.
		supply_scale=minf(1.,minf(flow_limit/maxf(flow_demand,1e-12),float(sim.lift_pump_electrical_power_w_per_truck)*float(sim.lift_pump_efficiency)/maxf(power_demand,1.)))
		return_scale=minf(1.,flow_limit/maxf(return_demand,1e-12))
	var relief_energy:=0.
	var cavitation:=0.
	var total_down:=0.
	var stored_energy:=0.
	var pump_flow:=0.
	var return_flow:=0.
	var pump_power:=0.
	var return_power:=0.
	for i in range(4):
		flow_values[i]*=supply_scale if flow_values[i]>=0. else return_scale
		var raw:=previous[i]+dt/compliance*(flow_values[i]-area*speeds[i])
		var p:=clampf(raw,0.,limit)
		accumulator.append(p);mean_pressure.append((previous[i]+p)/2.)
		relief_energy+=.5*compliance*(pow(maxf(raw,0.),2)-p*p)
		cavitation+=compliance*maxf(-raw,0.)
		stored_energy+=.5*compliance*p*p
		var force:=clampf(area*mean_pressure[i]-float(sim.linear_damping_ns_m)/4.*speeds[i],0.,area*limit)
		values.append(force);total_down-=force*units[i].z
		pump_flow+=maxf(flow_values[i],0.);return_flow-=minf(flow_values[i],0.)
		pump_power+=mean_pressure[i]*maxf(flow_values[i],0.);return_power-=mean_pressure[i]*minf(flow_values[i],0.)
	var scale:=minf(1.,float(cfg.working_envelope.truck_vertical_reaction_limit_n)/maxf(total_down,1.))
	var mechanical_work:=0.
	var damping_loss:=0.
	var limiter:=false
	var displacement_flow:Array[float]=[]
	for i in range(4):
		values[i]*=scale
		mechanical_work+=dt*values[i]*speeds[i]
		damping_loss+=dt*maxf((area*mean_pressure[i]-values[i])*speeds[i],0.)
		limiter=limiter or (speeds[i]<0. and values[i]<area*mean_pressure[i]-1.)
		displacement_flow.append(area*speeds[i]*60000.)
	var telemetry:Dictionary={"accumulator_pressure_pa":accumulator,"compliance_m3_pa_per_cylinder":compliance,"stored_energy_j":stored_energy,
		"pump_flow_l_min":pump_flow*60000.,"return_flow_l_min":return_flow*60000.,"cylinder_displacement_flow_l_min":displacement_flow,
		"pump_hydraulic_power_w":pump_power,"relief_energy_j":relief_energy,"cavitation_deficit_m3":cavitation,"pressure_limiter_active":limiter}
	if dt>0.:
		storage.accumulator_pressure_pa=accumulator.duplicate()
		var updates:Dictionary={"pump_work_j":pump_power*dt,"return_loss_j":return_power*dt,"mechanical_work_j":mechanical_work,"damping_loss_j":damping_loss,"relief_energy_j":relief_energy,
			"energy_residual_j":stored_energy-old_energy-(pump_power*dt-return_power*dt-mechanical_work-damping_loss-relief_energy),"cavitation_deficit_m3":cavitation}
		for key in updates:storage[key]=float(storage.get(key,0.))+float(updates[key])
		telemetry.cumulative=storage.duplicate(true)
	return {"values":values,"telemetry":telemetry,"supply_scale":supply_scale,"power_demand":power_demand,"flow_demand":flow_demand}
