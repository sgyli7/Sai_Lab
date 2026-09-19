extends SceneTree
var clock:=preload("res://hub/simulation_clock.gd").new()
var probe:Probe
var rates:=[.1,1.,3.]
var index:=-1
var results:Array=[]
func _initialize() -> void:
	probe=Probe.new();root.add_child(probe)
	advance.call_deferred()
func advance() -> void:
	index+=1
	if index==rates.size():
		print("SIMULATION_CLOCK ",JSON.stringify(results))
		quit(0 if results.all(func(x):return x.passed) else 1);return
	clock.set_rate(rates[index]);probe.count=0;probe.maximum_error=0.
func _process(_delta:float) -> bool:
	if index>=0 and index<rates.size() and probe.count>=100:
		var capacity:float=Engine.max_physics_steps_per_frame*30./2000.
		results.append({"rate":rates[index],"physics_delta_error":probe.maximum_error,"scheduling_capacity":capacity,
			"passed":probe.maximum_error<1e-10 and capacity>=rates[index]})
		advance()
	return false
class Probe extends Node:
	var count:=0
	var maximum_error:=0.
	func _physics_process(delta:float) -> void:
		count+=1;maximum_error=maxf(maximum_error,absf(delta-.0005))
