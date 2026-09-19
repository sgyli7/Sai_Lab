extends AcceptDialog
signal vehicle_selected(model:String)

func _ready()->void:
	title="03 · 极地雪原"
	dialog_text="选择极地载具\nSainiverse v0.1：新车驾驶与舱内机器人\n001：原有 Sai 协作场景"
	get_ok_button().text="返回"
	add_button("驾驶 Sainiverse v0.1",true,"003")
	add_button("进入 001",true,"001")
	custom_action.connect(func(action:String):vehicle_selected.emit(action);hide())
	var fontpath:="res://atelier/ui_font.tres"
	if ResourceLoader.exists(fontpath):add_theme_font_override("font",load(fontpath))
