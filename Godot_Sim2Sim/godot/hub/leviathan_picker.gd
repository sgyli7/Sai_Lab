extends AcceptDialog
signal vehicle_selected(model:String)

func _ready()->void:
	title="03 · 极地雪原"
	dialog_text="选择利维坦\n003：100 km/h 高速驾驶\n001：原有 Sai 协作场景"
	get_ok_button().text="返回"
	add_button("驾驶 003",true,"003")
	add_button("进入 001",true,"001")
	custom_action.connect(func(action:String):vehicle_selected.emit(action);hide())
	var fontpath:="res://atelier/ui_font.tres"
	if ResourceLoader.exists(fontpath):add_theme_font_override("font",load(fontpath))
