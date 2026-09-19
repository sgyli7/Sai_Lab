extends RefCounted
## Retained settings from the six-view, motion and shared-window timing checks.
## Explicit environment values remain available for reproducible comparisons.
const DEFAULTS := {
	"MD_UI_FONT": "1", "MD_COMPACT_HUD": "1", "MD_OPEN_ROUTE": "1", "MD_ROBOT_NORMALS": "1",
	"MD_LENS_INK": "0", "MD_LENS_COATING": "1", "MD_MSAA": "8",
	"MD_SHADOW_FILTER": "high", "MD_SHADOW_SIZE": "8192",
	"MD_CAMERA_RELEASE": "slow", "MD_OCCLUSION_FOV": "1", "MD_LABEL_LOD": "1",
	"MD_SERVICE_NOTE": "1", "MD_CART_FRAME": "1", "MD_ROBOT_HATCH": "0.03"
}

static func value(key: String) -> String:
	if OS.has_environment(key):return OS.get_environment(key)
	if OS.get_environment("MD_MODE")=="baseline" or OS.get_environment("MD_PROFILE")=="legacy":return ""
	return str(DEFAULTS.get(key,""))

static func resolved() -> Dictionary:
	var settings: Dictionary = {}
	for key in DEFAULTS:settings[key]=value(key)
	return settings
