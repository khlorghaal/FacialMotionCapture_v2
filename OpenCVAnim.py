import bpy

class OpenCVPanel(bpy.types.WorkSpaceTool):
    """Creates a Panel in the Object properties window"""
    bl_label = "OpenCV Facemark"
    bl_space_type = 'VIEW_3D'
    bl_context_mode='OBJECT'
    bl_idname = "ui_plus.opencv"
    bl_options = {'REGISTER'}
    bl_icon = "ops.generic.bone_data"
        
    def draw_settings(context, layout, tool):
        row = layout.row()
        op = row.operator("wm.opencv_operator", text="Capture", icon="OUTLINER_OB_CAMERA")
		#todo gui 
			#label camera readyness
			#button halt capture
			#button destruct
def register():   bpy.utils.register_tool(  OpenCVPanel, separator=True, group=True)
def unregister(): bpy.utils.unregister_tool(OpenCVPanel)

if __name__ == "__main__":
    register()
