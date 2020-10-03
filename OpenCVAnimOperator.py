#todo sensible selection gui

import bpy
from numpy import *
import cv2 as cv
import time

#https://github.com/jkirsons/FacialMotionCapture_v2
#license- ??????

# Download trained model (lbfmodel.yaml)
# https://github.com/kurnianggoro/GSOC2017/tree/master/data
# into //./opencv/ where //. is the directory of this blendfile
cvpath= bpy.path.abspath('//')+'opencv/GSOC2017-master/data'
face_detect_path =    cvpath+"/haarcascade_frontalface_default.xml"
landmark_model_path = cvpath+"/lbfmodel.yaml"
#uncertain but since this is included with cv it should be bsd license

# Install prerequisites:

# Linux: (may vary between distro's and installation methods)
# This is for manjaro with Blender installed from the package manager
# python3 -m ensurepip
# python3 -m pip install --upgrade pip --user
# python3 -m pip install opencv-contrib-python numpy --user

# MacOS
# open the Terminal
# cd /Applications/Blender.app/Contents/Resources/2.81/python/bin
# ./python3.7m -m ensurepip
# ./python3.7m -m pip install --upgrade pip --user
# ./python3.7m -m pip install opencv-contrib-python numpy --user

# Windows:
# Open Command Prompt as Administrator
# cd "%PROGRAMFILES%\Blender Foundation\Blender 2.90\2.90\python\bin"
# python -m pip install --upgrade pip
# python -m pip install opencv-contrib-python numpy

#test ensure cv is using gpu
#todo support recorded video and not just webcam
#todo async thread

class OpenCVAnimOperator(bpy.types.Operator):
	bl_idname = "wm.opencv_operator"
	bl_label = "OpenCV Face Animation Operator"
		
	# Load models
	cas = cv.CascadeClassifier(face_detect_path)
	fm = cv.face.createFacemarkLBF()
	fm.loadModel(landmark_model_path)
	#fixme do these need dtors?
	
	cap = None #cv capture object
	_timer = None
	
	# Webcam resolution
	#todo autoconfig https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
	width = 480
	height = 640
	
	# 3D model points. 
	#todo calibrate
	#todo auto extract from default pose
	model_points = array([
								(   0.,    0.,    0.),# Nose tip
								(   0., -330.,  -65.),# Chin
								(-225.,  170., -135.),# Left eye left corner
								( 225.,  170., -135.),# Right eye right corne
								(-150., -150., -125.),# Left Mouth corner
								( 150., -150., -125.) # Right mouth corner
							], dtype = float32)
	camera_matrix = array(
							[[width, 0, width/2.],
							[0, width, height/2.],
							[0, 0, 1.2]], dtype = float32
							)#ww approx atan fov, model seems to not care much, it could probably be ortho scronched
							
	#todo raze selfness
	#todo kalman
	def smooth_value(self, name, length, value):
		if not hasattr(self, 'smooth'):
			self.smooth = {}
		if not name in self.smooth:
			self.smooth[name] = array([value])
		else:
			self.smooth[name] = insert(arr=self.smooth[name], obj=0, values=value)
			if self.smooth[name].size > length:
				self.smooth[name] = delete(self.smooth[name], self.smooth[name].size-1, 0)
		return average(self.smooth[name])
	# Keeps min and max values, then returns the value in a range 0 - 1
	def get_range(self, name, value):
		if not hasattr(self, 'range'):
			self.range = {}
		if not name in self.range:
			self.range[name] = array([value, value])
		else:
			self.range[name] = array([min(value, self.range[name][0]), max(value, self.range[name][1])] )
		val_range = self.range[name][1] - self.range[name][0]
		if val_range != 0:
			return (value - self.range[name][0]) / val_range
		else:
			return 0.0
	
	def modal(self, context, event):
		if event.type in {'RIGHTMOUSE', 'ESC'}:
			#todo keybindable button instead
			self.cancel(context)
			return {'CANCELLED'}
		if event.type == 'TIMER': #### MAIN LOOP ####
			err,image = self.cap.read()
			assert(not err)
			assert(self.cap.isOpened())
			#gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
			#image= cv.resize(image, dsize=(int(self.width/2),int(self.height/2)), interpolation=cv.INTER_NEAREST)
			#gray = cv.equalizeHist(gray) histonorming is chonky op, if bad lighting then fuck the user, or maybe an option so user may chose to possibly be fucked for perf gain
			#model is trained on rgb data, feeding it greyscale doesnt affect noticeably
			
			print('====')
			print(err)
			print('____')
			print(image)
			
			# find faces
			faces = self.cas.detectMultiScale(
				image, 
				flags=cv.CASCADE_SCALE_IMAGE,
				minSize=(int(self.width/8), int(self.width/8)))#what does denominator signify?
		 	#is classifier necessary? significant perf cost;
			#test no classifier, presume a single face is present, directly ram image into facemark
			
			#find biggest face, and only keep it
			if type(faces) is ndarray and faces.size > 0: 
				biggestFace = zeros(shape=(1,4))
				for face in faces:
					if face[2] > biggestFace[0][2]:
						biggestFace[0] = face
				for landmark in self.fm.fit(image, faces=biggestFace):
					shape = mark[0]
					#2D image points. If you change the image, you need to change vector
					image_points = array([shape[30],# Nose tip - 31
										  shape[ 8],# Chin - 9
										  shape[36],# Left eye left corner - 37
									 	  shape[45],# Right eye right corne - 46
									 	  shape[48],# Left Mouth corner - 49
										  shape[54]	# Right mouth corner - 55
											], dtype = float32)
					# Assume no lens distortion
					dist_coeffs = zeros((4,1))
				 
					# determine head rotation
					if hasattr(self, 'rotation_vector'):
						(success, self.rotation_vector, self.translation_vector) = cv.solvePnP(
							self.model_points, image_points, self.camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE, 
							rvec=self.rotation_vector, tvec=self.translation_vector, 
							useExtrinsicGuess=True)
					else:
						(success, self.rotation_vector, self.translation_vector) = cv.solvePnP(
							self.model_points, image_points, self.camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE, 
							useExtrinsicGuess=False)
				 
					if not hasattr(self, 'first_angle'):
						self.first_angle = copy(self.rotation_vector)
					
					# set bone rotation/positions
					bones = bpy.context.selected_objects[0].pose.bones
					
					# head rotation 
					bones["head"].rotation_euler[0] = self.smooth_value("h_x", 3,  (self.rotation_vector[0] - self.first_angle[0]))# Up/Down
					bones["head"].rotation_euler[2] = self.smooth_value("h_y", 3, -(self.rotation_vector[1] - self.first_angle[1]))# Rotate
					bones["head"].rotation_euler[1] = self.smooth_value("h_z", 3,  (self.rotation_vector[2] - self.first_angle[2]))# Left/Right
					
					bones["head"].keyframe_insert(data_path="rotation_euler", index=-1)
					
					# mouth position
					#todo unhack
					bones["mouth_ctrl"].location[1] = 6*self.smooth_value("m_h", 2, -self.get_range("mouth_height", linalg.norm(shape[62] - shape[66])) )
					#use scale instead? how does constrain that do???
					# bones["mouth_ctrl"].location[0] = 800.*self.smooth_value("m_w", 1, (self.get_range("mouth_width",  linalg.norm(shape[54] - shape[48])) - 0.5) * -0.04)
					bones["mouth_ctrl"].keyframe_insert(data_path="location", index=-1)
					#eyebrows
					#bones["brow_ctrl_L"].location[2] = self.smooth_value("b_l", 3, (self.get_range("brow_left", linalg.norm(shape[19] - shape[27])) -0.5) * 0.04)
					#bones["brow_ctrl_R"].location[2] = self.smooth_value("b_r", 3, (self.get_range("brow_right", linalg.norm(shape[24] - shape[27])) -0.5) * 0.04)
					#bones["brow_ctrl_L"].keyframe_insert(data_path="location", index=2)
					#bones["brow_ctrl_R"].keyframe_insert(data_path="location", index=2)					
					# eyelids
					#l_open = self.smooth_value("e_l", 2, self.get_range("l_open", -linalg.norm(shape[48] - shape[44]))  )
					#r_open = self.smooth_value("e_r", 2, self.get_range("r_open", -linalg.norm(shape[41] - shape[39]))  )
					#eyes_open = (l_open + r_open) / 2.0 # looks weird if both eyes aren't the same...
					#bones["eyelid_up_ctrl_R"].location[2] =   -eyes_open * 0.025 + 0.005
					#bones["eyelid_low_ctrl_R"].location[2] =  eyes_open * 0.025 - 0.005
					#bones["eyelid_up_ctrl_L"].location[2] =   -eyes_open * 0.025 + 0.005
					#bones["eyelid_low_ctrl_L"].location[2] =  eyes_open * 0.025 - 0.005
					#bones["eyelid_up_ctrl_R"].keyframe_insert(data_path="location", index=2)
					#bones["eyelid_low_ctrl_R"].keyframe_insert(data_path="location", index=2)
					#bones["eyelid_up_ctrl_L"].keyframe_insert(data_path="location", index=2)
					#bones["eyelid_low_ctrl_L"].keyframe_insert(data_path="location", index=2)
					
					# draw face markers
					for (x, y) in shape:
						cv.circle(image, (x, y), 2, (0, 255, 255), -1)
			
			# draw detected face
			for (x,y,w,h) in faces:
				cv.rectangle(image,(x,y),(x+w,y+h),(255,0,0),1)
			
			# Show camera image in a window					 
			cv.imshow("Output", image)
			#cv.waitKey(1) #DO NOT BLOCK REEEEEEE

		return {'PASS_THROUGH'}
			
	def stop_playback(self, scene):
		bpy.ops.screen.animation_cancel(restore_frame=False)

	def webcam_init(self):		
		#fixme gui report webcam/video init failure
		#test does cv properly handle if camera was activated after cv init?
		if self.cap:
			self.cap.release()
		self.cap = cv.VideoCapture(0)
		assert(self.cap.isOpened())
		self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
		self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)
		self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

	def execute(self, context):
		bpy.app.handlers.frame_change_pre.append(self.stop_playback)
		self.webcam_init()
		wm= context.window_manager
		wm.modal_handler_add(self)
		self._timer= wm.event_timer_add(1./30., window=context.window)
		#todo variable timestep mechanism
		return {'RUNNING_MODAL'}
	
	def cancel(self, context):
		wm = context.window_manager
		if self._timer:
			wm.event_timer_remove(self._timer)
			self._timer = None
		if self.cap:
			self.cap.release()
			self.cap = None
		cv.destroyAllWindows()

def register():   bpy.utils.register_class(  OpenCVAnimOperator)
def unregister(): bpy.utils.unregister_class(OpenCVAnimOperator)

if __name__ == "__main__":
	register()
	# test call
	bpy.ops.wm.opencv_operator()


