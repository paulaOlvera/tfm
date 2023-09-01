from plugin import Plugin
from pyglui import ui

from glfw import *

from pyglui.ui import get_opensans_font_path
from pyglui.pyfontstash import fontstash
from pyglui.cygl.utils import draw_polyline, draw_points, draw_x, RGBA

import zmq
# from msgpack import loads
import torch
from torch import hub # Hub contains other models like FasterRCNNmodel
import numpy

# import time
import msgpack
import math
import base64

from numpy import asarray
from time import time,sleep
import cv2


model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  
device = 'cpu'
model.to(device)

CUSTOM_TOPIC = "custom_topic"



class Module3_TFM(Plugin):
    uniqueness = "by_class" # only one plugin instance can exist at a time
    

    def __init__(self, g_pool,queue_size=1,frame=None,run=False,min_duration=300):
        super().__init__(g_pool)
        self.order=1
        self.glfont = fontstash.Context()
        self.glfont.add_font('opensans', get_opensans_font_path())
        self.glfont.set_size(20)
        self.frame=frame
        self.pupil_display_list = []
        self.queue_size = queue_size
        self.run = run
        self.x=None
        self.y=None
        self.num_labels=len(model.names)
        self.conf_thres=0.85
        self.fixation_norm_pos = None
        self.results=None
        self.index=0
       
        self.prev_timestamp=0
        self.prev_index=0
        self.last_timestamp=0
        self.time_threshold=1.0
        self.fixation_time=0
        
        self.box_image=None

        self.x_gaze=None
        self.y_gaze=None
        self.z_gaze=None

    def recent_events(self,events):
        if self.run == True:
            if 'frame' in events:
                frame = events['frame']
                img = frame.img
                
                # img2=cv2.resize(img,(640,640))
                self.frame=img
                height,width,_=self.frame.shape
                numpydata =numpy.ascontiguousarray(self.frame)
                base64_img = base64.b64encode(numpydata)
                self.x_gaze=None
                self.y_gaze=None
                self.z_gaze=None
            if 'gaze' in events:
                gaze = events['gaze']
                for g in gaze:
                    self.x_gaze=g["gaze_point_3d"][0]
                    self.y_gaze=g["gaze_point_3d"][1]
                    self.z_gaze=g["gaze_point_3d"][2]
            
            if 'fixations' in events:
                fixations = events['fixations']

                for f in fixations:
                    self.index=f["id"]
                    self.timestamp=f["timestamp"]
         
                    if self.timestamp-self.last_timestamp>self.time_threshold and self.prev_index==self.index:
                        self.fixation_time=self.timestamp-self.last_timestamp
                    
                        if len(fixations)!=0:
                            
                            detections = model(img[..., ::-1])
                            self.results=detections.xyxy[0].cpu().detach().numpy()

                            self.x=int(f["norm_pos"][0]*frame.width)
                            self.y=int((1.0-f["norm_pos"][1])*frame.height)

                    if self.prev_index!=self.index:
                       self.last_timestamp=f["timestamp"]
                       self.prev_index=self.index
                       self.fixation_time=0



    def gl_display(self):
        if self.run==True:
            if self.fixation_time==0 or self.fixation_time<self.time_threshold:
                self.glfont.set_color_float((0.8,1.0,0.6, 1.0)) # set color to the text           
                self.glfont.draw_text(20,80,'Not enough fixation') # set the labels 
     
            else:
                height, width, _= self.frame.shape

            
                final_results=self.results
                distance_list=[]
                xmin=final_results[:,0]
                ymin=final_results[:,1]
                xmax=final_results[:,2]
                ymax=final_results[:,3]
                confidence=final_results[:,4]
                classnum=final_results[:,5]
                num_box=0
                
                candidates_list=[]
                for i in range(len(xmin)):
                    
                    if confidence[i]>self.conf_thres:
                        num_box+=1
                        bottom_left=[xmin[i],ymax[i]]
                        bottom_right=[xmax[i],ymax[i]]
                        top_left=[xmin[i],ymin[i]]
                        top_right=[xmax[i],ymin[i]]
                        vertice=[top_left,top_right,bottom_right,bottom_left,top_left]


                        center=[(xmax[i]+xmin[i])/2.0,(ymax[i]+ymin[i])/2.0]
                        point1=[center]
                        draw_points(point1,size=10.0,color=RGBA(1.0,1.0,1.0,1.0),sharpness=1.0)
                        draw_polyline(vertice,thickness=3,color=RGBA(classnum[i]/self.num_labels,classnum[i]/self.num_labels,classnum[i]/self.num_labels, 1.0))
                        self.glfont.set_color_float((classnum[i]/self.num_labels,classnum[i]/self.num_labels,classnum[i]/self.num_labels, 1.0))
                        self.glfont.draw_text(xmin[i],ymin[i],model.names[classnum[i].astype(int)] + " : " + confidence[i].astype(str)) # set the labels

        
                        if xmin[i]<self.x<xmax[i]:
                            if ymin[i]<self.y<ymax[i]:
                                dist=math.hypot(center[0]-self.x,center[1]-self.y)
                                candidates_list.append([dist,[center[0],center[1]],(xmax[i]-xmin[i])/3.0,(ymax[i]-xmin[i])/3.0,model.names[classnum[i].astype(int)],xmin[i],xmax[i],ymin[i],ymax[i]])           

                if candidates_list:
                    candidates_list=sorted(candidates_list)
                    draw_x([candidates_list[0][1]],candidates_list[0][2],candidates_list[0][3], thickness=5, color=RGBA(0.0,0.0,0.0,1.0))
                    self.label=candidates_list[0][4]
                    self.glfont.set_color_float((0.8,1.0,0.6, 1.0))
                    self.glfont.draw_text(20,120,'Match with: '+ candidates_list[0][4])
                   

                # number of boxes detected inside our accepted confidence
                if num_box==0:
                        self.glfont.set_color_float((1.0,1.0,0.1, 1.0)) 
                        self.glfont.draw_text(20,80,'No object detected') 
                        
                elif num_box==1:
                        self.glfont.set_color_float((0.8,1.0,0.6, 1.0))
                        self.glfont.draw_text(20,80,'1 object detected') 
                        
                elif num_box==2:
                        self.glfont.set_color_float((0.0,0.0,1.0, 1.0))
                        self.glfont.draw_text(20,80,'2 objects detected') 
    
                elif num_box==3:
                        self.glfont.set_color_float((0.5,0.0,0.0, 1.0))
                        self.glfont.draw_text(20,80,'3 objects detected') 

                else:
                        self.glfont.set_color_float((0.5,0.25,0.25, 1.0))
                        self.glfont.draw_text(20,80,'More than 3 objects') 
    
            if self.x_gaze==None:

                self.glfont.set_color_float((0.8,1.0,0.6, 1.0)) # set color to the text           
                self.glfont.draw_text(30,200,"Processing")
            else:
                self.glfont.set_color_float((0.8,1.0,0.6, 1.0)) # set color to the text           
                self.glfont.draw_text(30,200,"Position in x: "+str(round(self.x_gaze,2)))
                
                self.glfont.set_color_float((0.8,1.0,0.6, 1.0)) # set color to the text           
                self.glfont.draw_text(30,210,"Position in y: "+str(round(self.y_gaze,2)))

                self.glfont.set_color_float((0.8,1.0,0.6, 1.0)) # set color to the text           
                self.glfont.draw_text(30,220,"Position in z: "+str(round(self.z_gaze,2)))

    def init_ui(self):
        self.add_menu()
        self.menu.label = 'MODULE 3'
        self.menu.append(ui.Switch('run', self, label='Run module3'))

    def deinit_ui(self):
        self.remove_menu()