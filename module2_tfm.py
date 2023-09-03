from plugin import Plugin
from pyglui import ui

from glfw import *


from pyglui.ui import get_opensans_font_path
from pyglui.pyfontstash import fontstash
from pyglui.cygl.utils import draw_rounded_rect,RGBA


import zmq

import numpy
from threading import Thread
# import time
import msgpack
import math
import base64

from numpy import asarray
# from time import time,sleep
import cv2


import time

from PIL import Image

from primesense import openni2#, nite2
from primesense import _openni2 as c_api


CUSTOM_TOPIC = "custom_topic"

class Module2_TFM(Plugin):
    uniqueness = "by_class" # only one plugin instance can exist at a time
    icon_chr = chr(0xe061)
    icon_font='pupil_icons' 

    def __init__(self, g_pool,frame=None,run=False):
        super().__init__(g_pool)
        self.order=1
        self.glfont = fontstash.Context()
        self.glfont.add_font('opensans', get_opensans_font_path())
        self.glfont.set_size(10)
        self.frame=frame
        self.run = run
        self.count=0
        self.message=None
        self.message2=None
        self.x=None
        self.y=None
        self.z=None
        
        host = "127.0.0.1"
        port = "555"
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)

        self.socket.bind("tcp://{}:{}".format(host, port))
        

    def recent_events(self,events):
        if self.run == True:
            # count=0
            
            if 'frame' in events:
                frame = events['frame']
                img = frame.img
                self.frame=img
                
                height,width,_=self.frame.shape
                numpydata =numpy.ascontiguousarray(img)
                base64_img = base64.b64encode(numpydata)
                
                self.count=self.count+1
            if 'fixations' in events:
                # if 'gaze' in events:
                fixations = events['fixations']
                for f in fixations:
                    self.x=f["gaze_point_3d"][0]
                    self.y=f["gaze_point_3d"][1]
                    self.z=f["gaze_point_3d"][2]
                
                
            else:    
                self.x=None
                self.y=None
                self.z=None

            custom_datum = {
            "topic": CUSTOM_TOPIC,"frame":base64_img,"height":height,"width":width,"x":self.x,"y":self.y,"z":self.z,}

            
            self.socket.send(msgpack.dumps(custom_datum,use_bin_type=True))
                   
           

    def gl_display(self):
        if self.run==True:

            if self.z==None:
                self.glfont.set_color_float((0.8,1.0,0.6, 1.0)) # set color to the text           
                self.glfont.draw_text(30,200,"Processing")
            else:
                self.glfont.set_color_float((0.8,1.0,0.6, 1.0)) # set color to the text           
                self.glfont.draw_text(30,200,"Position in x: "+str(self.x))
                
                self.glfont.set_color_float((0.8,1.0,0.6, 1.0)) # set color to the text           
                self.glfont.draw_text(30,210,"Position in y: "+str(self.y))

                self.glfont.set_color_float((0.8,1.0,0.6, 1.0)) # set color to the text           
                self.glfont.draw_text(30,220,"Position in z: "+str(self.z))

                

    def init_ui(self):
        self.add_menu()
        self.menu.label = 'MODULE 2'
        self.menu.append(ui.Switch('run', self, label='Run Model 2'))

    def deinit_ui(self):
        self.remove_menu()