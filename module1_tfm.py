from plugin import Plugin
from pyglui import ui

from pyglui.ui import get_opensans_font_path
from pyglui.pyfontstash import fontstash

import socket
import sys

class Module1_TFM(Plugin):
    uniqueness = "by_class" # only one plugin instance can exist at a time
    icon_chr = chr(0xe061)
    icon_font='pupil_icons' 

    def __init__(self, g_pool,frame=None,run=False):
        super().__init__(g_pool)
        self.order=1
        self.glfont = fontstash.Context()
        self.glfont.add_font('opensans', get_opensans_font_path())
        self.glfont.set_size(30)
        self.frame=frame
        self.run = run
        self.timestamp=0
        self.prev_timestamp=0
        self.blink=0
        self.type_blink="offset"
        self.mode=0
        self.x_gaze=0
        self.y_gaze=0
        self.z_gaze=0
        # Creates a socket instance
        self.host = "10.4.173.52"
        self.port = 49999
 
        self.sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.letter_x='N'.encode()
        self.letter_y='N'.encode()
        self.confidence=0
        
        self.prev_timestamp = 0 
        self.prev_type_blink = "offset"

    def recent_events(self,events):
        if self.run == True:
            if 'blinks' in events:
                
                blinks= events["blinks"]
                for b in blinks:

                    self.timestamp=b["timestamp"]

                    # the blink detector uses the confidence of the pupil detection. There are two types of blink: onset and offset.
                    #  "onset" is when the signal starts decreasing,  the start of blinking (eye open -> eye close).
                    #  "offset" is when the signal starts increasing, the end of blinking (eye close -> eye open)

                    self.type_blink=b["type"]
                
                    
                # When the blink goes from offset to onset means there is a blink. 
                if self.type_blink=="onset" and self.prev_type_blink=="offset":
                    

                    # To detect voluntary blinks, two blinks are needed in less than 0.5 seconds.
                    if 0 <self.timestamp - self.prev_timestamp <0.5:
                        self.blink = 1
                        print("Voluntary blink")
                        sys.stdout.flush()
                        self.timestamp = 0 
                    else:
                        self.blink = 0
                    self.prev_timestamp = self.timestamp

                self.prev_type_blink = self.type_blink
             
                # change modes 
                if self.blink==1:
                    self.mode=self.mode+1
                    self.blink=0
                    if self.mode==4:
                        self.mode=0
                    self.send_mode=self.mode
                else:
                    self.send_mode=None
                
            if 'gaze' in events:
                gaze = events['gaze']
                for g in gaze:
                    # gaze point in mm with respect front camera of pupil labs
                    self.x_gaze=g["gaze_point_3d"][0]
                    self.y_gaze=g["gaze_point_3d"][1]
                    self.z_gaze=g["gaze_point_3d"][2]
                
                    self.confidence=g["confidence"]
                    
                    if self.confidence>0.95:
                        if self.x_gaze:
                            self.prev_x_gaze=self.letter_x
                            if self.x_gaze<-40:
                                self.letter_x='L'
                                self.letter_x=self.letter_x.encode()  
                                
                            if self.x_gaze>30:
                                self.letter_x='R'
                                self.letter_x=self.letter_x.encode()

                            if -40<=self.x_gaze<=30:
                                self.letter_x='N'
                                self.letter_x=self.letter_x.encode()

                        if self.y_gaze:
                            self.prev_y_gaze=self.letter_y
                            if self.y_gaze>50:
                                self.letter_y='B'
                                self.letter_y=self.letter_y.encode()
                            
                            if self.y_gaze<-25:
                                self.letter_y='F'
                                self.letter_y=self.letter_y.encode()
                            
                            if -25<=self.y_gaze<=50:
                                self.letter_y='N'
                                self.letter_y=self.letter_y.encode()
                            self.prev_y_gaze=self.letter_y
                        
                    # if self.send_mode!=None:
                    #     print(self.send_mode)
                    # print(self.mode)
                    # sys.stdout.flush()
                        # self.sock.sendto(self.mode,(self.host,self.port))
                    
                    # self.sock.sendto(self.letter_x+self.letter_y+str(self.mode).encode(),(self.host,self.port))

    def gl_display(self):
        if self.run==True:

            if self.z_gaze==None:
                self.glfont.set_color_float((0.8,1.0,0.6, 1.0)) # set color to the text           
                self.glfont.draw_text(30,200,"Processing")
            else:

                self.glfont.set_color_float((0.0,0.0,1.0, 1.0)) # set color to the text           
                self.glfont.draw_text(30,560,"Mode:"+str(self.mode))

                self.glfont.set_color_float((1.0,0.0,0.0, 1.0)) # set color to the text           
                self.glfont.draw_text(30,600,"x :"+str(round(self.x_gaze,2))+"    letter_x: "+str(self.letter_x.decode()))
                
                self.glfont.set_color_float((1.0,0.0,0.0, 1.0)) # set color to the text           
                self.glfont.draw_text(30,640,"y :"+str(round(self.y_gaze,2))+"    letter_y: "+str(self.letter_y.decode()))
                
    def init_ui(self):
        self.add_menu()
        self.menu.label = 'MODULE 1'
        self.menu.append(ui.Switch('run', self, label='Run module 1'))

    def deinit_ui(self):
        self.remove_menu()



