from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
import cv2
import shutil
import natsort
import os
import subprocess
import seaborn as sns
sns.set(color_codes=True)
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import numpy as np
import qimage2ndarray
import numpy as np
from statistics import mean 
import time
import matplotlib
from matplotlib import pyplot as plt
import statistics
import math
from math import pi as PI
from sklearn.linear_model import LinearRegression
import statistics
matplotlib.use('agg')
import sys
import pyqtgraph as pg
from pyqtgraph import PlotWidget, plot
from random import randint
from numpy import savetxt
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as geek

#np.set_printoptions(threshold=sys.maxsize)


# Subclass QMainWindow to customise your application's main window
class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle("Fish Analyzer")

       
        self.start_app()


        
    def start_app(self):

        self.setFixedSize(660,400)
        self.move(350, 50)
        self.start_x = 0
        self.start_y = 0
        self.end_x = 640
        self.end_y = 360
        self.step = 0
        self.stop = 0
        self.fish_size = 1
        self.head_choosen = 0
        self.abort_counting = 0
        self.init_frame=0
        self.final_frame = ''

        self.name_pdf = ''

        self.back_sub_type = 'numbers'

        self.thr = 1
        self.background = 61

        self.choice_heatmap_lines = 6
        self.choice_heatmap_collumns = 9
        self.choice_histogram_collumns = 10


        self.first_array_interaction = 0
        self.chunk_anterior = 0 

        self.csv_saved = 0

        self.mean_points = 0.3

        self.file_csv = ''
        self.file_txt = ''

        self.min_head_position = 0.3
        self.hub_position = 0.6
        self.max_tail_position = 640

        self.pan_number = 50
        
        self.fish_pre_model = np.array([1])
        self.fish_model = np.array([1])
        self.advance_block = 0
        self.angle = 0
        self.x = []
        self.y = []
        #self.x_count = 0

        self.values_color = 0
        self.size_minimum = 0
        self.size_maximum = 194400
        
        self.croped_background = 0
        self.croped_study_area = 0            
        #self.max_tail_position = 0
        
        self.file = ''
        self.number = -1
        self.abort = 0






       






        self.layout = QVBoxLayout()

        self.label = QLabel(self)        
        self.pixmap = QPixmap('happyfish2.png')
        self.pixmap = self.pixmap.scaledToWidth(640)
        self.label.setPixmap(self.pixmap)
        self.layout.addWidget(self.label)

        self.img_back = ''         
        

        self.initial_message1 = QLabel('', self)
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.initial_message1.setSizePolicy(sizePolicy)        
        #self.initial_message1.setToolTip('Leave blank to extract all frames')
        self.initial_message1.hide()

        self.initial_message2 = QLabel('', self)
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.initial_message2.setSizePolicy(sizePolicy)        
        self.initial_message2.setToolTip('These are the video metadata')
        self.initial_message2.hide()      
        

        self.slider_time = QSlider()
        self.slider_time.setOrientation(Qt.Horizontal)              
        self.slider_time.setTickInterval(1)
        self.slider_time.setMinimum(10)        
        self.slider_time.valueChanged.connect(self.function_slider_time)
        self.slider_time.hide()     

        self.initial_open = QLineEdit(self)        
        self.initial_open.textChanged.connect(self.function_slider_time_change) 
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.initial_open.setSizePolicy(sizePolicy)               
        self.initial_open.setValidator(QIntValidator()) #Settings can only input data of type int
        self.initial_open.setValidator(QDoubleValidator()) #Settings can only input data of type double
        self.initial_open.hide()

        
        # choose of the fish is oriented to left or to the right
        self.orientation_message = QLabel('Select the fish orientation:', self)
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.orientation_message.setSizePolicy(sizePolicy)        
        self.orientation_message.setToolTip('Choose if the fish is headed to left or right')
        self.orientation_message.hide()

        self.left_checked = QCheckBox("Head to left")
        self.left_checked.setToolTip('Select the fish orientation')
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.left_checked.setSizePolicy(sizePolicy)
        self.left_checked.hide()

        self.right_checked = QCheckBox("Head to right")
        self.right_checked.setToolTip('Select the fish orientation')
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.right_checked.setSizePolicy(sizePolicy) 
        self.right_checked.hide()      
        #self.right_checked.hide()
        self.left_checked.stateChanged.connect(lambda:self.head_choose(self.left_checked))      
        self.right_checked.stateChanged.connect(lambda:self.head_choose(self.right_checked))        
 
       
        self.layout_h_head_orientation = QHBoxLayout()
        self.layout_h_head_orientation.addWidget(self.orientation_message)
        self.layout_h_head_orientation.addWidget(self.left_checked)         
        self.layout_h_head_orientation.addWidget(self.right_checked)


        

        # choose the method to extract backgorund

        self.extract_back_message = QLabel('Background subtraction method:', self)
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.extract_back_message.setSizePolicy(sizePolicy)        
        self.extract_back_message.setToolTip('Choose the method of background subtraction')       
        self.extract_back_message.hide()

       
        self.image_back = QCheckBox("Image Subtraction")
        self.image_back.setToolTip('Select an image, without the fish, to make subtraction')
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.image_back.setSizePolicy(sizePolicy) 
        self.image_back.hide()      
        
       
        self.image_back.stateChanged.connect(self.checkBoxChangedAction)


        self.button_send_background = QPushButton('Upload', self)
        self.button_send_background.setToolTip('Upload the background picture')        
        self.button_send_background.clicked.connect(self.on_click_Background_upload)
        self.button_send_background.hide() 


        self.layout_back_subtraction_type = QHBoxLayout()
        self.layout_back_subtraction_type.addWidget(self.extract_back_message)                 
        self.layout_back_subtraction_type.addWidget(self.image_back)
        self.layout_back_subtraction_type.addWidget(self.button_send_background)



        self.layout_h0 = QHBoxLayout()
        self.layout_h0.addWidget(self.initial_message1)
        self.layout_h0.addWidget(self.initial_open)
        self.layout_h0.addWidget(self.initial_message2)

        self.layout.addLayout(self.layout_h0)
        self.layout.addWidget(self.slider_time)
        self.layout.addLayout(self.layout_h_head_orientation)

        self.layout.addLayout(self.layout_back_subtraction_type)
        
        self.label_fish_id = QLabel('Fish ID', self)
        self.label_fish_id.hide()

        self.edit_fish_id = QLineEdit(self)        
        self.edit_fish_id.hide()


        self.label_fish_specie = QLabel('Fish Specie', self)
        self.label_fish_specie.hide()

        self.edit_fish_specie = QLineEdit(self)        
        self.edit_fish_specie.hide()





        self.button_save_back_pic = QPushButton('Save Picture', self)
        self.button_save_back_pic.setToolTip('Save this picture for posterior use as background subtraction')        
        self.button_save_back_pic.clicked.connect(self.on_click_save_back_pic)
        self.button_save_back_pic.hide() 



        self.layout_fish_identification = QHBoxLayout()
        self.layout_fish_identification.addWidget(self.label_fish_id)
        self.layout_fish_identification.addWidget(self.edit_fish_id)
        self.layout_fish_identification.addWidget(self.label_fish_specie)
        self.layout_fish_identification.addWidget(self.edit_fish_specie)
        self.layout_fish_identification.addWidget(self.button_save_back_pic)
        self.layout.addLayout(self.layout_fish_identification)




        self.button_start_framming = QPushButton('Extract Frames and Advance', self)                 
        self.button_start_framming.clicked.connect(self.function_start_framming)            
        self.button_start_framming.hide()
        self.layout.addWidget(self.button_start_framming)
        self.button_start_framming.hide() 


        self.Hinitial = QHBoxLayout()

        self.button_open = QPushButton('New MP4 analyses', self)                 
        self.button_open.clicked.connect(self.getfile)            
        self.button_open.show()
        self.Hinitial.addWidget(self.button_open)


        self.button_open_csv = QPushButton('Generate Reports', self)                 
        self.button_open_csv.clicked.connect(self.build_report_step2)            
        self.button_open_csv.show()
        self.Hinitial.addWidget(self.button_open_csv) 

        
        



        self.layout.addLayout(self.Hinitial)                

        
        self.processed_image = 0
        self.process_image = QLabel('processed images: ' + str(self.processed_image), self)
        self.layout.addWidget(self.process_image)        
        self.process_image.hide()
        
        self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)        
        self.origin = QPoint()


        # add all layoyt instance to a widget and set all them as central widget
        self.widget_central = QWidget()
        self.widget_central.setLayout(self.layout)
        self.setCentralWidget(self.widget_central)
       
        self.button_abort = QPushButton('Abort', self)       
        self.button_abort.clicked.connect(self.function_abort)
        self.layout.addWidget(self.button_abort)
        self.button_abort.hide()



    def on_click_save_back_pic(self):
        self.name_jpg_0 = QFileDialog.getSaveFileName(self, 'Where to save the background picture',
                'c:\\',"JPG file (*.jpg)")
        self.name_jpg = self.name_jpg_0[0]
        if self.name_jpg != '': 
            #print(self.name_jpg)
            #print('#')
            #print(self.file[:-4] + '_' + str(self.time_now) + '/' + str(self.number) +'.jpg')     
            shutil.copy(self.file[:-4] + '_' + str(self.time_now) + '/' + str(self.number) +'.jpg', self.name_jpg)
            QMessageBox.about(self, "Alert", "You saved the picture for future use as background subtraction.\nThe patch is: " + self.name_jpg)
        else:
            QMessageBox.about(self, "Alert", "You did not save any image")

       
    def on_click_Background_upload(self):
        fname_type = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\',"Image files (*.jpg)") 
        self.img_back = fname_type[0]
        if self.img_back != '':
            image = cv2.imread(self.img_back)
            cv2.imshow('Background Image - Close this window if it seems ok', image)
            self.back_sub_type = 'image'
            self.set_mask()
        else:
            QMessageBox.about(self, "Alert", "You did not choose any background image.\nYou need to choose one before take any action, or just uncheck the \"Image Subtraction\" box")




    def checkBoxChangedAction(self, state):
        #print('clicou')
        if (Qt.Checked == state):
            self.button_send_background.show()
            self.slider_background.show()
            self.slider.hide()
            self.label_slider_title.hide()
            self.label_slider.hide()
            fname_type = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\',"Image files (*.jpg)") 
            self.img_back = fname_type[0]

            if self.img_back != '':
                image = cv2.imread(self.img_back)
                cv2.imshow('Background Image - Close this window if it seems ok', image)
                self.back_sub_type = 'image'
                self.set_mask()

            else:
                QMessageBox.about(self, "Alert", "You did not choose any background image.\nYou need to choose one before take any action, or just uncheck the \"Image Subtraction\" box")
                
                #self.repaint()

                '''self.image_back.setChecked(False)
                self.image_back.hide()
                self.image_back.show()                                
                print(self.image_back.checkState()) '''

            self.slider_background.show()
            self.slider.hide()
            self.label_slider_title.hide()
            self.label_slider.hide()
            self.back_sub_type = 'image'


        else:
            self.button_send_background.hide()
            self.slider_background.hide()
            self.slider.show()
            self.label_slider_title.show()
            self.label_slider.show()
            self.back_sub_type = 'values'

                            
    

    @pyqtSlot()
    def function_choose_head(self):            
            self.min_head_position = self.slider_head.value()/100
            #print(self.min_head_position)
            self.set_mask()
            self.angle_calculation_direct()   


    @pyqtSlot()
    def function_choose_hub(self):
            self.hub_position = self.slider_hub.value()/100
            #print(self.hub_position)
            self.set_mask()
            self.angle_calculation_direct()


    @pyqtSlot()
    def function_choose_tail(self):
            self.max_tail_position = self.slider_tail.value()
            #print(self.max_tail_position)
            self.set_mask()
            self.angle_calculation_direct()
   


    
    def angle_calculation_direct(self): 
            
        
        if 1==1:
            
            #print(self.array_for_angle)
            
            if 1 ==1:

                
                t1 = time.time()
                

              
                from numpy  import array

                #myarray = array(a)

               

                self.array_for_angle_final = self.array_for_angle



 
                #print('af')
                #print(self.array_for_angle_final)



                self.max1 = np.amax(self.array_for_angle_final, axis=0)[0]
                #print(self.max1)  # Maxima along the first axis
                self.min1 = np.amin(self.array_for_angle_final, axis=0)[0]   # minima along the first axis 
                #print(self.min1)               
                self.total_lenght = (self.max1 - self.min1)



                #bellow we extract pat of the tail, if selected by the first slider

                if self.max_tail_position < self.total_lenght:
                    if self.max_tail_position < (self.total_lenght * 0.5):
                        self.max_tail_position = (self.total_lenght * 0.5)
                   
                    self.total_lenght = self.max_tail_position
                    value_final_new = self.min1 + self.total_lenght
                    self.array_for_angle_final = self.array_for_angle_final[self.array_for_angle_final[:, 0] < value_final_new]
                    

                '''self.max1_no_tail_value = int((self.min1 + self.total_lenght) * self.max_tail_position)
                
                self.max1_matrix_no_tail = self.array_for_angle_final[self.array_for_angle_final[:, 0] < self.max1_no_tail_value]
                
                self.max1 = np.amax(self.max1_matrix_no_tail, axis=0)[0]  # Maxima along the first axis without the maximum choosen
                
                self.total_lenght = (self.max1 - self.min1)
                self.array_for_angle_final = self.max1_matrix_no_tail'''
              







                self.max1 = np.amax(self.array_for_angle_final, axis=0)[0]  # Maxima along the first axis
                self.min1 = np.amin(self.array_for_angle_final, axis=0)[0]   # minima along the first axis                
                self.total_lenght = (self.max1 - self.min1)

                

                





                
                if (self.hub_position < (0.3)):
                    self.hub_position = (0.3)
                if self.hub_position < (self.min_head_position + 0.1):
                    self.hub_position = (self.min_head_position  + 0.1)

                if self.min_head_position > (self.hub_position - 0.1):
                    self.min_head_position = (self.hub_position - 0.1)
                
                '''if self.max_tail_position < (self.hub_position + 0.1):
                    self.max_tail_position = (self.hub_position + 0.1) '''




                  

                self.limit_min = int(self.min1 + ((self.total_lenght) * (self.min_head_position)))  #### exclude the 30% initial of the body, whhere will be considered the head

                self.limit_max = int(self.min1 + ((self.total_lenght) * (self.hub_position)))   #### choose where is the meeting betwen body and tail lines   
                
                



                matrix_bod_0 = self.array_for_angle_final[self.array_for_angle_final[:,0] < self.limit_max]   
                matrix_bod =  matrix_bod_0[np.where(matrix_bod_0[:,0] > self.limit_min)]


                

                
                matrix_tai = self.array_for_angle_final[self.array_for_angle_final[:, 0] > self.limit_min]  


                matrix_bod_y = matrix_bod[:, 1]
                matrix_bod_x = matrix_bod[:, 0] 


                matrix_tai_y = matrix_tai[:, 1]
                matrix_tai_x = matrix_tai[:, 0]

                # point 1 is the line segment of the body (not the tail)
                point1xmin = np.amin(matrix_bod,axis=0)[0]
                point1xmax = np.amax(matrix_bod,axis=0)[0]
                #point1ymin = np.amin(matrix_bod,axis=0)[1]
                #point1ymax = np.amax(matrix_bod,axis=0)[1]

                

                

                # point 2 is the line segment of the tail
                point2xmax = np.amax(matrix_tai,axis=0)[0]  

                #calculate averages y1 e y2 (only of the body extremes of the body) to use in the model and trace the virtual line until de tail
                
          
                y_body_mean = np.mean(matrix_bod_y)


                body_val_low = matrix_bod[matrix_bod[:, 1] < y_body_mean]
                body_val_high = matrix_bod[matrix_bod[:, 1] > y_body_mean]

                #print(body_val_low)
                #print(body_val_high)

                body_val_low_sorted = body_val_low[body_val_low[:,0].argsort(kind='mergesort')]
                body_val_high_sorted = body_val_high[body_val_high[:,0].argsort(kind='mergesort')]
                #body_val_low_sorted = body_val_low[body_val_low[:,0].argsort()]
                #body_val_high_sorted = body_val_high[body_val_high[:,0].argsort()]


               
                #Values are in pixelmap, not in real plot 
                body_min_val_low = body_val_low_sorted[0][1]
                body_min_val_high = body_val_high_sorted[0][1]
                body_max_val_high = body_val_high_sorted[-1][1]
                body_max_val_low = body_val_low_sorted[-1][1]

                
               


               
                y1 = np.average([body_min_val_low, body_min_val_high])              
                y2 = np.average([body_max_val_high, body_max_val_low])



                
             


                ##################################################################


                #calculate averages y_final_body para traçar a reta caudal até e média y da cauda
                y_tail = self.array_for_angle_final[self.array_for_angle_final[:, 0] == int(self.max1)]
                y_tail = y_tail[:,[1]]



                #print('y_tail')
                #print(y_tail)
                while len(y_tail) < 2:
                     self.max1 -= 1
                     y_tail = self.array_for_angle_final[self.array_for_angle_final[:, 0] == int(self.max1)]
                     y_tail = y_tail[:,[1]]
                   

                

                
                #print(y_tail)
                y_tail = (np.amin(y_tail,axis=0)[0] + np.amax(y_tail,axis=0)[0])/2

                #print(y_tail)

                matrix_tai_x_max = np.amax(matrix_tai_x)
                
               

                #predizer o valor de y onde x é igual ao final da cauda
                from sklearn.linear_model import LinearRegression
                x = np.array([point1xmin, point1xmax]).reshape((-1, 1))
                y = np.array([y1, y2])
                model_regression = LinearRegression()
                model_regression.fit(x, y)
                model_regression = LinearRegression().fit(x, y)
                x1 = np.array([matrix_tai_x_max]).reshape((-1, 1))
                y_pred_final_tail = model_regression.predict(x1)

                painter = QPainter(self.pixmap)
                pen = QPen()
                pen.setWidth(2)
                pen.setColor(QColor('orange'))
                painter.setPen(pen)
                painter.drawLine(point1xmin, y1,point2xmax, y_pred_final_tail)

                pen.setWidth(2)
                pen.setColor(QColor('green'))
                painter.setPen(pen)
                painter.drawLine(point1xmax, y2,matrix_tai_x_max, y_tail)

                pen.setWidth(2)
                pen.setColor(QColor('black'))
                painter.setPen(pen)

                font = QFont()
                font.setFamily('Times')
                font.setBold(True)
                font.setPointSize(10)
                painter.setFont(font)

                painter.drawText(20, 20, str(self.angle) + ' degrees' )

                
               
                painter.end() 

                self.label.setPixmap(self.pixmap)


                l1 = [(point1xmin,y1), (matrix_tai_x_max, y_pred_final_tail)]     # body
                l2 = [(point1xmax,y2),(matrix_tai_x_max, y_tail)]       # tail


                m1 = (l1[1][1]-l1[0][1])/(l1[1][0]-l1[0][0])

                m2 = (l2[1][1]-l2[0][1])
                m3 = (l2[1][0]-l2[0][0])

                if m3==0:
                    m4=0
                else:
                    m4=m2/m3

                angle_rad = abs(math.atan(m1) - math.atan(m4))
                angle_deg = angle_rad*180/PI


                angulofinal = round(angle_deg,2)


                if y_tail > y_pred_final_tail: 
                    angulofinal2 = angulofinal*-1 
                else: 
                    angulofinal2 = angulofinal

                

                    #angulofinal2 = angulofinal 

            else:
                angulofinal2 = 'NaN'

            self.angle = angulofinal2
            #print('ângulo = ' + str(angulofinal2))
               
        else:
            angulofinal2 = 'NaN'

            self.angle = angulofinal2
            #print('ângulo = ' + str(angulofinal2))
        
        

       

        
        
        
        

        '''self.y.append(self.angle)
        self.x.append(self.number)

        if len(self.y) > 0:
            self.update_plot_data()'''

        




    def function_slider_time(self):
        
        self.number_processed = self.slider_time.value()
        self.initial_open.setText(str(self.number_processed))
        
    
    def function_slider_time_change(self):
        self.number_processed = self.initial_open.text()
        #print(self.number_processed)
        if self.number_processed == '':
            self.number_processed = 10
        #if self.number_processed == 0:
            #self.number_processed = 1
        self.slider_time.setValue(int(self.number_processed))
        self.show_frame_choosen()

    def function_abort(self):
        self.abort = 1
        self.button_abort.hide()


    def function_start_framming(self):
        self.specie = self.edit_fish_specie.text()
        self.fish_id = self.edit_fish_id.text()
        if (self.head_choosen == 0 or self.specie =='' or self.fish_id =='' ):
            QMessageBox.about(self, "Alert", "Please select fish orientation, fish ID and fish specie first")
        else:
            self.button_start_framming.hide()
            self.frames()
        

    def mousePressEvent(self, event):

        if event.button() == Qt.LeftButton:
        
            self.origin = QPoint(event.pos())
            self.rubberBand.setGeometry(QRect(self.origin, QSize()))
            self.rubberBand.show()            
            self.start_x = event.pos().x() - 9
            if self.start_x < 0:
                self.start_x = 0
            self.start_y = event.pos().y() - 9
            if self.start_y < 0:
                self.start_y = 0
            #print(self.start_x)
            #print(self.start_y)
            
    def mouseMoveEvent(self, event):
    
        if not self.origin.isNull():
            self.rubberBand.setGeometry(QRect(self.origin, event.pos()).normalized())
    
    def mouseReleaseEvent(self, event):
    
        if event.button() == Qt.LeftButton:             
            
            self.end_x = event.pos().x() - 9
            if self.end_x > 640:
                self.end_x = 640
            self.end_y = event.pos().y() - 9
            if self.end_y > 360:
                self.end_y = 360         
            #print(self.end_x)
            #print(self.end_y)           
            
    def show_picture(self):
        #print(self.number)     
        
        
        if (self.number == -1):
            
            self.button_open = QPushButton('Open File', self)       
            self.button_open.clicked.connect(self.getfile)            
            self.button_open.show()
            self.layout.addWidget(self.button_open)
        
        else:
            pass 
            self.show_buttons()   
            
        
        #self.show()        
        self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)
        self.origin = QPoint()
        
            


    def getfile(self):       

       self.abort = 0

       fname_0 = QFileDialog.getOpenFileName(self, 'Open file', 
         'c:\\',"Image files (*.mp4)")          
      
       self.fname = fname_0[0] + '/'

       self.step = 0
              
       #print(self.fname)
      
       if self.fname != '/' :
         #self.frames(fname)
         self.file = self.fname[:-1]         
         cap = cv2.VideoCapture(self.file)
         self.fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"         
         self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

         self.number_processed = self.frame_count
         self.slider_time.setMaximum(self.frame_count)
         self.initial_open.setValidator(QIntValidator(10, self.frame_count))


         self.duration = int(self.frame_count/self.fps)
         #print('fps = ' + str(self.fps))
         #print('number of frames = ' + str(self.frame_count))
         #print('duration (S) = ' + str(self.duration))
         self.minutes = int(self.duration/60)
         self.seconds = self.duration%60
         #print('duration (M:S) = ' + str(self.minutes) + ':' + str(self.seconds))
         self.time_now = int(time.time()) 
         success, image = cap.read()
         if success:
                #print('sucess')
                
                try:
                    os.mkdir(self.fname[:-5] + '_' + str(self.time_now))
                except:
                    print('folder already exists')

                #print(self.fname[:-5] + '_' + str(self.time_now) + "/first_frame.jpg")

                resized_frame = cv2.resize(image, (640, 360), interpolation = cv2.INTER_LINEAR) # 640, 360

                cv2.imwrite(self.fname[:-5] + '_' + str(self.time_now) + "/first_frame.jpg", resized_frame)

                self.pixmap = QPixmap(self.fname[:-5] + '_' + str(self.time_now) + "/first_frame.jpg")
                #self.pixmap = self.pixmap.scaledToWidth(640)
                #self.pixmap = self.pixmap.scaledToHeight(360)
                self.label.setPixmap(self.pixmap)


         cap.release()
         self.initial_message1.show()
         self.initial_message2.show()  
         self.initial_open.show()
         self.orientation_message.show()
         self.left_checked.show()
         self.right_checked.show()
         self.button_start_framming.show()

         self.button_open_csv.hide()

         self.label_fish_specie.show()
         self.label_fish_id.show()
         self.edit_fish_id.show()
         self.edit_fish_specie.show()
         self.button_save_back_pic.show()

         self.show_frame_choosen()
         
         self.initial_open.setText(str(self.frame_count))
         self.slider_time.show()        
         self.slider_time.setMaximum(self.frame_count)
      
         
         self.setFixedSize(660,525)   #486
         #self.frames(fname) 
    
    def show_frame_choosen(self):
         self.duration_choosen = int(self.number_processed)/int(self.fps) ###### 
         minutes_choosen = int(self.duration_choosen/60)
         seconds_choosen = self.duration_choosen%60
         self.initial_message1.setText('Extract the first')
         self.initial_message2.setText('frames (or ' + str(minutes_choosen) + 'm:' + str(int(seconds_choosen)) + 's) of ' + str(self.frame_count) + ' frames (or ' + str(self.minutes) + 'm:' + str(self.seconds) + 's)\nfps= ' + str(round(self.fps)))
    

    def frames(self):       
        #self.button_start_selections.hide()
        self.process_image.show()
        self.button_abort.show() 
        self.button_open.hide()  
        
        
        #print(self.fname) 
        #print(self.fname[:-5])          
        self.file = self.fname[:-1]
        cap= cv2.VideoCapture(self.file)
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        i=0
        
        #print(self.fname[:-5] + '_' + str(self.time_now))    
        try:
            os.mkdir(self.fname[:-5] + '_' + str(self.time_now))
        except:
            print('folder already exists')     
        while(cap.isOpened()):        
            ret, frame = cap.read()
            if ret == False:
                break                   
            
            resized_frame = cv2.resize(frame, (640, 360), interpolation = cv2.INTER_LINEAR) # 640, 360

            
            if (self.head_position == 'right'):

                        flipHorizontal = cv2.flip(resized_frame, 1)
                        resized_frame = flipHorizontal
            
            cv2.imwrite(self.fname[:-5] + '_' + str(self.time_now) + '/' + str(i) +'.jpg',resized_frame)               
             
            i+=1
            
            
            self.process_image.setText('Processed '+ str(i) + ' frames')
            self.process_image.adjustSize()
            #self.process_image.show()
            self.repaint()
           
            
            print ("opened: " + str(i))
            self.processed_image = str(i)

            self.number_processed = self.initial_open.text()
            if self.number_processed != '':
                if i == int(self.number_processed):
                    break
            if self.abort == 1:                
                self.button_open.show()
                #self.button_sta.hide()
                break
        
            QApplication.processEvents()    
            
        cap.release()

        if self.abort == 1:
        
            self.restart_app()
            self.abort = 0
        else:
            print ("Done with frames!")
            self.process_image.hide()     
            
            
            list = os.listdir(self.file[:-4] + '_' + str(self.time_now)) # dir is your directory path
            self.number_files = len(list) -1   # to exclude first image
            #print(str(self.number_files) + ' images ')

            
            '''if self.abort == 0:
                self.show_buttons()
                #self.button_start_selections.show()       
                self.setFixedSize(660,420)'''

            

            self.button_abort.hide()

            self.show_buttons()
            self.on_click_next()

    def show_buttons(self):






        self.graphWidget = pg.PlotWidget()
        self.graphWidget.setBackground('w')
        pen = pg.mkPen(color=(0, 255, 0))
        self.data_line =  self.graphWidget.plot(self.x, self.y, pen=pen, width=2)
        self.layout.addWidget(self.graphWidget)
        self.graphWidget.hide()
        
                
        

        self.slider_video = QSlider()
        self.slider_video.setOrientation(Qt.Horizontal)
        #self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider_video.setTickInterval(1)
        self.slider_video.setMinimum(0)
        self.slider_video.setMaximum(int(self.number_files-1))        
        #self.slider_video.valueChanged.connect(self.function_slider_video)
        self.slider_video.valueChanged.connect(self.function_slider_video)
        self.slider_video.sliderMoved.connect(self.function_slider_video_pressed)
        self.slider_video.sliderMoved.connect(self.function_slider_video_moved)
        self.slider_video.sliderReleased.connect(self.function_slider_video_released)
        self.slider_video.hide()


        self.L1 = QLabel('message', self)
        self.L1.hide()  


        self.button_first = QPushButton('First Frame', self)
        self.button_first.setToolTip('Go back to first frame')       
        self.button_first.clicked.connect(self.function_first)
        self.button_first.hide()

        self.button_next = QPushButton('Advance Frame', self)
        self.button_next.setToolTip('Use to check how your selections looks like in different frames')
       
        self.button_next.clicked.connect(self.on_click_next)
        self.button_next.hide()  

        
        self.button_previous = QPushButton('Back Frame', self)
        self.button_previous.setToolTip('Use to check how your selections looks like in different frames')
       
        self.button_previous.clicked.connect(self.on_click_previous)
        self.button_previous.hide()

        
            

        self.how_many_Label = QLabel(self)
        self.how_many_Label.setText('How many:')
        self.how_many_Label.setToolTip('How many frames to advance when button is clicked?')
        self.how_many_Label.hide()
        self.how_many_edit = QLineEdit(self)
        self.how_many_edit.setText('1')
        self.how_many_edit.setValidator(QIntValidator()) #Settings can only input data of type int
        self.how_many_edit.setValidator(QDoubleValidator()) #Settings can only input data of type double
        #self.how_many_edit.setFixedWidth(120)
        self.how_many_edit.hide()


        self.button_init_frame = QPushButton('Set initial frame', self)       
        self.button_init_frame.clicked.connect(self.on_click_set_init_frame)
        self.button_init_frame.setToolTip('It will be the first frame if not selected')
        self.button_init_frame.hide()

        '''self.initial_frame_edit = QLineEdit(self)        
        self.initial_frame_edit.setValidator(QIntValidator()) #Settings can only input data of type int
        self.initial_frame_edit.setValidator(QDoubleValidator()) #Settings can only input data of type double
        self.initial_frame_edit.hide()'''

        self.initial_frame_edit = QLineEdit(self)        
        self.initial_frame_edit.setValidator(QIntValidator()) #Settings can only input data of type int
        self.initial_frame_edit.setValidator(QDoubleValidator()) #Settings can only input data of type double
        self.initial_frame_edit.setValidator(QIntValidator(0, self.number_files -1))          
        self.initial_frame_edit.hide()

        
        self.button_final_frame = QPushButton('Set final frame', self)       
        self.button_final_frame.clicked.connect(self.on_click_set_final_frame)
        self.button_final_frame.setToolTip('It will be the last frame if not selected')
        self.button_final_frame.hide()

        '''self.final_frame_edit = QLineEdit(self)        
        self.final_frame_edit.setValidator(QIntValidator()) #Settings can only input data of type int
        self.final_frame_edit.setValidator(QDoubleValidator()) #Settings can only input data of type double
        self.final_frame_edit.setValidator(QIntValidator(1, 999)) #Settings can only input data of type double
        self.final_frame_edit.hide()'''

        self.final_frame_edit = QLineEdit(self)        
        self.final_frame_edit.setValidator(QIntValidator()) #Settings can only input data of type int
        self.final_frame_edit.setValidator(QDoubleValidator()) #Settings can only input data of type double
        self.final_frame_edit.setValidator(QIntValidator(0, self.number_files-1))       
        self.final_frame_edit.hide()

        self.bar = QProgressBar(self)        
        self.p_value = self.bar.text()
        
        self.label_bar = QLabel(' ', self)
        self.bar.setValue(0)
        self.bar.hide()
        self.label_bar.hide()

        self.resulting_label = QLabel(' ', self)
        self.resulting_label.hide()

        self.slider_pan = QSlider()
        self.slider_pan.setOrientation(Qt.Horizontal)
        #self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider_pan.setTickInterval(1)
        self.slider_pan.setMinimum(50)
        self.slider_pan.setMaximum(200)        
        #self.slider_video.valueChanged.connect(self.function_slider_video)
        self.slider_pan.valueChanged.connect(self.update_plot_data)        
        self.slider_pan.hide()

        self.pan_label = QLabel('x axis\nrange', self)
        self.pan_label.hide()     


        self.button_play = QPushButton('Play', self)
        self.button_play.setToolTip('Watch the video')       
        self.button_play.clicked.connect(self.function_play)        
        self.button_play.hide()

        self.button_stop = QPushButton('Stop', self)
        self.button_stop.setToolTip('Stop the video')       
       
        self.button_stop.clicked.connect(self.function_stop)  
        self.button_stop.hide()

        self.button_play_again = QPushButton('Play Again', self)       
        self.button_play_again.clicked.connect(self.function_play_again)  
        self.button_play_again.hide()

        self.combo_play = QComboBox(self)
        self.combo_play.addItem("60 fps")              
        self.combo_play.addItem("30 fps")
        self.combo_play.addItem("10 fps")
        self.combo_play.setToolTip('Watching velocity')  
        self.combo_play.hide()
        self.combo_play.activated[str].connect(self.onChanged_play)       


        self.button_select = QPushButton('Select Study Area', self) 
        self.button_select.setToolTip('Select the area where the fish can be found at any time')  

        self.button_select.setStyleSheet("border :1px solid red")        
        self.button_select.clicked.connect(self.crop_study_area)             
        self.button_select.hide()

        self.button_select_back = QPushButton('Select Fish', self)
        self.button_select_back.setToolTip('Select the fish so we can know its aproximate size') 
        self.button_select_back.setStyleSheet("border :1px solid green")        
        self.button_select_back.clicked.connect(self.crop_back_area)         
        self.button_select_back.hide()




        self.button_restart = QPushButton('Restart Application', self)
        self.button_restart.setToolTip('Go back and choose another video') 
        
        self.button_restart.clicked.connect(self.restart_app)         
        self.button_restart.hide() 

        

        self.button_save = QPushButton('Save the CSV and TXT files', self)
        self.button_save.setToolTip('Save the CSV and TXT files')        
        self.button_save.clicked.connect(self.save_csv)         
        self.button_save.hide()

        self.button_build_report = QPushButton('Set up report', self)
        self.button_build_report.setToolTip('Choose the report options')        
        self.button_build_report.clicked.connect(self.build_report_step1)         
        self.button_build_report.hide()


        


        self.button_next_step = QPushButton('Next Step', self)        
        self.button_next_step.clicked.connect(self.function_next_step)
        self.button_next_step.setToolTip('Go to next step')         
        self.button_next_step.hide() 

        self.button_previous_step = QPushButton('Previous Step', self)        
        self.button_previous_step.clicked.connect(self.function_previous_step)
        self.button_previous_step.setToolTip('Back to previous step')         
        self.button_previous_step.hide()


        self.button_abort_counting = QPushButton('Abort', self)        
        self.button_abort_counting.clicked.connect(self.function_abort_counting)
        self.button_abort_counting.setToolTip('Abort counting')         
        self.button_abort_counting.hide()  

        '''self.color_edit = QLineEdit(self)        
        self.color_edit.setValidator(QIntValidator()) #Settings can only input data of type int
        self.color_edit.setValidator(QDoubleValidator()) #Settings can only input data of type double
        self.color_edit.setValidator(QIntValidator(0, 255))       
        self.color_edit.hide()


        self.button_apply = QPushButton('Apply', self)
        self.button_apply.setToolTip('Apply changes')       
       
        self.button_apply.clicked.connect(self.function_apply)  
        self.button_apply.hide()'''

        
        self.layout_h6 = QHBoxLayout()
        self.layout_h6_1 = QHBoxLayout()     
        self.layout_h1 = QHBoxLayout()              
        self.layout_h2 = QHBoxLayout()        
        self.layout_h3 = QHBoxLayout()
        self.layout_reports = QHBoxLayout()

       

        self.layout_h6.addWidget(self.slider_video)
        self.layout_h6.addWidget(self.L1)

        self.layout_h6_1.addWidget(self.pan_label)
        self.layout_h6_1.addWidget(self.slider_pan)

        self.layout_h1.addWidget(self.button_next)        
        self.layout_h1.addWidget(self.button_previous)
        self.layout_h1.addWidget(self.button_first) 

        self.layout_h1.addWidget(self.how_many_Label)        
        self.layout_h1.addWidget(self.how_many_edit)

        self.layout_h1.addWidget(self.button_play)       
        self.layout_h1.addWidget(self.button_stop)         
        self.layout_h1.addWidget(self.button_play_again)
        self.layout_h1.addWidget(self.combo_play)
        

        self.layout_h2.addWidget(self.button_init_frame)
        self.layout_h2.addWidget(self.initial_frame_edit)

        self.layout_h2.addWidget(self.button_final_frame)         
        self.layout_h2.addWidget(self.final_frame_edit)            

        self.layout_h3.addWidget(self.button_select)         
        self.layout_h3.addWidget(self.button_select_back)
        
        
        self.layout_reports.addWidget(self.button_save)
        
        self.layout_reports.addWidget(self.button_build_report)


            
       
        self.layout.addLayout(self.layout_h6)
        self.layout.addWidget(self.bar)
        self.layout.addWidget(self.label_bar)
        self.layout.addWidget(self.resulting_label)
        self.layout.addLayout(self.layout_h6_1)                     
        self.layout.addLayout(self.layout_h1)        
        self.layout.addLayout(self.layout_h2) 
        self.layout.addLayout(self.layout_h3)
        #self.layout.addWidget(self.button_previous_step)          
        self.layout.addWidget(self.button_next_step)        
        self.layout.addWidget(self.button_restart)
        self.layout.addLayout(self.layout_reports)


        #self.layout.addWidget(self.color_edit)
        #self.layout.addWidget(self.button_apply)

            
        


        
    def show_secondary_buttons(self):

        self.button_select_model = QPushButton('Set Fish Model', self)
        self.button_select_model.setToolTip('If you have a good fish silhouet,\nfit it as the model for better accuracy')             
        self.button_select_model.clicked.connect(self.function_select_model)
        self.button_select_model.hide()

        self.button_unselect_model = QPushButton('Clear the Fish Model', self)
        self.button_unselect_model.setToolTip('Clear fish model if you made a mistake')             
        self.button_unselect_model.clicked.connect(self.function_unselect_model)
        self.button_unselect_model.hide()


        self.label_slider_title = QLabel('Background Subtraction Threshold')         
        self.label_slider_title.hide()

        self.slider = QSlider()
        self.slider.setOrientation(Qt.Horizontal)
        
        self.slider.setMinimum(0)
        self.slider.setTickInterval(1)
        
        self.slider.setMaximum(255)
        if self.values_color > 0:
            self.slider.setValue(self.values_color)        
        self.slider.valueChanged.connect(self.function_slider)
        self.slider.hide()
        self.label_slider_title.hide()


        self.slider_thr = QSlider()
        self.slider_thr.setOrientation(Qt.Horizontal)        
        self.slider_thr.setTickInterval(1)
        self.slider_thr.setMinimum(3)
        self.slider_thr.setMaximum(255)           
        self.slider_thr.valueChanged.connect(self.function_slider_thr)
        self.slider_thr.show()


        self.slider_background = QSlider()
        self.slider_background.setOrientation(Qt.Horizontal)        
        self.slider_background.setTickInterval(1)
        self.slider_background.setMinimum(0)
        self.slider_background.setMaximum(255)
        self.slider_background.setValue(61)           
        self.slider_background.valueChanged.connect(self.function_slider_background)
        self.slider_background.hide()



        self.label_slider = QLabel(self)
        self.label_slider.hide()



        self.label_slider_minimum_title = QLabel('Minimum Size for Detection')        
        self.label_slider_minimum_title.hide()

        self.slider_minimun_size = QSlider()
        self.slider_minimun_size.setOrientation(Qt.Horizontal)             
        self.slider_minimun_size.setTickInterval(self.fish_size)
        #self.fish_area = (self.end_y_background - self.start_y_background) * (self.end_x_background - self.start_x_background)
        self.slider_minimun_size.setMinimum(1)
        self.slider_minimun_size.setMaximum(self.fish_size)        
        if self.size_minimum != 0:
            self.slider_minimun_size.setValue(self.size_minimum)
        else:
            self.slider_minimun_size.setValue(0)
            

        self.slider_minimun_size.valueChanged.connect(self.function_slider_minimum)
        self.slider_minimun_size.hide()


        self.label_slider_minimum = QLabel(self)
        self.label_slider_minimum.hide()


        
        self.label_slider_maximum_title = QLabel('Maximum Size for Detection')         
        self.label_slider_maximum_title.hide()
        

        self.slider_maximum_size = QSlider()
        self.slider_maximum_size.setOrientation(Qt.Horizontal)
        #self.slider_maximum_size.setStyleSheet("height: 10px")        
        self.slider_maximum_size.setTickInterval(self.fish_size)
        self.slider_maximum_size.setMinimum(self.fish_size/3)
        self.slider_maximum_size.setMaximum(self.fish_size)
        if self.size_maximum != 194400:
            self.slider_maximum_size.setValue(self.size_maximum)
        else:
            self.slider_maximum_size.setValue(self.fish_size*3)

        self.slider_maximum_size.valueChanged.connect(self.function_slider_maximum)
        self.slider_maximum_size.hide()

        self.label_slider_maximum = QLabel(self)
        self.label_slider_maximum.hide()

        
        self.layout_h7 = QHBoxLayout()
        
        self.layout_h0 = QHBoxLayout()
        self.layout_min_max_title = QHBoxLayout()
        self.layout_min_max = QHBoxLayout()


        
        self.layout_h7.addWidget(self.button_select_model)
        self.layout_h7.addWidget(self.button_unselect_model)
        self.layout_h7.addWidget(self.button_previous_step)
        self.layout_h7.addWidget(self.button_next_step)
        self.layout_h7.addWidget(self.button_abort_counting)
        self.layout_h7.addWidget(self.button_restart)

        
        self.layout_min_max_title.addWidget(self.label_slider_maximum_title)
        self.layout_min_max_title.addWidget(self.label_slider_minimum_title)
        #self.layout_min_max_title.setAlignment(Qt.AlignCenter) 


        self.layout_min_max.addWidget(self.slider_maximum_size)
        self.layout_min_max.addWidget(self.label_slider_maximum)

        self.layout_min_max.addWidget(self.slider_minimun_size)
        self.layout_min_max.addWidget(self.label_slider_minimum)

        self.layout_h0.addWidget(self.slider)
        self.layout_h0.addWidget(self.label_slider)

        
        self.layout.addLayout(self.layout_h7)
        self.layout.addWidget(self.label_slider_title)         
        self.layout.addLayout(self.layout_h0)
        self.layout.addWidget(self.slider_thr)
        self.layout.addWidget(self.slider_background)

        self.layout.addLayout(self.layout_min_max_title)             
        self.layout.addLayout(self.layout_min_max)
            
             


    def function_slider_thr(self):       
        self.thr_0 = self.slider_thr.value()
        #print(self.thr % 2)
        if (self.thr_0 % 2) != 0:
            self.thr = self.thr_0
            #print(self.thr)
            self.compose_pictures()


    def function_slider_background(self):
        self.background = self.slider_background.value()
        self.set_mask()


    def compose_pictures(self):

        if self.step == 0:      
             
            self.button_select_back.show()
            self.button_select.show()


            #print('compose_pictures')

            self.edit_fish_id.setEnabled(False)
            self.edit_fish_specie.setEnabled(False)
            self.button_open_csv.hide()
  


            self.pixmap = QPixmap(self.file[:-4] + '_' + str(self.time_now) + '/' + str(self.number) +'.jpg') 
            self.pixmap = self.pixmap.scaledToWidth(640)
            
            if (self.croped_study_area == 1 or self.croped_background == 1):
                self.painterInstance = QPainter(self.pixmap)             


            if self.croped_study_area == 1:
                self.penRectangle = QPen(Qt.red)
                self.penRectangle.setWidth(2)
                self.painterInstance.setPen(self.penRectangle)
                
                self.painterInstance.drawRect(self.start_x_study, self.start_y_study, (self.end_x_study - self.start_x_study), (self.end_y_study - self.start_y_study))            
                
            
            
            if self.croped_background == 1:
                self.penRectangle = QPen(Qt.green)
                self.penRectangle.setWidth(2)
                self.painterInstance.setPen(self.penRectangle)
                self.painterInstance.drawRect(self.start_x_background, self.start_y_background, (self.end_x_background - self.start_x_background), (self.end_y_background - self.start_y_background))        
                
            if (self.croped_study_area == 1 or self.croped_background == 1):
                self.painterInstance.end()
            
            #self.pixmap = self.pixmap.scaledToWidth(640)
            self.label.setPixmap(self.pixmap)
           
            self.setFixedSize(660,574)   #534
        
        elif self.step == 1:
            self.set_mask()


        elif self.step == 2:
            self.set_mask()
            self.angle_calculation_direct()


        else:
            pass


    
    def masking_step(self):
        if self.back_sub_type == 'numbers':
            self.slider.show()
            self.label_slider_title.show()
            self.label_slider.show()
        if self.back_sub_type == 'image' and self.img_back == '':
            QMessageBox.about(self, "Alert", "You did not upload any background image.\nYou need to choose one before take any action, or just uncheck the \"Image Subtraction\" box")
        
        self.label_slider.show()        
        self.label_slider_minimum.show()
        self.label_slider_maximum.show()
        self.slider_minimun_size.show() 
        self.slider_maximum_size.show()
        self.label_slider_title.show()
        self.label_slider_minimum_title.show()
        self.label_slider_maximum_title.show()

        if self.fish_model.shape[0] > 1:
            self.button_select_model.hide()
            self.button_unselect_model.show()
        else:
            self.button_select_model.show()
            self.button_unselect_model.hide()

              
        



    def set_mask(self):



        if self.step == 1:
            self.masking_step()        

        c_index = []                 
        self.button_select.hide()        
        self.button_select_back.hide()
        self.left_checked.hide() 
        self.right_checked.hide()        

       
        
        image = self.file[:-4] + '_' + str(self.time_now) + '/' + str(self.number) +'.jpg'         
        
        self.image = cv2.imread(image)
        self.clone = self.image.copy()

        self.crop_img = self.clone[self.start_y_study:self.end_y_study, self.start_x_study:self.end_x_study]

        #we changed background funciton, now it define the fishes´s size        

        
        
        if self.back_sub_type == 'image' and self.img_back != '':        
            depth_back_ini = cv2.imread(self.img_back)
            depth_back = depth_back_ini[self.start_y_study:self.end_y_study, self.start_x_study:self.end_x_study]
        else:
            average_background = [int(self.values_color),int(self.values_color),int(self.values_color)]        
            depth_back = np.zeros((self.image.shape[0], self.image.shape[1], 3), np.uint8)
            depth_back[:,:] = (average_background)
            depth_back = depth_back[self.start_y_study:self.end_y_study, self.start_x_study:self.end_x_study]            
            

        original_img_grey = cv2.cvtColor(self.crop_img, cv2.COLOR_BGR2GRAY)
        depth_back = cv2.cvtColor(depth_back, cv2.COLOR_BGR2GRAY)
        

        if self.back_sub_type == 'image' and self.img_back != '':       
            depth_diff = cv2.absdiff(depth_back, original_img_grey)
            depth_diff = cv2.GaussianBlur(depth_diff, (self.thr,self.thr) ,0)
            depth_back = cv2.GaussianBlur(depth_back, (self.thr,self.thr) ,0)          
            ret,thresh = cv2.threshold(depth_diff,self.background,255,cv2.THRESH_BINARY)
        else:
            depth_diff = abs(depth_back - original_img_grey)        
            depth_diff = cv2.GaussianBlur(depth_diff, (self.thr,self.thr) ,0)
            ret, thresh = cv2.threshold(depth_diff, 127, 255, 0)









        #contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)       
        
        
        blank_image = np.zeros((self.image.shape[0], self.image.shape[1], 3), np.uint8)
        blank_image[:, :] = (255,255,255) 

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cont_size_filtered = []


        c_index = []
        x=0
        for cnt in contours:
            
            area = cv2.contourArea(cnt)    

            if (area > int(self.slider_minimun_size.value()) and area < int(self.slider_maximum_size.value())):
            #if (area > 2):
                c_index.append(x)
                cont_size_filtered.append(cnt)
                x=x+1
             

        # from now, we only use cont_size_filtered and c_index for all other filters                 


        if c_index:

            if self.fish_model.shape[0] > 1:  # has a model, then filter as much as possible              
                
                list_values = []
                list_index = [] 

                list_area = []
                list_index_mass_X = []
                list_index_mass_Y = []
                n_init = 0

                #print(c_index)
                #print(cont_size_filtered)
                
                for c in c_index:

                    
                    M = cv2.moments(cont_size_filtered[c])
                    #print('M')
                    #print(M)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                                    

                    #area = cv2.matchShapes(self.fish_model, cont_size_filtered[c], 3, 0)
                    area = cv2.contourArea(cont_size_filtered[c])

                    list_values.append(cont_size_filtered[c])
                    list_index.append(n_init)
                    #list_similarity.append(similarity)
                    list_area.append(area)                    
                    list_index_mass_X.append(cX)
                    list_index_mass_Y.append(cY)    

                    n_init += 1
                   
                # now we changed our contor list to list_values and list_index

                #index_of_model = np.argmin(list_similarity)   # get the index of the model, and aply the values to the list bellow

                index_of_model = np.argmax(list_area)   

               
             
                list_final_values = [cont_size_filtered[index_of_model]]
                       

                model_point_maxX = np.amax(cont_size_filtered[index_of_model], axis=0)[0][0]
                model_point_minX = np.amin(cont_size_filtered[index_of_model], axis=0)[0][0]
               

                for v in list_index:
                    if v != index_of_model and list_index_mass_X[v] < (model_point_minX + (self.model_lenght * 1.1)) \
                    and list_index_mass_X[v] > int(model_point_maxX * 0.9) \
                    and list_index_mass_Y[v] > (list_index_mass_Y[index_of_model] - (self.model_height/3)) \
                    and list_index_mass_Y[v] < (list_index_mass_Y[index_of_model] + (self.model_height/3)):                        
                        list_final_values.append(cont_size_filtered[v])          


                

                
                self.array_for_angle_prepare = np.vstack(list_final_values)

            


                self.array_for_angle_prepare=self.array_for_angle_prepare[:,0]

                self.array_for_angle_prepare = self.array_for_angle_prepare[self.array_for_angle_prepare[:,0].argsort(kind='mergesort')]



              
    
                
                self.final_array = self.array_for_angle_prepare.reshape(self.array_for_angle_prepare.shape[0], 1, self.array_for_angle_prepare.shape[1])
                
             

                final_list = []
                final_list.append(self.final_array)


              
                blank_image = np.zeros((self.image.shape[0], self.image.shape[1], 3), np.uint8)
                blank_image[:, :] = (255,255,255) 
                self.final_image = cv2.drawContours(blank_image, final_list, -1, color=(255,0,0),thickness=-1)

                img_for_cont = cv2.cvtColor(self.final_image, cv2.COLOR_BGR2GRAY)
                img_for_cont = cv2.bitwise_not(img_for_cont)

                #img_for_cont = cv2.GaussianBlur(img_for_cont, (3,3) ,0)

                contours, _ = cv2.findContours(img_for_cont, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                
                self.final_image = cv2.drawContours(blank_image, contours, -1, color=(255,0,0),thickness=-1)   




                self.array_for_angle_prepare = np.vstack(contours)   #aqui vai ser utilizado para os angulos
                self.array_for_angle = np.vstack(self.array_for_angle_prepare)  

                                  

                areas = [cv2.contourArea(c) for c in contours]
                max_index = np.argmax(areas)
                max_area=contours[max_index]                
                self.fish_pre_model = max_area


                self.heat_x, self.heat_y = list_index_mass_X[index_of_model], list_index_mass_Y[index_of_model]
                #print('self.heat_x')
                #print(self.heat_x)


            else: # does not have model but has c_index

                #print('tem c_index, mas não model')               
                
                self.final_image = cv2.drawContours(blank_image, cont_size_filtered, -1, color=(255,0,0),thickness=-1)

                '''self.array_for_angle_0 = np.vstack(cont_size_filtered)                                   
                self.array_for_angle = self.array_for_angle_0[:,0]'''


                # Find the largest contour to use as the model
                areas = [cv2.contourArea(c) for c in cont_size_filtered]
                max_index = np.argmax(areas)
                cnt=cont_size_filtered[max_index]
                self.fish_pre_model = cnt

                '''print('cont_size_filtered')
                print(cont_size_filtered)

                print('areas')
                print(areas)'''

                

            

                '''value = [255, 255, 255]
                self.bordered_diff = cv2.copyMakeBorder(depth_diff, self.start_y_study, (360 - self.end_y_study), self.start_x_study, (640 - self.end_x_study), cv2.BORDER_CONSTANT, None, value)
                self.final_image = self.bordered_diff '''



        

                
                    

        else:  # there is not c_index. just print a grey image

            self.fish_pre_model = np.array([1]) 
            value = [255, 255, 255]
            self.bordered_diff = cv2.copyMakeBorder(depth_diff, self.start_y_study, (360 - self.end_y_study), self.start_x_study, (640 - self.end_x_study), cv2.BORDER_CONSTANT, None, value)
            self.final_image = self.bordered_diff










            
        if self.step != 3:

            if (self.fish_model.shape[0] > 1) and c_index:
                pass

                '''rect = cv2.minAreaRect(cont_size_filtered[index_of_model])
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                self.final_image = cv2.drawContours(self.final_image,[box],0,(0,0,255), 2)

                print('box')
                print(box)'''

                '''cols = self.model_lenght
                [vx,vy,x,y] = cv2.fitLine(cont_size_filtered[index_of_model], cv2.DIST_L2,0,0.01,0.01)
                lefty = int((-x*vy/vx) + y)
                righty = int(((cols-x)*vy/vx)+y)
                self.final_image = cv2.line(self.final_image,(cols-1,righty),(0,lefty),(0,255,0),2)'''

                


                

                '''hull_list=[]
                chunks = [self.array_for_angle[i:i + 100] for i in range(0, len(self.array_for_angle), 100)]
                print('len_chunks')
                print(len(chunks))
                for n in range(len(chunks)):                               
                    #print(chunks[n])
                    hull = cv2.convexHull(chunks[n])                
                    final_hull = cv2.drawContours(self.final_image, [hull], -1, (0, 255, 0), -1)'''

                #epsilon = 0.01*cv2.arcLength(self.array_for_angle,True)
                #approx = cv2.approxPolyDP(self.array_for_angle,epsilon,True)
                #self.final_image = cv2.drawContours(self.final_image,[approx],-1,(0,0,255), 2)

            yourQImage=qimage2ndarray.array2qimage(self.final_image) 
            self.pixmap = QPixmap(yourQImage)

            if (self.fish_model.shape[0] > 1) and c_index:
                if list_index: 
                    painter = QPainter(self.pixmap)
                    pen = QPen()
                    pen.setWidth(5)
                    pen.setColor(QColor('green'))
                    painter.setPen(pen)
                    painter.drawPoint(list_index_mass_X[index_of_model], list_index_mass_Y[index_of_model])                    
                    painter.end()



               
                
               
                
              




            self.label.setPixmap(self.pixmap)

            

        
         
        
               
            
  


    @pyqtSlot()
    def function_first(self):
        self.x = []
        self.y = []      
        self.button_first.setEnabled(False)
        self.button_next.setEnabled(True)
        self.button_first.hide()
        self.number = -1
        self.button_play_again.hide()
        self.on_click_next()
        if self.step == 1:
            self.button_select.hide()
            self.button_select_back.hide()
            #self.label_fish_specie.setEnabled(False)
            #self.label_fish_id.setEnabled(False)
            





    @pyqtSlot()
    def on_click_next(self):
        #print('cliocu')
        #print(self.step)
        self.number += int(self.how_many_edit.text())

        if self.step == 1:
            self.button_select_back.hide()
            self.button_select.hide()
            self.left_checked.hide()
            self.right_checked.hide()




        
        
         
        if (self.number >= self.number_files - 1):
            self.number = self.number_files -1
            self.button_next.setEnabled(False)
            
            QMessageBox.about(self, "Alert", "We can´t advance " + str(self.how_many_edit.text()) + " frames. We will leave you in the last frame")     

            #print(self.number_files)
       
         

            
        
        #print(self.number) 

        if (self.number <= 0):
            #print(self.number)
            self.button_previous.show()
            self.button_previous.setEnabled(False)

        if (self.number > 0):
            self.button_previous.setEnabled(True)
            self.button_first.show()
            self.button_first.setEnabled(True)




        #print(self.number) 
        if (self.number >= 0):
            #print(self.number)           
            self.button_next.show()
            self.L1.show()
            self.L1.setText('Frame:\n ' + str(self.number) + ' of ' + str(self.number_files -1))
            self.slider_video.setValue(self.number)
            
            self.button_next.show()            
            self.how_many_Label.show()
            self.how_many_edit.show()
            
            
            self.button_open.hide()
            self.button_restart.show()
            self.button_next_step.show()
            self.button_play.show()
            self.right_checked.hide()
            self.left_checked.hide()
            self.orientation_message.hide()
            self.combo_play.show() 
            #self.button_start_selections.hide()
            self.initial_message1.hide()
            self.initial_message2.hide()
            self.initial_open.hide()            
            self.slider_video.show()
            self.slider_time.hide() 

            self.button_init_frame.show()
            self.button_final_frame.show()
            self.initial_frame_edit.show()
            self.final_frame_edit.show()

            



        '''if self.step == 1:
            self.button_select_back.hide()
            self.button_select.hide()
            self.left_checked.hide()
            self.right_checked.hide()'''

            

        self.compose_pictures()

        self.y.append(self.angle)
        self.x.append(self.number)

        
        if self.step == 2 or self.step == 3:
            if len(self.y) > 0:
                self.update_plot_data()     


    @pyqtSlot()
    def on_click_previous(self):
        #print('self_x')
        #print(self.x)              

        self.number -= int(self.how_many_edit.text())
        self.button_play_again.hide()
        self.button_play.show()

        if (self.number < 0):
            self.number = 0
            QMessageBox.about(self, "Alert", "We can´t back " + str(self.how_many_edit.text()) + " frames. We will leave you in the first frame")     
            self.button_next.setEnabled(True)

        

        if (self.number == 0):
            self.button_previous.setEnabled(False)
            self.button_first.hide()
        
        if (self.number > -1):
            #print(self.number)           
            self.button_next.show()
            self.L1.show()
            self.L1.setText('Frame:\n' + str(self.number) + ' of ' + str(self.number_files -1))
            self.slider_video.setValue(self.number)
            
            '''if self.step == 2:
                self.button_select_back.show()
                self.button_select.show()'''  
        #self.show_picture()
    
        if self.step == 1:
            self.button_select_back.hide()
            self.button_select.hide()
            self.left_checked.hide()
            self.right_checked.hide()
            #print('aqui?')
        

        self.compose_pictures() 

        
        if self.step == 2 or self.step == 3:
                if self.x:
                     self.x.pop()
                     self.y.pop()
                     #print(self.x)     
                     self.update_plot_data()   

        
    @pyqtSlot()
    def crop_study_area(self):
    
        self.rubberBand.hide()       
        part = 'study'
        self.crop_areas(part)

    @pyqtSlot()
    def crop_back_area(self):
       
        self.rubberBand.hide() 
        part = 'background'
        self.crop_areas(part)
    
    
    def crop_areas(self, part):
    
        self.croped_part = part            
       
        
        if self.croped_part == 'study':
            self.croped_study_area = 1
            self.start_x_study = self.start_x
            self.start_y_study = self.start_y
            self.end_x_study = self.end_x
            self.end_y_study = self.end_y
        else:
            self.croped_background = 1
            self.start_x_background = self.start_x
            self.start_y_background = self.start_y
            self.end_x_background = self.end_x
            self.end_y_background = self.end_y
       
        
        self.compose_pictures()



    @pyqtSlot()
    def on_click_set_init_frame(self):
        self.initial_frame_edit.setText(str(self.number))
        self.init_frame = self.number
        


    @pyqtSlot()
    def on_click_set_final_frame(self):
        self.final_frame_edit.setText(str(self.number))
        self.final_frame = self.number


    



    def back_sub_type_choose(self, state):
                     
        pass


    

        
        
  
    
    def head_choose(self, state):
        self.head_choosen = 1
             
        if state.text() == 'Head to left' and state.isChecked() == True:
            self.left_checked.setChecked(True)
            self.right_checked.setChecked(False)
            self.head_position = 'left'

        if state.text() == 'Head to left' and state.isChecked() == False:
            self.left_checked.setChecked(False)
            self.right_checked.setChecked(True)
            self.head_position = 'right'

            
            
        if state.text() == 'Head to right' and state.isChecked() == True:
            self.right_checked.setChecked(True)
            self.left_checked.setChecked(False)
            self.head_position = 'right'

        if state.text() == 'Head to right' and state.isChecked() == False:
            self.right_checked.setChecked(False)
            self.left_checked.setChecked(True)
            self.head_position = 'left'

        #print(self.head_position)      



    def function_stop(self):
        self.button_stop.hide()
        self.button_play.show()
        self.button_previous.setEnabled(True)
        #self.button_next.setEnabled(True)
        self.button_first.setEnabled(False)
        self.button_first.setEnabled(True)                      
        self.stop = 1
        #print('stop clicked')
        

    def function_play(self):

        #self.number += 1
        self.combo_play.setEnabled(False)        
        self.button_stop.show()
        self.button_play.hide()
        self.button_previous.setEnabled(False)
        self.button_next.setEnabled(False) 
        self.button_first.setEnabled(False)
        self.button_first.show()

        '''if self.step == 1:
            self.wait_time = self.combo_play.currentText()
        if self.step == 2:
            self.wait_time = self.combo_play_velocity.currentText()'''


        self.wait_time = self.combo_play.currentText()

        if (self.wait_time == '60 fps'):
            self.wait_time_value = 0.017
        elif (self.wait_time == '30 fps'):
            self.wait_time_value = 0.033
        else:
            self.wait_time_value = 0.1

        for f in range(self.number, self.number_files):
            self.slider_video.setEnabled(False)
            self.number += 1            
            self.advance_block = 1
            self.button_play.hide()
            self.button_next.setEnabled(False) 
            #self.compose_pictures()
            #print(self.number)

            self.compose_pictures()
            self.y.append(self.angle)
            self.x.append(self.number)            
            if self.step == 2 or self.step == 3:
                if len(self.y) > 0:
                    self.update_plot_data()     



            time.sleep(self.wait_time_value)            
            self.L1.setText('Frame:\n' + str(self.number) + ' of ' + str(self.number_files -1))
            self.slider_video.setValue(self.number)
            #print(self.number_files)
            if (self.number >= self.number_files-1): 
                self.button_stop.hide()
                self.button_play.hide()
                self.button_play_again.show()
                self.button_previous.setEnabled(True)
                self.button_first.setEnabled(True)       
                self.stop = 0
                self.button_next.setEnabled(False)
                #print('igual')
                break

            if self.stop == 1:
                self.stop = 0
                self.button_play.show()
                self.combo_play.setEnabled(True)
                self.button_next.setEnabled(True)                
                break
            

            QApplication.processEvents()
        self.slider_video.setEnabled(True)
        self.advance_block = 0 
        self.combo_play.setEnabled(True)  




    def function_play_again(self):
        self.number = 0        
        self.button_play_again.hide()
        self.x = []
        self.y = []       
        self.function_play()
        
        
            
        
    def onChanged_play(self):
        pass
       


  
    def restart_app(self):        
        #self.restart_application()
        
        self.step = 0
        self.head_choosen = 0
        self.rubberBand.hide()
        self.setFixedSize(660,400)        
        self.start_app()
        self.csv_saved = 0
        
    
    @pyqtSlot()
    def function_previous_step(self):
        print('actual step = ' + str(self.step))

        if self.step == 3:

            self.slider_pan.show()
            self.pan_label.show() 


            #print('voltou mesmo?')
            self.step=2
            print('actual step = ' + str(self.step))
            self.button_save.hide()
            self.bar.hide()
            self.label_bar.hide()
            self.resulting_label.hide()
            self.bar.setValue(0)
            self.setFixedSize(660,650)

            self.button_unselect_model.hide()

            self.graphWidget.show()

            self.label_slider_title.hide()
            self.slider.hide()
            self.label_slider.hide()

            self.x = []
            self.y = []   


            self.button_build_report.hide()    
         

            self.button_stop.hide()
            self.button_play.show()
            self.button_play_again.hide()
            self.button_previous.show()
            self.button_first.show()         
            self.button_next.show()

            self.number=0

            self.slider_video.show()

            self.label_slider_minimum_title.hide()
            self.label_slider_maximum_title.hide()
            self.label_slider_minimum.hide()
            self.label_slider_maximum.hide()

            self.button_select_back.hide()
            self.button_select.hide()
            self.button_select_model.hide()
            self.button_unselect_model.hide()
            self.graphWidget.show()           
            self.slider_minimun_size.hide()
            self.slider_maximum_size.hide()
            self.slider.hide()
            self.label_slider_title.hide()
            self.label_slider_minimum_title.hide()
            self.label_slider_maximum_title.hide()
            self.label_slider.hide()
            self.label_slider_minimum.hide()
            self.label_slider_maximum.hide()                
            self.combo_play.show()
            self.how_many_Label.show()
            self.how_many_edit.show()

            self.button_abort_counting.hide()

            self.button_init_frame.show()
            self.button_final_frame.show()
            self.initial_frame_edit.show()

            self.final_frame_edit.show() 

            self.button_next_step.setText('Start Counting')


                                   
            self.slider_pan.show()
            self.pan_label.show() 

            self.label.show()
            self.L1.show()

            self.set_mask()
            self.angle_calculation_direct()

            self.buttons_lines()    # call buttons for choosing lines positions

           

            self.slider_head.setValue(int(self.min_head_position*100))
            self.slider_hub.setValue(int(self.hub_position*100))
            self.slider_tail.setValue(int(self.max_tail_position))

            #self.button_next_step.setText('Next Step')
            self.button_next_step.show() 




        elif self.step==2:

            ############   set the mask step
            self.step=1
            print('actual step = ' + str(self.step))
            

            #self.slider_background.show()
            self.slider_thr.show() 


            
            self.image_back.show()
            self.extract_back_message.show()
            if self.back_sub_type == 'image':
                self.button_send_background.show()





            

            #self.fish_model = np.array([1])
           
            #print('aqui')
            #print(self.values_color)

            self.setFixedSize(660,640) 
           
            self.slider_tail.hide()            
            #self.show_secondary_buttons()

            #print(self.fish_model.shape[0])

            if self.fish_model.shape[0] > 1:
                self.button_select_model.hide()
                self.button_unselect_model.show()
                #print('delete button')
            else:
                self.button_select_model.show()
                self.button_unselect_model.hide()

            self.masking_step()
            #self.label.clear()
            #self.set_mask()
            self.graphWidget.hide()   


            self.slider_hub.hide()
            self.tail_label.hide()
            self.slider_head.hide()
            self.head_label.hide()
            self.slider_tail.hide()
            self.hub_label.hide()


            self.button_init_frame.show()
            self.button_final_frame.show()
            self.initial_frame_edit.show()
            self.final_frame_edit.show()
            self.button_final_frame.show()
            self.button_init_frame.show()

            self.compose_pictures()
                                   
            self.slider_pan.hide()
            self.pan_label.hide()
            self.button_next_step.setText('Next Step')

            self.button_abort_counting.hide() 


        elif self.step==1:
            ############set background and fish position step            
            self.step=0 
            print('actual step = ' + str(self.step))


            
            self.image_back.hide()
            self.extract_back_message.hide()
            self.button_send_background.hide()

            
            self.button_save_back_pic.show() 


            self.label_fish_specie.show()
            self.edit_fish_specie.show()
            self.label_fish_id.show()
            self.edit_fish_id.show()
            #########################
            self.button_previous_step.hide()
            self.button_select_back.show()
            self.button_select.show()

            self.slider.hide()
            self.label_slider_title.hide()
            self.label_slider.hide()

            self.button_select_model.hide()
            self.slider_maximum_size.deleteLater()
            self.slider_minimun_size.deleteLater()
            self.slider.deleteLater()
            
            self.label_slider_minimum.deleteLater()
            self.label_slider_maximum.deleteLater()
            self.label_slider.deleteLater()
            self.label_slider_title.deleteLater()
            self.label_slider_minimum_title.deleteLater()
            self.label_slider_maximum_title.deleteLater()

            
            #self.slider_background.hide()
            self.slider_thr.hide()

            
            self.combo_play.show()


            

            

            


            #self.fish_pre_model = np.array([1])
            #self.fish_model = np.array([1])
            self.setFixedSize(660,600)
            
            self.compose_pictures()       



    @pyqtSlot()
    def function_next_step(self):
                     

        if self.step == 0:
            if (self.croped_background == 0 or self.croped_study_area == 0):
                QMessageBox.about(self, "Alert", "You need to select Study Area and Fish Position to advance to next step")   
            else:

                
                ############# mask step (step 1)
                self.step = 1 
                print('actual step = ' + str(self.step))
                self.setFixedSize(660,640)


                
                self.image_back.show()
                self.extract_back_message.show()
                if self.back_sub_type == 'image':
                    self.button_send_background.show()


                self.label_fish_specie.hide()
                self.label_fish_id.hide()
                self.edit_fish_id.hide()
                self.edit_fish_specie.hide()
                self.button_save_back_pic.hide()
                self.button_select_back.hide()
                self.button_select.hide()
                self.left_checked.hide()
                self.right_checked.hide()
                self.button_previous_step.show()                                   
                self.fish_size = (self.end_y_background - self.start_y_background) * (self.end_x_background - self.start_x_background)
                #print('fish size')
                #print(self.fish_size)

                self.button_init_frame.show()
                self.button_final_frame.show()
                self.initial_frame_edit.show()
                self.final_frame_edit.show()
            
                self.show_secondary_buttons()               
                self.masking_step()
                self.set_mask()


                   
                

        elif self.step == 1:

            #print(self.back_sub_type)
            #print(self.img_back)

            if self.back_sub_type == 'image' and self.img_back == '':
                QMessageBox.about(self, "Alert", "You did not upload any background image.\nYou need to choose one before take any action, or just uncheck the \"Image Subtraction\" box")
                    
           

            if (self.fish_model.shape[0] <= 1):
                QMessageBox.about(self, "Alert", "You need to select a model fish silhoute first.\nIt is a critical step to the system accuracy.\nPlease read the guidelines to choose a good model.")   
            else:

                ##### count angle step ##########      ( step 2)

                #print('avançou')
                self.step = 2

                self.button_send_background.hide() 
               
                self.image_back.hide()
                self.extract_back_message.hide()                
                self.slider_background.hide()

                print('actual step = ' + str(self.step))
                self.setFixedSize(660,690)
                self.move(350, 0)
                self.button_select_back.hide()
                self.button_select.hide()
                self.button_select_model.hide()
                self.button_unselect_model.hide() 


                self.graphWidget.show()  

                #self.slider_background.hide()
                self.slider_thr.hide()      

               
                self.slider_minimun_size.hide()
                self.slider_maximum_size.hide()
                self.slider.hide()
                self.label_slider_title.hide()                
                self.label_slider_minimum_title.hide()
                self.label_slider_maximum_title.hide()
                self.label_slider.hide()
                self.label_slider_minimum.hide()
                self.label_slider_maximum.hide()                
                self.combo_play.show()

                self.button_init_frame.show()
                self.button_final_frame.show()
                self.initial_frame_edit.show()
                self.final_frame_edit.show()               
                                      
                self.slider_pan.show()
                self.pan_label.show() 

                self.set_mask()
                self.angle_calculation_direct()

                self.buttons_lines()    # call buttons for choosing lines positions

               

                self.slider_head.setValue(int(self.min_head_position*100))
                self.slider_hub.setValue(int(self.hub_position*100))
                self.slider_tail.setValue(int(self.max_tail_position))

                self.button_next_step.setText('Start Counting') 


        elif self.step == 2:
            self.step=3
            print('actual step = ' + str(self.step))           

            self.graphWidget.show()
            self.button_init_frame.hide()
            self.button_final_frame.hide()
            self.initial_frame_edit.hide()
            self.final_frame_edit.hide()               
                                  
            #self.slider_pan.hide()
            #self.pan_label.hide() 

            self.button_stop.hide()
            self.button_play.hide()
            self.button_play_again.hide()
            self.button_previous.hide()
            self.button_first.hide()         
            self.button_next.hide()

            self.slider_video.hide()
            self.slider_hub.hide()
            self.slider_tail.hide()
            self.slider_head.hide()
            self.label.hide()

            self.hub_label.hide()
            self.head_label.hide()
            self.tail_label.hide()
            self.button_previous_step.setEnabled(False)
            self.button_restart.setEnabled(False)
            

            self.how_many_Label.hide()

            self.button_next_step.hide()
            self.label_slider.hide()
            self.combo_play.hide()

            self.L1.hide()

            self.how_many_edit.hide()

            self.button_abort_counting.show()

            self.slider_pan.hide()
            self.pan_label.hide() 

            self.start_countig()




    def act_sub(self):
        #print(self.back_sub_type)
        if self.back_sub_type == 'numbers':
            #self.method = 'numbers'
            self.slider_background.hide()
            self.slider.show()
            self.label_slider_title.show()
            self.label_slider.show()
            self.repaint()
        else:
            #self.method = 'image'
            fname_type = QFileDialog.getOpenFileName(self, 'Open file', 
         'c:\\',"Image files (*.jpg)")            

            self.img_back = fname_type[0]

            if self.img_back != '':                
                img_back = cv2.imread(self.img_back)                
                cv2.imshow('Background Image', img_back)
                self.set_mask()
                self.slider_background.show()
                self.slider.hide()
                self.label_slider_title.hide()
                self.label_slider.hide()
                self.repaint()
            else:
                self.back_sub_type = 'numbers'
                
                self.image_back.setChecked(False)
                self.slider_background.hide()
                self.slider.show()
                self.label_slider_title.show()
                self.label_slider.show()



         


    def buttons_lines(self):




        self.slider_head = QSlider()
        self.slider_head.setOrientation(Qt.Horizontal)
        self.slider_head.setMinimum(5)
        self.slider_head.setMaximum(80)
        self.slider_head.setTickInterval(1)  
        self.slider_head.setValue(30)

        #self.layout.addWidget(self.slider_head)                     
        self.slider_head.valueChanged.connect(self.function_choose_head)
        self.slider_head.show()

        self.head_label = QLabel(self)
        self.head_label.setText('Head point')
        self.head_label.setToolTip('Choose where is the initial point (the body, excluding the head) of the fish')
        self.head_label.setAlignment(Qt.AlignCenter)       

        

        self.slider_hub = QSlider()
        self.slider_hub.setOrientation(Qt.Horizontal)        
        self.slider_hub.setTickInterval(1)        
        #self.layout.addWidget(self.slider_hub)
        self.slider_hub.setMinimum(20)
        self.slider_hub.setMaximum(80)
        self.slider_hub.setValue(60)                    
        self.slider_hub.valueChanged.connect(self.function_choose_hub)
        self.slider_hub.show()

        self.hub_label = QLabel(self)
        self.hub_label.setText('Hub Point')
        self.hub_label.setToolTip('Choose where is the final point of the body (the initial point of the tail)')
        self.hub_label.setAlignment(Qt.AlignCenter)    



        self.slider_tail = QSlider()
        self.slider_tail.setOrientation(Qt.Horizontal)        
        self.slider_tail.setTickInterval(1)        
        #self.layout.addWidget(self.slider_tail)
        self.slider_tail.setMinimum(0)
        self.slider_tail.setMaximum(640)
        self.slider_tail.setValue(640)                    
        self.slider_tail.valueChanged.connect(self.function_choose_tail)
        self.slider_tail.show()

        self.tail_label = QLabel(self)
        self.tail_label.setText('Final Tail')
        self.tail_label.setToolTip('Choose where is the final point of the tail (Excluding soft parts of the tail, if any)')
        self.tail_label.setAlignment(Qt.AlignCenter)    
 



        self.layout_h12 = QHBoxLayout()
        self.layout_h13 = QHBoxLayout()

        self.layout_h12.addWidget(self.slider_head)
        self.layout_h13.addWidget(self.head_label)
        self.layout_h12.addWidget(self.slider_tail)
        self.layout_h13.addWidget(self.tail_label)
        self.layout_h12.addWidget(self.slider_hub)
        self.layout_h13.addWidget(self.hub_label)

        self.layout.addLayout(self.layout_h12)
        self.layout.addLayout(self.layout_h13)          

        
    def start_countig(self):
        self.heat_map_integral = []
        self.setFixedSize(660,460)
        #self.bar.setGeometry(0, 0, self.frame_count, 30)
        self.bar.setMaximum(self.number_files-1) 
        self.bar.show()
        self.label_bar.show()
        self.resulting_label.show()        
        #print('start couting')

        self.abort_counting = 0
        
        self.y = []
        self.x = []
        self.number = self.init_frame
        #print('init_frame:' + str(self.init_frame))
        if self.final_frame == '':
            self.final_frame = self.number_files -1

        #print('final_frame:' + str(self.final_frame))

        while self.number < self.final_frame:
            #print('self_number1: ' + str(self.number))

            self.set_mask()
            self.angle_calculation_direct()
            self.y.append(self.angle)
            self.x.append(self.number)
            self.update_plot_data()

            self.heat_map_integral.append([self.heat_x, self.heat_y])


            #self.array_for_angle_final = self.array_for_angle[:,0]
            
            
            #print('abort_counting: ' + str(self.abort_counting))
            self.number += 1

            if self.abort_counting == 1:                
                self.y = []
                self.x = []
                self.number=0                
                break
            
            #print('self_number2: ' + str(self.number))

            self.bar.setValue(self.number)
            self.label_bar.setText("Frame " + str(self.number) + " of " + str(self.number_files-1))

            QApplication.processEvents()

        #print('heat_map_integral')
        #print(self.heat_map_integral)

        



        self.button_abort_counting.hide()
        self.button_previous_step.setEnabled(True)
        self.button_restart.setEnabled(True)
        
        if self.abort_counting == 0:
            self.where = "Resulting CSV and TXT files will be saved at:\n" + self.fname[:-5] + '_' + str(self.time_now) + '/angles.csv\n' + self.fname[:-5] + '_' + str(self.time_now) + '/heatmap.txt'
            self.resulting_label.setText(self.where)
            self.button_save.show()

        
        



    @pyqtSlot()
    def function_slider_minimum(self):
        self.size_minimum = self.slider_minimun_size.value()
        self.label_slider_minimum.setText(str(self.size_minimum))
        self.compose_pictures()
  

        
    @pyqtSlot()
    def function_slider_maximum(self):
        self.size_maximum = self.slider_maximum_size.value()
        self.label_slider_maximum.setText(str(self.size_maximum))
        self.compose_pictures()
     
    



    @pyqtSlot()
    def function_slider(self):
        size = self.slider.value()
        self.values_color = self.slider.value()
        self.label_slider.setText(str(size))
        self.compose_pictures()




    @pyqtSlot()
    def function_select_model(self):
        if self.fish_pre_model.shape[0] > 1:
            self.fish_model = self.fish_pre_model
            self.button_unselect_model.show()
            self.button_select_model.hide()            
            self.min_tail = np.amin(self.fish_model, axis=0)[0][0]
            self.max_tail = np.amax(self.fish_model, axis=0)[0][0]
            self.model_lenght = self.max_tail-self.min_tail
            M = cv2.moments(self.fish_model)                   
            self.mass_x_model = int(M["m10"] / M["m00"])            
            self.fish_model_std=self.fish_model[:,0]
            

            self.mass_y_model = int(M["m01"] / M["m00"])
            self.height_array = self.fish_model_std[self.fish_model_std[:, 0] == self.mass_x_model]
            self.model_height_max = np.amax(self.height_array, axis=0)[1]
            self.model_height_min = np.amin(self.height_array, axis=0)[1]
            self.model_height = self.model_height_max - self.model_height_min

            self.model_area = cv2.contourArea(self.fish_model)

           
            
            QMessageBox.about(self, "Alert", "We have set the actual image as a model")            
        else:
            QMessageBox.about(self, "Alert", "It seems you don´t have a good fish silhouet to fit the model.\nIt is a critical step to the system accuracy.\nTry with another frame or make adjustments.\nRead the guidelines to choose a good model.")  




    @pyqtSlot()
    def function_unselect_model(self):
        self.fish_model = np.array([1])       
        self.button_select_model.show()
        self.button_unselect_model.hide()
        QMessageBox.about(self, "Alert", "You have cleared the model, select\nanother one before proceed to the next step")

        

    @pyqtSlot()
    def function_slider_video_pressed(self):
        self.button_play.show()
        #self.set_mask()
        #self.angle_calculation_direct()

    @pyqtSlot()
    def function_slider_video_moved(self):
        frame = self.slider_video.value()
        self.x = []
        self.y = []

    @pyqtSlot()
    def function_slider_video_released(self):
        self.compose_pictures() 
        



    @pyqtSlot()
    def function_slider_video(self):
        self.button_play.setEnabled(True)
        
        frame = self.slider_video.value()
        #print('slider_vide')
        #print(frame)
       #print(self.frame_count)        
        if frame == self.number_files-1:
            frame = self.number_files-1
            self.button_play.setEnabled(False)
        self.number = frame

        #self.compose_pictures()          ###############  problem here

        self.L1.setText('Frame:\n' + str(self.number) + ' of ' + str(self.number_files -1))

        if frame < self.number_files-1:
            self.button_play_again.hide()

            #self.button_play.show()
            if self.advance_block == 0:
                self.button_next.setEnabled(True)

        
        


    @pyqtSlot()
    def function_abort_counting(self):
        self.abort_counting = 1
        self.button_previous_step.setEnabled(True)
        self.button_restart.setEnabled(True)




    def update_plot_data(self):

        self.pan_number = self.slider_pan.value()
        #self.data_line.setData(self.x, self.y)

        #self.graphWidget.setRange(rect=None, xRange=(self.x_i,self.x_f), yRange=None, padding=None, update=True, disableAutoRange=True)

        #print('self.x')
        #print(self.x)



        if self.x:
            init = self.x[-1]
            init = init - self.pan_number
            if init < 0:
                init = 0
            final = self.x[-1] + self.pan_number


            self.x_i = self.number+(self.pan_number*-1)
            self.x_f = self.number+(self.pan_number)
        else:
            self.x_i = 0
            self.x_f = 0


        

        self.graphWidget.setRange(rect=None, xRange=(self.x_i,self.x_f), yRange=None, padding=None, update=True, disableAutoRange=True)
        #self.graphWidget.setRange(x_i, x_f)


        self.data_line.setData(self.x, self.y)  # Update the data.




    def function_slider_pan(self):
        pass
        

        
        #self.graphWidget.clear()

        pen = pg.mkPen(color=(0, 255, 0))
        self.data_line =  self.graphWidget.plot(self.x, self.y, pen=pen, width=2)
        

        
        
    '''def open_cvs(self):
        self.build.report()'''   
    

    def save_csv(self):

        df = pd.DataFrame(self.y)        
        df.columns = ["angle"]
        df['fps'] = int(self.fps) 
        #df.fps.append(300)
        #df.angle.append(self.y)
        df.to_csv(self.fname[:-5] + '_' + str(self.time_now) + '/angles.csv', index=False)  


        #savetxt(self.fname[:-5] + '_' + str(self.time_now) + '/angles.csv', self.y, delimiter=',')
        #savetxt('/home/concursoadapta/myprojectdir/csvs/' + folderprimeiro + '/' + folderprimeiro + '.csv', list_row, delimiter=',', fmt='%s')
        self.csv_saved = 1

        self.button_build_report.show()
        QMessageBox.about(self, "Alert", "Your csv and txt file with angles and heatmap were saved.\nYou can generate a report now or later")
        self.button_build_report.show()

    def build_report_step1(self):


        self.button_generate_report = QPushButton('Generate Report', self)
        self.button_generate_report.setToolTip('Choose the report options')        
        self.button_generate_report.clicked.connect(self.generate_report)
        self.layout.addWidget(self.button_generate_report)
        self.button_generate_report.show()

        self.label_generate_report = QLabel('Report will be saved at: ' + self.fname[:-5] + '_' + str(self.time_now) + '/report.pdf', self)
        self.name_pdf = self.fname[:-5] + '_' + str(self.time_now) + '/report.pdf'

        self.layout.addWidget(self.label_generate_report)
        self.label_generate_report.show()

        self.button_send_background.hide() 


        #self.button_generate_report_change = QLabel('Change Destination', self)
        self.layout.addWidget(self.label_generate_report)
        self.label_generate_report.show()

        

             
        self.graphWidget.deleteLater()
        self.slider_pan.deleteLater()
        self.button_previous_step.deleteLater()
        self.bar.deleteLater()
        self.label_bar.deleteLater()
        self.pan_label.deleteLater()
        self.resulting_label.deleteLater()
        self.button_build_report.deleteLater()
        self.button_save.deleteLater()
        self.setFixedSize(660,200)

        self.build_report_step2()

        

    def build_report_step2(self):


        if self.csv_saved == 0:
            self.button_generate_report = QPushButton('Generate Report', self)
            self.button_generate_report.setToolTip('Choose the report options')        
            self.button_generate_report.clicked.connect(self.generate_report)
            self.layout.addWidget(self.button_generate_report)
            self.button_generate_report.show()


            self.label_generate_report = QLabel('', self)
            self.layout.addWidget(self.label_generate_report)
            self.label_generate_report.show()


        self.setFixedSize(660,250)

        self.label.deleteLater()
        self.button_open.deleteLater()
        self.button_open_csv.deleteLater()

        self.button_open_csv = QPushButton('Load angles (CSV file)', self)       
        self.button_open_csv.clicked.connect(self.get_file_csv)            
        self.button_open_csv.show()
        self.layout.addWidget(self.button_open_csv)

        self.label_open_csv = QLabel('', self)
        self.layout.addWidget(self.label_open_csv)
        self.label_open_csv.show()
        if self.csv_saved == 1:
            self.label_open_csv.setText(self.fname[:-5] + '_' + str(self.time_now) + '/angles.csv') 
            self.file_csv = self.fname[:-5] + '_' + str(self.time_now) + '/angles.csv'



        self.button_open_txt = QPushButton('Load heatmap (txt file)', self)       
        self.button_open_txt.clicked.connect(self.get_file_txt)            
        self.button_open_txt.show()
        self.layout.addWidget(self.button_open_txt)

        self.label_open_txt = QLabel('', self)
        self.layout.addWidget(self.label_open_txt)
        self.label_open_txt.show()

        '''self.layout_file_csv = QHBoxLayout()
        self.layout_file_csv.addWidget(self.button_open_csv)
        self.layout_file_csv.addWidget(self.label_open_csv)'''

        '''self.layout_file_txt = QHBoxLayout()
        self.layout_file_txt.addWidget(self.button_open_txt)
        self.layout_file_txt.addWidget(self.label_open_txt)'''

        '''self.layout.addLayout(self.layout_file_csv)
        self.layout.addLayout(self.layout_file_txt)'''



        self.combo_heatmap_lines = QComboBox(self)
        self.combo_heatmap_lines.addItem("3")              
        self.combo_heatmap_lines.addItem("6")
        self.combo_heatmap_lines.addItem("9")
        self.combo_heatmap_lines.setToolTip('Number of rows in heatmap')  
        self.combo_heatmap_lines.show()
        self.combo_heatmap_lines.setCurrentIndex(1) 
        self.combo_heatmap_lines.activated[str].connect(self.onChanged_combo_heatmap_lines)

        self.label_heatmap_lines = QLabel('Number of rows of heatmap', self)
        self.label_heatmap_lines.show()

        self.combo_heatmap_collumns = QComboBox(self)
        self.combo_heatmap_collumns.addItem("3")              
        self.combo_heatmap_collumns.addItem("6")
        self.combo_heatmap_collumns.addItem("9")
        self.combo_heatmap_collumns.setToolTip('Number of collums in heatmap')  
        self.combo_heatmap_collumns.show()
        self.combo_heatmap_collumns.setCurrentIndex(2) 
        self.combo_heatmap_collumns.activated[str].connect(self.onChanged_combo_heatmap_collumns)

        self.label_heatmap_collumns = QLabel('Number of collumns of heatmap', self)
        self.label_heatmap_collumns.show()

        self.combo_histogram_collumns = QComboBox(self)
        self.combo_histogram_collumns.addItem("3")              
        self.combo_histogram_collumns.addItem("6")
        self.combo_histogram_collumns.addItem("10")
        self.combo_histogram_collumns.setToolTip('Number of collums in heatmap')  
        self.combo_histogram_collumns.show()
        self.combo_histogram_collumns.setCurrentIndex(2) 
        self.combo_histogram_collumns.activated[str].connect(self.onChanged_combo_histogram_collumns)

        self.label_histogram_collumns = QLabel('Number of collumns of histogram', self)
        self.label_histogram_collumns.show()
        

        self.combo_box_final_labels = QHBoxLayout()
        self.combo_box_final_labels.addWidget(self.label_heatmap_lines)
        self.combo_box_final_labels.addWidget(self.label_heatmap_collumns)
        self.combo_box_final_labels.addWidget(self.label_histogram_collumns)
        self.layout.addLayout(self.combo_box_final_labels)
        #self.combo_box_final_labels.setAlignment(Qt.AlignCenter)  




        self.combo_box_final = QHBoxLayout()
        self.combo_box_final.addWidget(self.combo_heatmap_lines)
        self.combo_box_final.addWidget(self.combo_heatmap_collumns)
        self.combo_box_final.addWidget(self.combo_histogram_collumns)
        self.layout.addLayout(self.combo_box_final)

        self.heat_wait = QLabel("", self)       
        self.layout.addWidget(self.heat_wait)
        self.heat_wait.show()


    def generate_report(self):

        '''self.heat_wait = QLabel("xxx", self)       
        self.layout.addWidget(self.heat_wait)
        self.heat_wait.show()''' 

        try:
            self.tabs.deleteLater() #check if tabs exist, then delete it           
        except:
            print('no_tabs yet')

        get_csv = self.label_open_csv.text()
        get_txt = self.label_open_txt.text()

        if get_csv == '' and get_txt == '':

            QMessageBox.about(self, "Alert", "You need to choose a CVS and/or a TXT file first")

        else:

            #self.name_pdf = QFileDialog.getSaveFileName(self, 'Where to save report',
            #'c:\\',"PDF files (*.pdf)")

            #self.name_pdf = self.name_pdf[0]
            

            if self.name_pdf != '':
                self.file_pdf = open(self.name_pdf,'w')
                self.label_generate_report.setText('PDF saved at ' + self.name_pdf)  
                
                self.run_report() 
            else:
                #QMessageBox.about(self, "Alert", "You need to choose a folder to save the report")                
                self.name_pdf = QFileDialog.getSaveFileName(self, 'Where to save report',
                'c:\\',"PDF files (*.pdf)")
                self.name_pdf = self.name_pdf[0]
                if self.name_pdf != '':
                    self.label_generate_report.setText('PDF saved at ' + self.name_pdf) 
                    self.run_report()                  
                

    def get_file_csv(self):

        self.file_csv = QFileDialog.getOpenFileName(self, 'Open CSV file', 
         'c:\\',"CSV files (*.csv)")
        if self.file_csv != '':
            self.label_open_csv.setText(self.file_csv[0])
      


    def get_file_txt(self):

        self.file_txt = QFileDialog.getOpenFileName(self, 'Open TXT file', 
         'c:\\',"Text files (*.txt)")
        if self.file_txt != '':
            self.label_open_txt.setText(self.file_txt[0])

       

    def onChanged_combo_heatmap_lines(self):
        self.choice_heatmap_lines = self.combo_heatmap_lines.currentText()
        print(self.choice_heatmap_lines)

    def onChanged_combo_heatmap_collumns(self):
        self.choice_heatmap_collumns = self.combo_heatmap_collumns.currentText()
        print(self.choice_heatmap_collumns)

    def onChanged_combo_histogram_collumns(self):
        self.choice_histogram_collumns = self.combo_histogram_collumns.currentText()
        print(self.choice_histogram_collumns)


    def run_report(self):




        self.move(350, 0)

        if self.csv_saved == 1:
            fps10 = pd.read_csv(self.fname[:-5] + '_' + str(self.time_now) + '/angles.csv')
        else:
            fps10 = pd.read_csv(self.file_csv[0])        
        lista_original_filter = fps10
        lista_original_filter['angle'].loc[(fps10['angle'].isnull()) ] = 0
       
        lista_original_filter['angle'].loc[(fps10['angle'] > 40)] = 0
        lista_original_filter['angle'].loc[(fps10['angle'] < -40)] = 0
        #lista_original_filter_angles = lista_original_filter['angle']


        
        self.no_zeros_ini = lista_original_filter[lista_original_filter !=0.00]
        self.no_zeros = self.no_zeros_ini.dropna()

        self.no_zeros.reset_index(drop=True, inplace=True)

       

        
        self.fps = fps10['fps'][0]
        self.totaltime = fps10.angle.count() / self.fps
        count_orig_row = len(fps10)  
        self.count_final_row = self.no_zeros.shape[0]
        quality = self.count_final_row / count_orig_row * 100
        self.quality = round(quality,2)





        
        
        #chamas os graficos especificos
        self.peaks()
        
             
    
    

    def peaks(self):          

        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        

        dataset = self.no_zeros
        #plt.title("Caudal beating rate") #The title of our plot
        #plt.plot(dataset.angle) #Draw the plot object

        import numpy as np
        import math

        #Calculate moving average with 0.75s in both directions, then append do dataset
        hrw = self.mean_points #One-sided window size, as proportion of the sampling frequency


        mov_avg = dataset.angle.rolling(int(hrw*self.fps)).mean() #Calculate moving average
        
        #Impute where moving average function returns NaN, which is the beginning of the signal where x hrw


        avg = (np.mean(abs(dataset.angle)))        
        avg_hr = 0
        mov_avg = [avg_hr if 1==1 else x for x in mov_avg]
              
        
        window1 = []
        peaklist1 = []
        fps_skip = []
        window2 = []
        peaklist2 = []
        listpos = 0 #We use a counter to move over the different data columns
        comum=0
        for datapoint in dataset.angle:
          
            if (datapoint > 0): 
                window1.append(datapoint)
                                
            else: 
                if len(window1) >= 2:
                    #maximum = max(window1)
                    beatposition = listpos - len(window1) + (window1.index(max(window1))) #Notate the position on the X-axis
                    if max(window1) > (avg*0.3):
                        if comum > max(window1):
                            pass
                        elif comum > 0 and comum < max(window1):
                            peaklist1[-1] = beatposition
                            comum = max(window1)
                        else:                                
                            peaklist1.append(beatposition)
                            comum=max(window1)
                            fps_skip.append(len(window1))
                        
                    window1 = [] 
                    
                else:
                    window1 = [] #Clear marked ROI
                    
                      
                
        


        
        
            
            if (datapoint < 0): 
                window2.append(datapoint)
                                  
            else: 
                if len(window2) >= 2:
                    #minimum = min(window2)
                    beatposition = listpos - len(window2) + (window2.index(min(window2))) #Notate the position on the X-axis
                    if min(window2) < ((avg*-1)*0.3):
                        if comum < min(window2):
                            pass
                        elif comum < 0 and comum > min(window2):
                            peaklist2[-1] = beatposition
                            comum = max(window2)
                        else:                                
                            peaklist2.append(beatposition)
                            comum=min(window2)
                            fps_skip.append(len(window2))
                        
                    window2 = [] 
                    
                else:
                    window2 = [] #Clear marked ROI
                    
            listpos += 1 
                
                
                               
           
                    
        fps_skip_total = np.mean(fps_skip)
        
        if fps_skip_total > 4:               
            
           
            skip_state = ' The frame rate applyed seems ok, with about ' + str(round(fps_skip_total, 2)) + ' in each peak'

        else:
           
            skip_state = ' Be carefull: the frame rate applyed seems low, with about ' + str(round(fps_skip_total,2)) + ' in each peak'
          
       
        
        
        
    
        ybeat1 = [dataset.angle[x] for x in peaklist1] #Get the y-value of all peaks for plotting purposes    
        ybeat2 = [dataset.angle[x] for x in peaklist2] #Get the y-value of all peaks for plotting purposes 
        
      
        peaklist_t = peaklist1 + peaklist2
        ybeat_t = ybeat1 + ybeat2
        '''f = open('/home/concursoadapta/myprojectdir/uploads/' + certnum + '/' + certnum[30:] + '_peaks.csv', 'w')
        for i in range(len(peaklist_t)):
            f.write("{},{}\n".format(peaklist_t[i], ybeat_t[i]))
        f.close()'''

        amp = np.mean(ybeat1)
        soma = np.sum(ybeat1)
        ampx = round(amp,2) * 2
        ms_dist1 = 0
        RR_list = []
        cnt = 0
        while (cnt < (len(peaklist1)-1)):
            RR_interval = (peaklist1[cnt+1] - peaklist1[cnt]) #Calculate distance between beats in # of samples    
            ms_dist = ((RR_interval / self.fps) * 1000.0) #Convert sample distances to ms distances
            RR_list.append(ms_dist) #Append to list
            cnt += 1

        amp2 = np.mean(ybeat2)
        soma2 = np.sum(ybeat2)
        ampy = round(amp2,2) * 2
        ampy = abs(ampy)
        ampy_neg = ampy * -1
        ms_dist2 = 0
        RR_list2 = []
        cnt = 0
        while (cnt < (len(peaklist2)-1)):
            RR_interval2 = (peaklist2[cnt+1] - peaklist2[cnt]) #Calculate distance between beats in # of samples    
            ms_dist2 = ((RR_interval2 / self.fps) * 1000.0) #Convert sample distances to ms distances
            RR_list2.append(ms_dist2) #Append to list
            cnt += 1   


        amp_medio1 = [ampx]
        amp_medio2 = [ampy] 

        amp_medio1.extend(amp_medio2)
     

        amp_final = statistics.mean(amp_medio1)    


        if not any ((RR_list, RR_list)):
            print('não lista')
            bps = 0
            ms_dist = 'indetectable'
        else:
            #print(ms_dist)
            #print(ms_dist2)

           
            ms_dist = [ms_dist]
            ms_dist.append(ms_dist2)        
            ms_dist = statistics.mean(ms_dist)
            #print(ms_dist)

            RR_list.extend(RR_list2)
            
            
          

            bps = 60000 / np.mean(RR_list) /60 #60000 ms (1 minute) / average R-R interval of signal
       
        self.picos = (len(RR_list)/2) + 1
      
        if ms_dist != 'indetectable':
            value = ms_dist/1000
            value2 = str(round(value,2))
        else:
            value2 = 'indetectable'
        
        #frequancia da batida
        duration = self.totaltime / self.picos
        value22 = 1/duration
        
        
        
        texto1 = "Beating frequency [Hz]: " + str(round(value22,2))
        texto2 = 'Average of tail´s movement amplitude [degrees]: ' + str(round(amp_final,2))
        texto2_1 = 'Sinoids detected: ' + str(self.picos) 
        texto3 = 'FPS detected: ' + str(self.fps) 
   


        #val = np.float32(self.totaltime)
        #totaltime = val.item()

      
        
        #quit()
        
        text_6 = 'Beating cicle duration [seconds]: ' + str(round(duration, 2)) + '\n*Peaks with no spots were considered outliers and no computed.'
        quality_message = 'Fish detection indice in the frames: '  + str(self.quality) + ';\n' + texto3
       

        health = quality_message    


        #print('dataset.angle')
        #print(dataset.angle)
        
        plt.title(skip_state + '\n\n Detected angle x frame number')        
        plt.xlim(0,self.count_final_row)
        plt.plot(dataset.angle, alpha=0.5, color='blue', label="raw signal") #Plot semi-transparent HR
        plt.plot(mov_avg, color ='green') #Plot moving average
        plt.scatter(peaklist1, ybeat1, color='red', label='_nolegend_') #Plot detected peaks positive   %bps
        plt.scatter(peaklist2, ybeat2, color='orange', label='_nolegend_') #Plot detected peaks negative

        plt.legend(loc=4, framealpha=0.6)
        
        plt.xlabel('Absolute number of video frames \n\n Results:\n' + texto1 + '\n' + texto2 + '\n' + texto2_1 + '\n' + text_6 + '\n' + health)
       
        
        plt.ylabel('Tail angle (degrees)')

        #fig = plt.figure()
        #fig.savefig(self.name_pdf)
       
        '''plt.savefig(self.name_pdf, dpi=None, figsize=(2, 1.5), clear=True, facecolor='w', edgecolor='w',
                    papertype=None, 
                    transparent=False, bbox_inches='tight', pad_inches=0.6,
                    metadata=None)''' #####################################save figure but is rendering a error.

        import random


        self.setFixedSize(660,690)









        self.figure = plt.figure()

        
        ############### colocar a figura em um canvas específico
        
        #self.layout.addWidget(self.canvas)

        #self.layout.addWidget(self.toolbar)

        data = [random.random() for i in range(10)]

        # instead of ax.hold(False)
        self.figure.clear()

        # create an axis
        ax = self.figure.add_subplot(111)

        ax.set_title('Detected angle x frame number')  

        # discards the old graph
        # ax.hold(False) # deprecated, see above

        # plot data
        #ax.plot(data, '*-')

        #ax.title(skip_state + '\n\n Detected angle x frame number')        
        #ax.xlim(0,self.count_final_row)
        ax.plot(dataset.angle, alpha=0.5, color='blue', label="raw signal") #Plot semi-transparent HR
        ax.plot(mov_avg, color ='green') #Plot moving average
        ax.scatter(peaklist1, ybeat1, color='red', label='_nolegend_') #Plot detected peaks positive   %bps
        ax.scatter(peaklist2, ybeat2, color='orange', label='_nolegend_') #Plot detected peaks negative

        ax.legend(loc=4, framealpha=0.6)


        ########################################aquijjjjjjjjjjjjjj



        self.canvas_peaks = FigureCanvas(self.figure)

        self.toolbar = NavigationToolbar(self.canvas_peaks, self)
        
        # Create first tab
        self.tab1 = QWidget()
        self.tab1.layout = QVBoxLayout(self)        
        self.tab1.layout.addWidget(self.canvas_peaks)
        self.tab1.layout.addWidget(self.toolbar)

        results = texto1 + '; ' + texto2 + '; ' + texto2_1 + '\n' + 'Beating cicle duration [seconds]: ' + str(round(duration, 2)) + '; ' + 'Fish detection indice in the frames: '  + str(self.quality) + '; ' + texto3 + '\n*Eventual peaks with no spots were considered outliers and are not shown.\n' + skip_state


        
        self.label_result_report = QLabel(results, self)      
        self.tab1.layout.addWidget(self.label_result_report)

        self.tab1.setLayout(self.tab1.layout)


        

        #self.insert_tabs_in_layout()



        
        self.heat_map()
            


    def heat_map(self):


        
        self.heat_map_total = []
        indice_inicial = 0


        self.heat_wait.setText('Adding frames to heatmap...')

        


     

        
        

        for n in self.heat_map_integral:
            #print('n')
            #print(n)
            x_center = n[0]
            y_center = n[1]

            '''print('x_center')
            print(x_center)
            print('y_center')
            print(y_center)'''
            array_heat_b_zeros_template = np.zeros((640,360))
            array_heat_b_zeros = array_heat_b_zeros_template[self.start_x_study:self.end_x_study, self.start_y_study:self.end_y_study]
            
            #print('array_heat_b_zeros.shape')
            #print(array_heat_b_zeros.shape)
            #print(array_heat_b_zeros)
            #array_heat_b_zeros = array_heat_b_zeros_zero[0:360,0:640] #########!!!!!!!!!!!!!!!!!!!!!

            #array_heat_b_zeros = array_heat_b_zeros_zero[scaled_y:(scaled_y + scaled_height),scaled_x:(scaled_x + scaled_width)]
            #array_heat_b_zeros_x = array_heat_b_zeros_zero[self.start_y_study:self.end_y_study, self.start_x_study:self.end_x_study]
            
            array_heat_b_zeros[x_center, y_center]=1

            '''print(array_heat_b_zeros)
            print(array_heat_b_zeros.shape)'''

            '''print('array_heat_b_zeros')
            print(array_heat_b_zeros)'''
            '''if 1 in array_heat_b_zeros:
                print('tem')
            else:
                print('não tem')'''


            self.lines = self.choice_heatmap_lines  ###################################################################################
            self.col = self.choice_heatmap_collumns
            total  = int(self.lines) * int(self.col)

            '''print('linhas e colunas')
            print(self.lines)
            print(self.col)'''


            #corrige for non evenly by the asked bumber colum or row
            par_impar = 1
            while (array_heat_b_zeros.shape[0] % int(self.lines) != 0):   

                if par_impar % 2 == 0:
                    array_heat_b_zeros = np.append(array_heat_b_zeros,np.zeros([1, len(array_heat_b_zeros[1])]),0) #adicionar linhas de 0
                else:
                    array_heat_b_zeros = np.concatenate((np.zeros([1, len(array_heat_b_zeros[1])]), array_heat_b_zeros), axis=0)
                par_impar = par_impar + 1

            par_impar = 1    
            while (array_heat_b_zeros.shape[1] % int(self.col) != 0):

                if par_impar % 2 == 0:
                    array_heat_b_zeros = np.append(array_heat_b_zeros,np.zeros([array_heat_b_zeros.shape[0],1]),1) #adicionar colunas de 0
                else:

                    array_heat_b_zeros = np.concatenate((np.zeros([array_heat_b_zeros.shape[0],1]), array_heat_b_zeros), axis=1)
                par_impar = par_impar + 1


            final_lines = array_heat_b_zeros.shape[0]/int(self.lines)

            final_col = array_heat_b_zeros.shape[1]/int(self.col)

            def split(arr, nrows, ncols):

                h, w = arr.shape
                assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
                assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
                return (arr.reshape(h//nrows, nrows, -1, ncols)
                           .swapaxes(1,2)
                           .reshape(-1, nrows, ncols))

            array_final = split(array_heat_b_zeros, int(final_lines), int(final_col))

            array_flatten = array_final.flatten()

            lista = list(array_flatten)

            divisor = len(lista)/total

            temp = [sum(lista[i:i+int(divisor)]) for i in range(1, len(lista), int(divisor))]

            for i in range(0, len(temp)): 
                temp[i] = int(temp[i])

            chunks = [temp[x:x+int(self.col)] for x in range(0, len(temp), int(self.col))]

            if indice_inicial == 0:               

                self.heat_map_total = chunks
            else: 

                self.heat_map_total=geek.add(self.heat_map_total, chunks)
                print('adding chunks to heat map: ' + str(indice_inicial))
              



            
           
            '''print('self.heat_map_total')
            print(self.heat_map_total)'''
            indice_inicial = indice_inicial + 1

        self.heat_wait.hide()    
            
            
        self.heat_map_start()  



        '''file.write(file_to_save)
        file.close()'''

    def heat_map_start(self):

        #self.heat_map_total_fliped = np.fliplr(self.heat_map_total)
        #print(self.heat_map_total)
        #print(self.heat_map_total_fliped)

        '''print('self.heat_map_total')
        print(self.heat_map_total)'''

        #lines = self.lines
        #col = self.col
        #total  = self.lines * self.col        
        #ar = loadtxt('/home/concursoadapta/myprojectdir/csvs/' + csvfile + '/' + 'heat_b_array_' + certnum + '.mp4.csv', delimiter=',')     
        fig, ax = plt.subplots(figsize=(7,7)) 
        ax.set_title("Heat map of the fishe´s center of mass (in the selected area)", pad=15) 

        figure_plot = sns.heatmap(self.heat_map_total, ax=ax, annot=True, cbar_kws={"orientation": "horizontal"}, annot_kws={"size": 8}, square=True, linewidths=.1) # , cmap="YlGnBu_r"
        #ax.invert_xaxis() 
        # fix for mpl bug that cuts off top/bottom of seaborn viz
        '''b, t = plt.ylim() # discover the values for bottom and top
        b += 0.5 # Add 0.5 to the bottom
        t -= 0.5 # Subtract 0.5 from the top
        plt.ylim(b, t) # update the ylim(bottom, top) values'''

        
        self.figure = figure_plot.get_figure()
        
        self.canvas_heat = FigureCanvas(self.figure)

        self.tab2 = QWidget()
        self.tab2.layout = QVBoxLayout(self)        
        self.tab2.layout.addWidget(self.canvas_heat)      
        self.tab2.setLayout(self.tab2.layout)

        

        
        #self.aceleration()

        
        self.histogram_of_raw()


    def histogram_of_raw(self):

        fig, ax = plt.subplots(figsize=(6,6)) 
        ax.set_title("Frequency Histogram of the caudal angle", pad=15) 
        figure_histo_raw = sns.distplot(self.no_zeros['angle'], axlabel='Angle', bins=int(self.choice_histogram_collumns), kde=False, rug=True, color="m", ax=ax);        
        plt.subplots_adjust(top = 0.9)
        plt.subplots_adjust(bottom = 0.2)
        self.figure = figure_histo_raw.get_figure()
        
        self.canvas_histo_raw = FigureCanvas(self.figure)

        self.tab3 = QWidget()
        self.tab3.layout = QVBoxLayout(self)        
        self.tab3.layout.addWidget(self.canvas_histo_raw)      
        self.tab3.setLayout(self.tab3.layout)

        self.velocity_calculation()


        


    def velocity_calculation(self):
        #total_time = self.duration*self.picos        
        corrector = 1/self.fps
        col_one_list2 = self.no_zeros['angle'].tolist()

        
        lista_finalz = [abs(j-(i)) for i, j in zip(col_one_list2[:-1], col_one_list2[1:])]
        #print(lista_finalz)
        #lista_finalz =  [abs(ele) for ele in lista_finalz]
        #print(lista_finalz)
        lista_finalz = [ round((x/corrector/1000),3) for x in lista_finalz]
        #print(lista_finalz)
        self.my_array_fin = np.array(lista_finalz)
        #dfc = self.no_zeros.iloc[1:]

        self.velocity_histogram()

    def velocity_histogram(self):
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_title("Frequency Histogram of the caudal velocity", pad=15)  
        figure_velocity_histogram = sns.distplot(self.my_array_fin, axlabel='velocity (degress/ms)', bins=int(self.choice_histogram_collumns), kde=False, rug=True, color="g", ax=ax);        
        self.figure = figure_velocity_histogram.get_figure()
        plt.subplots_adjust(top = 0.9)
        plt.subplots_adjust(bottom = 0.2)

        self.canvas_histo_velocity = FigureCanvas(self.figure)

        self.tab4 = QWidget()
        self.tab4.layout = QVBoxLayout(self)        
        self.tab4.layout.addWidget(self.canvas_histo_velocity)      
        self.tab4.setLayout(self.tab4.layout)

       

        self.kernel_density()
        

    def kernel_density(self):      
        
        kernel_plot = sns.jointplot(x=self.my_array_fin, y=self.no_zeros['angle'][1:], kind="kde", cmap="jet", cbar = False, height=6)            
        kernel_plot.plot_joint(plt.scatter, c="black", s=30, linewidth=1, marker=".")           
        kernel_plot.set_axis_labels("degrees/ms", "Angle")      
        kernel_plot.fig.subplots_adjust(top=0.9, bottom = 0.2)
        kernel_plot.fig.suptitle("Kernel-density of the tail angle (degrees X tail velocity)", fontsize=12, wrap=True)
                
        self.figure = kernel_plot.fig 
        

        self.canvas_kernel_density = FigureCanvas(self.figure)

        self.tab5 = QWidget()
        self.tab5.layout = QVBoxLayout(self)        
        self.tab5.layout.addWidget(self.canvas_kernel_density)      
        self.tab5.setLayout(self.tab5.layout)

        self.aceleration()

    def aceleration(self):
        lista_acel = []
        for n in self.heat_map_integral:                      
            x_center = n[0]
            lista_acel.append(x_center)
            
      
        soma_value = []      
        x=0
        for i in range (0, len(lista_acel)):
            if x > 0 and lista_acel[x] > lista_acel[x-1]:                
                diff_value = lista_acel[x] - lista_acel[x-1]
                soma_value.append(diff_value)
            x=x+1    
        
        
        if len(soma_value) > 0:
            self.final_value = round(mean(soma_value), 2)
          


        self.acceleration_message = QLabel("The fish Acceleration rate is: " + str(self.final_value) + " pixels/frame", self)
        self.acceleration_message.move(100, 100)
        self.acceleration_message.setFont(QFont('Arial', 16))          
        self.acceleration_message.show()       

        self.tab6 = QWidget()
        self.tab6.layout = QVBoxLayout(self)        
        self.tab6.layout.addWidget(self.acceleration_message)      
        self.tab6.setLayout(self.tab6.layout)   


        ############### insert all tabs in layout ###################

        self.insert_tabs_in_layout()

    


    def insert_tabs_in_layout(self):
        # refresh canvas
        self.canvas_peaks.draw()
        self.canvas_heat.draw()
        self.canvas_histo_raw.draw()
        self.canvas_histo_velocity.draw()
        self.canvas_kernel_density.draw()
        #self.acceleration_message.draw()

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        #self.tab1 = QWidget()
        #self.tab2 = QWidget()
        self.tabs.resize(300,200)

        # Add tabs
        self.tabs.addTab(self.tab1,"Peaks")
        self.tabs.addTab(self.tab2,"Heat Map")
        self.tabs.addTab(self.tab3,"Angle Histogram")
        self.tabs.addTab(self.tab4,"Velocity Histogram")
        self.tabs.addTab(self.tab5,"Kernel Density")
        self.tabs.addTab(self.tab6,"Acceleration")   
  

         
    

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()