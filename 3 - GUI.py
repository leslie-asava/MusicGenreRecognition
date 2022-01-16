# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 04:39:59 2021

@author: samna
"""

import sys
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication
import librosa.display
from matplotlib.backends.qt_compat import QtCore
import tensorflow as tf
import numpy as np
import librosa

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QTextEdit, QPushButton
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy import signal
import os

initialized = False

SIG, SR = None, None
LEVEL = None
HIGH_LEVEL_TEXT, MID_LEVEL_TEXT, LOW_LEVEL_TEXT = "","",""
home = None

"""if QtCore.qVersion() >= "5.":
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure"""


MODEL_PATH = "model.json"
H5_PATH = "h5model.h5"
DATA_PATH = "MFCC.json"
SAMPLING_RATE = 22050
NUM_OF_MFCC = 13
N_SEGMENTS = 10
NUM_FFT = 2048
HOP_LEN = 512
MUSIC_LENGTH = 30
SAMPLES_PER_FILE = SAMPLING_RATE * MUSIC_LENGTH
SAMPLES_PER_SEGMENT = int(SAMPLES_PER_FILE / N_SEGMENTS)

# loading the trained model
file = open(MODEL_PATH, 'r')
model_json = file.read()
file.close()
loaded_model = tf.keras.models.model_from_json(model_json)
loaded_model.load_weights(H5_PATH)
model = loaded_model


# Canvas that will be used to plot the WAV signal
class WaveChartCanvas(FigureCanvas):
    def __init__(self, parent):
        fig, self.ax = plt.subplots(figsize=(10.8, 2.8), dpi = 75)
        fig.set_facecolor("lightgray")
        super().__init__(fig)
        self.setParent(parent)

# Canvas that will be used to plot the signal mel spectrogram
class SpectrogramChartCanvas(FigureCanvas):
    def __init__(self, parent):
        self.fig, self.ax = plt.subplots(figsize=(10.8, 3.7), dpi = 75)
        self.fig.set_facecolor("lightgray")
        super().__init__(self.fig)
        self.setParent(parent)

# Canvas that will be used to plot the signal mel spectrogram
class SignalChartCanvas(FigureCanvas):
    def __init__(self, parent):
        self.fig, self.ax = plt.subplots(figsize=(10.8, 3.7), dpi = 75)
        self.fig.set_facecolor("lightgray")
        super().__init__(self.fig)
        self.setParent(parent)

class Home(QMainWindow): # "HOME" window - Main screen
    def __init__(self):
        super(Home, self).__init__()
        loadUi("Home.ui", self)

        self.signal = None
        self.sample_rate = None
        
        # buttons on the top
        self.bInstructions.clicked.connect(self.goToInstructions)
        self.bHow.clicked.connect(self.goToHowDoesThisWork)
        self.bAbout.clicked.connect(self.goToAbout)
        self.bUpload.clicked.connect(self.uploadFile)

        self.loadingLabel.resize(800,30)

        # Instantiate the wave chart canvas
        self.wave_chart = WaveChartCanvas(self)
        # Position the chart
        self.wave_chart.move(0,155)
        #self.wave_chart.resize(800,230)

        self.time_label = QLabel(self)
        self.time_label.setText("Time")
        self.time_label.resize(800,18)
        self.time_label.move(0,359)
        self.time_label.setStyleSheet("background-color:#D3D3D3")
        self.time_label.setAlignment(Qt.AlignCenter)

        # Instantiate the spectrogram chart canvas
        self.spectrogram_chart = SpectrogramChartCanvas(self)

        # Position the chart
        self.spectrogram_chart.move(0,523)

        # Instantiate the spectrogram chart canvas
        self.signal_chart = SignalChartCanvas(self)

        # Position the chart
        self.signal_chart.move(0,523)

        self.output_textbox = QTextEdit(self)
        self.output_textbox.resize(800,110)
        self.output_textbox.move(0,377)

        self.high_level_button = QPushButton(self)
        self.high_level_button.setText("HIGH-LEVEL FEATURES")
        self.high_level_button.resize(266,34)
        self.high_level_button.move(0,488)
        self.high_level_button.clicked.connect(self.on_high_level_button_click)

        self.mid_level_button = QPushButton(self)
        self.mid_level_button.setText("MID-LEVEL FEATURES")
        self.mid_level_button.resize(266,34)
        self.mid_level_button.move(267,488)
        self.mid_level_button.clicked.connect(self.on_mid_level_button_click)

        self.low_level_button = QPushButton(self)
        self.low_level_button.setText("LOW-LEVEL FEATURES")
        self.low_level_button.resize(266,34)
        self.low_level_button.move(533,488)
        self.low_level_button.clicked.connect(self.on_low_level_button_click)

        if SR:
            signal, sample_rate = SIG, SR
            librosa.display.waveshow(signal, sr= sample_rate,alpha = 0.5, ax = self.wave_chart.ax)
            self.wave_chart.ax.set_title("Waveform")
            if LEVEL == "HIGH":
                self.on_high_level_button_click()

            elif LEVEL == "MID":

                self.on_mid_level_button_click()

            elif LEVEL == "LOW":

                self.on_low_level_button_click()

    def on_high_level_button_click(self):
        self.output_textbox.clear()
        self.spectrogram_chart.ax.clear()
        self.spectrogram_chart.show()
        self.signal_chart.hide()

        global LEVEL
        LEVEL = "HIGH"

        self.output_textbox.append(HIGH_LEVEL_TEXT)

        signal = SIG
        sample_rate = SR

        S = librosa.feature.melspectrogram(y=signal, sr=sample_rate)
        librosa.display.waveshow(signal, sr= sample_rate, ax = self.spectrogram_chart.ax) # For some reason it wouldn't work without this line
        spec = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), ax = self.spectrogram_chart.ax)

        try:
            self.cb.remove()
            self.fig.draw()
        except:
            pass

        self.cb = self.spectrogram_chart.fig.colorbar(spec)
        self.spectrogram_chart.ax.set_xlabel("Time")
        self.spectrogram_chart.ax.set_ylabel("Frequency")
        self.spectrogram_chart.ax.set_title("Mel-Spectrogram")
        #plt.colorbar()

    def on_mid_level_button_click(self):
        self.output_textbox.clear()
        self.signal_chart.ax.clear()
        self.spectrogram_chart.hide()
        self.signal_chart.show()

        global LEVEL
        LEVEL = "MID"

        self.output_textbox.append("Mid")

        signal = SIG
        sample_rate = SR

        fft = np.fft.fft(signal)
        spectrum = np.abs(fft)
        f = np.linspace(0, sample_rate, len(spectrum))

        left_spectrum = spectrum[:int(len(spectrum)/2)]
        left_f = f[:int(len(spectrum)/2)]

        librosa.display.waveshow(signal, sr= sample_rate, ax = self.signal_chart.ax) # For some reason it wouldn't work without this line
        self.signal_chart.ax.plot(left_f, left_spectrum, alpha=0.5, color = "red")
        self.signal_chart.ax.set_xlabel("Frequency")
        self.signal_chart.ax.set_ylabel("Magnitude")
        self.signal_chart.ax.set_title("Power Spectrum")

    def on_low_level_button_click(self):
        self.output_textbox.clear()
        self.signal_chart.ax.clear()
        self.spectrogram_chart.hide()
        self.signal_chart.show()

        global LEVEL
        LEVEL = "LOW"

        self.output_textbox.append("Low")

        FRAME_SIZE = 1024
        HOP_LENGTH = 512

        low = SIG

        # calculate the amplitude envelope
        def amplitude_envelope(signal, frame_size, hop_length):
            amplitude_envelope = []
            
            # calculate the amplitude envelope for each frame
            for i in range(0, len(signal), hop_length):
                current_frame_amplitude_envelope = max(signal[i:i+frame_size])
                amplitude_envelope.append(current_frame_amplitude_envelope)
                
            return np.array(amplitude_envelope)

        def fancy_aplitude_envelope(signal, frame_size, hop_length):
            return np.array([max(signal[i:i+frame_size]) for i in range(0, signal.size, hop_length)])

        ae_low = amplitude_envelope(low, FRAME_SIZE, HOP_LENGTH)

        fancy_ae_low = fancy_aplitude_envelope(low, FRAME_SIZE, HOP_LENGTH)

        frames = range(0, ae_low.size)
        t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)

        #self.spectrogram_chart.figure(figsize=(15, 17))

        #plt.subplot(3, 1, 1)
        librosa.display.waveshow(low, alpha=0.5, ax = self.signal_chart.ax)
        self.signal_chart.ax.plot(t, ae_low, color="r")
        self.signal_chart.ax.set_title("Amplitude Envelop")
        #plt.title("Low")
        #self.spectrogram_chart.ax.ylim((-1.5, 1.5))

       
        # making the top buttons work        
    def goToInstructions(self):
        instructions = Instructions()
        widget.addWidget(instructions)        
        widget.setCurrentIndex(widget.currentIndex()+1)
    
    def goToHowDoesThisWork(self):
        howDoesThisWork = HowDoesThisWork()
        widget.addWidget(howDoesThisWork)        
        widget.setCurrentIndex(widget.currentIndex()+1)
       
    def goToAbout(self):
        about = About()
        widget.addWidget(about)
        widget.setCurrentIndex(widget.currentIndex()+1)
  
    def uploadFile(self):
        global HIGH_LEVEL_TEXT, MID_LEVEL_TEXT, LOW_LEVEL_TEXT
        global SIG, SR 
        global home

        self.wave_chart.ax.clear()
        self.spectrogram_chart.ax.clear()
        self.loadingLabel.setText(" Uploading, please wait...")
        filename = QFileDialog.getOpenFileName()

        
        path = filename[0] # because it returns a tuple
        
        print("Loading...")
        if path.endswith(".wav"):            
            Analyze.display(self, path)
            
        else:
            errorMessage = (" Please upload a wav file. \n " + path[-3:] + \
                            " files are not yet supported. \n See the" \
                            "\"INSTRUCTIONS\" section if you need any help.")


            SIG, SR = None, None
            HIGH_LEVEL_TEXT, MID_LEVEL_TEXT, LOW_LEVEL_TEXT = "", "", ""

            # Reinitialize home to clear everything
            home = Home()
            widget.addWidget(home)        
            widget.setCurrentIndex(widget.currentIndex()+1)
            home.output_textbox.append(errorMessage)

            #self.loadingLabel.setText("")
            
        
    def outputBox(self, path):
        self.outputBox.setText(path)
          
class Instructions(QMainWindow): # "INSTRUCTIONS" window
    def __init__(self):
        super(Instructions, self).__init__()
        loadUi("Instructions.ui", self)
        self.bHome.clicked.connect(self.goToHome)
        self.bHow.clicked.connect(self.goToHowDoesThisWork)
        self.bAbout.clicked.connect(self.goToAbout)
        
    def goToHome(self):
        global home
        home = Home()
        widget.addWidget(home)        
        widget.setCurrentIndex(widget.currentIndex()+1)
        
    def goToHowDoesThisWork(self):
        howDoesThisWork = HowDoesThisWork()
        widget.addWidget(howDoesThisWork)        
        widget.setCurrentIndex(widget.currentIndex()+1)
       
    def goToAbout(self):
        about = About()
        widget.addWidget(about)
        widget.setCurrentIndex(widget.currentIndex()+1)
        
class HowDoesThisWork(QMainWindow): # "HOW DOES THIS WORK" window
    def __init__(self):
        super(HowDoesThisWork, self).__init__()
        loadUi("HowDoesThisWork.ui", self)
        self.bHome.clicked.connect(self.goToHome)
        self.bInstructions.clicked.connect(self.goToInstructions)
        self.bAbout.clicked.connect(self.goToAbout)
        
    def goToHome(self):
        global home
        home = Home()
        widget.addWidget(home)        
        widget.setCurrentIndex(widget.currentIndex()+1)
        
    def goToInstructions(self):
        instructions = Instructions()
        widget.addWidget(instructions)        
        widget.setCurrentIndex(widget.currentIndex()+1)
        
    def goToAbout(self):
        about = About()
        widget.addWidget(about)
        widget.setCurrentIndex(widget.currentIndex()+1)

class About(QMainWindow): # "ABOUT" window
    def __init__(self):
        super(About, self).__init__()
        loadUi("About.ui", self)
        self.bHome.clicked.connect(self.goToHome)
        self.bInstructions.clicked.connect(self.goToInstructions)
        self.bHow.clicked.connect(self.goToHowDoesThisWork)
        
    def goToHome(self):
        global home
        home = Home()
        widget.addWidget(home)        
        widget.setCurrentIndex(widget.currentIndex()+1)
        
    def goToInstructions(self):
        instructions = Instructions()
        widget.addWidget(instructions)        
        widget.setCurrentIndex(widget.currentIndex()+1)
        
    def goToHowDoesThisWork(self):
        howDoesThisWork = HowDoesThisWork()
        widget.addWidget(howDoesThisWork)        
        widget.setCurrentIndex(widget.currentIndex()+1)    
  

class Analyze(QMainWindow): # "ABOUT" window
   
    def __init__(self):
        super(About, self).__init__()
    
    def processFile(audio_file):

        global SIG, SR

        signal, sample_rate = librosa.load(audio_file, sr = SAMPLING_RATE)
        SIG, SR = signal, sample_rate

        home.wave_chart.ax.clear()
        home.spectrogram_chart.ax.clear()
        home.signal_chart.ax.clear()

        # Plot the WAV signal
        librosa.display.waveshow(signal, sr= sample_rate,alpha = 0.5, ax = home.wave_chart.ax)
        home.wave_chart.ax.set_title("Waveform")
  
        for d in range(10):

            # calculate start and finish sample for current segment
            first = SAMPLES_PER_SEGMENT * d
            last = first + SAMPLES_PER_SEGMENT
        
            # extract mfcc
            mfcc = librosa.feature.mfcc(signal[first:last], SAMPLING_RATE, n_mfcc = NUM_OF_MFCC, 
                                        n_fft = NUM_FFT, hop_length = HOP_LEN)
            mfcc = mfcc.T # transpose

        
            return mfcc
 
    def predictedGenre(path):
        genres = {0:"Blues",1:"Classical",2:"Country",3:"Disco",4:"Hip hop",5:"Jazz",6:"Metal",7:"Pop",8:"Reggae",9:"Rock"}
    
        inputFile = Analyze.processFile(path)
    
        predictFile = inputFile[np.newaxis, ..., np.newaxis]
    
        prediction = model.predict(predictFile)
    
        # fetch index with maximal value
        predictedIndex = np.argmax(prediction, axis = 1)
        
        return(" Predicted genre: "+ str(genres[int(predictedIndex)]))
           
    def display(self, path):

        global HIGH_LEVEL_TEXT, MID_LEVEL_TEXT, LOW_LEVEL_TEXT
        global SIG, SR  
        global home     
        
        song = path # changes the variable name so it's not confusing
        
        try:
            # removing the "Uploading, please wait..."
            self.loadingLabel.setText("")  
            
            # uploads the song to be analyzed
            song, sr = librosa.load(song)
            
            # size of the song
            songSize = str(song.size)
            
            #duration of 1 sample
            sample_duration = 1 / sr
            
            # duration of the audio signal in seconds
            duration = sample_duration * len(song)
            
            # outputs the analyzed information onto the screen
            genre = str(Analyze.predictedGenre(path))
            
            x = (str(genre) + "\n Size of the file: " +
                                   str(songSize) + "\n Duration of 1 sample: " +
                                   str(sample_duration) + " seconds" + 
                                   "\n Duration of the audio signal: " + 
                                   str(duration) + " seconds")

            HIGH_LEVEL_TEXT = x
            home.on_high_level_button_click()
            home.loadingLabel.setText("File " + os.path.basename(path) +" has been successfully analyzed")
             
             
        except:
            y = (" Unable to load the file. Please make sure" 
             " that:\n\n - It is a wav file\n - You haven't"
             " just changed the extension from mp3 to wav"
             " thinking that it's gonna \n   make it a wav file"
             "\n - You have read the INSTRUCTIONS on how to"
             " properly upload the file and see\n   the common"
             " mistakes")

            SIG, SR = None, None
            HIGH_LEVEL_TEXT, MID_LEVEL_TEXT, LOW_LEVEL_TEXT = "", "", ""
            
            # Reinitialize home to clear everything
            home = Home()
            widget.addWidget(home)        
            widget.setCurrentIndex(widget.currentIndex()+1)
            home.output_textbox.append(y)



# basic code to initialize the window frames and stacking positions
app = QApplication(sys.argv)
widget = QtWidgets.QStackedWidget()
home = Home()
widget.addWidget(home)
widget.setFixedHeight(800)
widget.setFixedWidth(800)
widget.setWindowTitle("Music Analyzer Tool")
widget.setWindowIcon(QtGui.QIcon("MusicAnalyzer.ico"))
widget.show()

try:
    sys.exit(app.exec_())
except:
    print("Program terminated. Goodbye!")