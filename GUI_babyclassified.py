from PIL import ImageTk
import PIL.Image as Pimg
import pylab
from tkinter import *
import sounddevice
from scipy.io.wavfile import write
import librosa
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
import os
import matplotlib
matplotlib.use('Agg')  # No pictures displayed


fs = 22050
second = 5
ai_model = tf.keras.models.load_model(
    'C:/Users/ppeng/Documents/AI_babyclassified/model_DataAugmented_ver21')


root = Tk()
root.option_add("*Font", "consolas 18")
root.title("BabyCryclasification")
rec_photo = PhotoImage(
    file='C:/Users/ppeng/Documents/AI_babyclassified/photo/rec_buttom.png')
ai_photo = PhotoImage(
    file='C:/Users/ppeng/Documents/AI_babyclassified/photo/classification.png')
ai_ans = StringVar()


N = [0]


def click_ai():
    ai_answer = 'answer'
    ai_ans.set(f'the ans is = {ai_answer}')


def click_rec(panel):
    global path
    global mel_spectro
    D = []
    record_voice = sounddevice.rec(int(second * fs), samplerate=fs, channels=2)
    sounddevice.wait()
    write('C:/Users/ppeng/Documents/AI_babyclassified/record/wavfile.wav',
          fs, record_voice)
    y, sr = librosa.load(
        'C:/Users/ppeng/Documents/AI_babyclassified/record/wavfile.wav', duration=4.97)
    ps = librosa.feature.melspectrogram(y=y, sr=sr)
    D.append(ps)
    pylab.axis('off')
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    librosa.display.specshow(
        librosa.power_to_db(ps, ref=np.max))
    pylab.savefig('C:/Users/ppeng/Documents/AI_babyclassified/photo/mel_spec{}.png'.format(N[-1]+1),
                  bbox_inches=None, pad_inches=0)
    N.append(N[-1]+1)
    pylab.close()

    path = 'C:/Users/ppeng/Documents/AI_babyclassified/photo/mel_spec' + \
        str(N[-1]) + '.png'
    img = ImageTk.PhotoImage(Pimg.open(path))
    panel.configure(image=img)
    panel.image = img  # keep a reference!


path = 'C:/Users/ppeng/Documents/AI_babyclassified/photo/mel_spec' + \
    str(N[-1]) + '.png'
img = ImageTk.PhotoImage(Pimg.open(path))
panel = Label(root, image=img, compound=TOP)
panel.image = img
panel.pack(fill=X)

Label(root, textvariable=ai_ans, compound=CENTER).pack(fill=X)

Button(root, image=rec_photo, borderwidth=0,
       compound=BOTTOM, command=lambda: click_rec(panel)).pack(side=LEFT, fill=X)
Button(root, image=ai_photo, borderwidth=0,
       compound=BOTTOM, command=click_ai).pack(side=RIGHT, fill=X)

root.mainloop()
