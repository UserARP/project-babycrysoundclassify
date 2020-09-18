import sounddevice
from scipy.io.wavfile import write
import librosa
import numpy as np
import tensorflow as tf

fs = 22050
second = 5
D = []
test_model = tf.keras.models.load_model(
    'C:/Users/ppeng/Documents/AI_babyclassified/model_DataAugmented_ver21')
print("recording...")
record_voice = sounddevice.rec(int(second * fs), samplerate=fs, channels=2)
sounddevice.wait()
write('C:/Users/ppeng/Documents/AI_babyclassified/record/wavfile.wav', fs, record_voice)
y, sr = librosa.load(
    'C:/Users/ppeng/Documents/AI_babyclassified/record/wavfile.wav', duration=4.97)
ps = librosa.feature.melspectrogram(y=y, sr=sr)
D.append(ps)

X_real = np.array([x.reshape((128, 215, 1)) for x in D])
X_test_test = np.asarray(X_real)
print(test_model.predict_classes(X_test_test))
