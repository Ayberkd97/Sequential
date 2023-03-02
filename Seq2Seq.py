import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Flatten
from keras import layers,models, Sequential
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ModelCheckpoint
import keras.backend as K


ball = np.load('ball.npy')
attacker = np.load('attacker.npy')
defender = np.load('defender.npy')


ball_test = ball[28000:,:]
ball = ball[:28000,:]

attacker_test = attacker[28000:,:]
attacker = attacker[:28000,:]

defender_test = defender[28000:,:]
defender = defender[:28000,:]




# Model
class Seq2Seq:
    
    error_list = []
    prediction_list = []
    objects = []
    def __init__(self, data, test, best_model, input_timesteps, output_timesteps):
        self.__class__.objects.append(self)
        self.train_input = data[:, data.shape[1]-(output_timesteps+input_timesteps):data.shape[1]-output_timesteps]
        self.train_target = data[:, data.shape[1]-output_timesteps:]
        self.test_input = test[:, test.shape[1]-(output_timesteps+input_timesteps):test.shape[1]-output_timesteps]
        self.test_target = test[:, test.shape[1]-output_timesteps:]
        self.best_model = best_model
        self.output_timesteps = output_timesteps
        self.input_timesteps = input_timesteps
        self.model = None
#         self.sample = np.random.choice(self.test_input.shape[0])
        
    
    def average_displacement_error(self, y_true, y_pred):
        return K.sqrt(K.sum(K.square(y_true - y_pred))) / y_pred.shape[1]
    
    
    def final_displacement_error(self, y_true, y_pred):
        return K.sqrt(K.sum(K.square(y_true - y_pred)))
    
    
    def train_model(self, epochs=30, batch_size=100, units=64, second_layer=False):       
        
        encoder_inputs = layers.Input(shape=(self.input_timesteps, 2))

        # Encoder
        encoder = layers.LSTM(units, return_state=True, return_sequences=False)
        
#         if second_layer:
#             encoder = layers.LSTM(units, return_state=True, return_sequences=False)

        _, state_h, state_c = encoder(encoder_inputs)

        # Decoder
        decoder_inputs = layers.RepeatVector(self.output_timesteps)(state_h)
        decoder_lstm = layers.LSTM(units, return_sequences=True, return_state=False)
        decoder_lstm = layers.LSTM(units, return_sequences=True, return_state=False)
        decoder = decoder_lstm(decoder_inputs, 
                               initial_state=[state_h, state_c])


        # Output
        # applies a same Dense (fully-connected) operation to every timestep of a 3D tensor
        out = layers.TimeDistributed(Dense(128))(decoder)
        layers.Dropout(0.2)
        out = layers.TimeDistributed(Dense(256))(out)
        layers.Dropout(0.2)
        out = layers.TimeDistributed(Dense(2))(out)


        model = models.Model(encoder_inputs, out)
        model.compile(optimizer='adam', loss=self.average_displacement_error, metrics = [self.final_displacement_error])

        callbacks = [ModelCheckpoint(filepath=self.best_model, monitor='val_loss', save_best_only=True)]

        model.fit(self.train_input, self.train_target,
              batch_size= batch_size,
              epochs= epochs, 
              validation_split=0.2,
              callbacks = callbacks,
              verbose=1,
              shuffle=True)
        
        self.model = model
        self.model.load_weights('./'+self.best_model)
        
        self.error_list.append(self.model.history.history['val_loss'])
    
    
    
    def plot_loss(self):
        model = self.model
        plt.figure(figsize=(10,7))
        plt.plot(model.history.history['loss'], label='Average Displacement Error: train')
        plt.plot(model.history.history['val_loss'], label='Average Displacement Error: validation')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Average Displacement Error')
        plt.show()
        
        
        
    def predict_trajectory(self, sample):
        pred = self.model.predict(self.test_input[sample].reshape(1,self.test_input.shape[1],2))
        pred = pred.reshape(self.output_timesteps, 2)
        self.prediction_list.append(pred)
        
        plt.figure(figsize=(10,7))
        plt.plot(pred[:,0], pred[:,1], marker='o', label='Seq2Seq Model: Test Data', color='navy', alpha=0.7)
        plt.plot(self.test_input[sample][-25:, 0], \
                 self.test_input[sample][-25:, 1], marker='o', \
                 label='Observed Trajectory', color='red', alpha=0.7)
        plt.plot(self.test_target[sample][:,0], self.test_target[sample][:,1], marker='o', label='Ground Truth', color='green', alpha=0.7)
        plt.xlabel('x-coordinate', fontsize=12)
        plt.ylabel('y-coordinate', fontsize=12)
        plt.title('Trajectory Prediction with Input Timesteps = {}'.format(self.input_timesteps), fontsize=18)
        plt.legend()
        plt.show()


input_timesteps = [25, 50, 100]

# train: ball
for n in input_timesteps:
    model = Seq2Seq(ball, ball_test, str(n)+'.h5', n, 25)
    model.train_model(epochs=2, batch_size=200)

# prediction: ball
sample = np.random.choice(2000)
for m in Seq2Seq.objects[-3:]:
    m.predict_trajectory(sample)

# errors: ball
plt.figure(figsize=(8,5))
for m,e in zip(Seq2Seq.objects[-3:], Seq2Seq.error_list[-3:]):
    plt.bar(str(m.input_timesteps), min(e))
    plt.xlabel('Input Timesteps', fontsize=13)
    plt.ylabel('Average Displacement Error', fontsize=13)
    plt.title('Comparison of Validation Errors of Models', fontsize=18)


# train: attackers
for n in input_timesteps:
    model = Seq2Seq(attacker, attacker_test, str(n)+'.h5', n, 25)
    model.train_model(epochs=50, batch_size=200)

# prediction: attackers
sample = np.random.choice(2000)
for m in Seq2Seq.objects[-3:]:
    m.predict_trajectory(sample)

# errors: attackers
plt.figure(figsize=(8,5))
for m,e in zip(Seq2Seq.objects[-3:], Seq2Seq.error_list[-3:]):
    plt.bar(str(m.input_timesteps), min(e))
    plt.xlabel('Input Timesteps', fontsize=13)
    plt.ylabel('Average Displacement Error', fontsize=13)
    plt.title('Comparison of Validation Errors of Models', fontsize=18)


# train: defenders
for n in input_timesteps:
    model = Seq2Seq(defender, str(n)+'.h5', n, 25)
    model.train_model(epochs=50, batch_size=200)

# prediction: defenders
sample = np.random.choice(2000)
for m in Seq2Seq.objects[-3:]:
    m.predict_trajectory(sample)

# errors: defenders
plt.figure(figsize=(8,5))
for m,e in zip(Seq2Seq.objects[-3:], Seq2Seq.error_list[-3:]):
    plt.bar(str(m.input_timesteps), min(e))
    plt.xlabel('Input Timesteps', fontsize=13)
    plt.ylabel('Average Displacement Error', fontsize=13)
    plt.title('Comparison of Validation Errors of Models', fontsize=18)
