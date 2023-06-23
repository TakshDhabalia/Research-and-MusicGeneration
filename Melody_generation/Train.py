#vocabulary size = 37  for me 
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense
from Melody_generation.main import generating_training_sequences , SEQUENCE_LENGHT

OUTPUT_UNITS = 37
NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 1
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model1.h5"

def build_model(output_units ,num_units ,loss ,learning_rate ):

    #create the model architecture palin and simple 
    input = keras.layers.Input(shape =(None , output_units))
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(output_units, activation = "softmax")(x)
    
    model = keras.Model(input,output)
    #compile the model 
    model.compile(loss = loss,
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])
    
    model.summary()

    return model


def train(output_units =OUTPUT_UNITS ,num_units = NUM_UNITS,loss =LOSS ,learning_rate =LEARNING_RATE):

    #generate the training sequencses which we use from the other file 
    inputs , targets = generating_training_sequences(SEQUENCE_LENGHT)

    #Build the network 
    model = build_model(output_units ,num_units ,loss ,learning_rate )
    #train the model 
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)
    #save the model 
    model.save(SAVE_MODEL_PATH)

if __name__ == "__main__":
    train()
