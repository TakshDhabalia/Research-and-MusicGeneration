from tensorflow import keras
import json
from Melody_generation.main import SEQUENCE_LENGHT,MAPPING_PATH
import numpy as np 
import music21 as m21


class MelodyGenerator:
    def __init__(self , model_path = "model1.h5"):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH ,"r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGHT

    def generate_melody(self ,seed ,num_steps , max_sequence_length, temperature ):
        #temperature - (0,infinity) --> way bwe sample output symbols 
        #"64___64_63"--->seed 
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        #map seed to integers from the look up table 
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):

            #limit the seed to the specified sequence lenght (max) or the last releevnt steps 
            seed = seed[-max_sequence_length:]

            #one hot encode the seed for simplicity 
            onehot_seed = keras.utils.to_categorical(seed , num_classes=len(self._mappings))
            onehot_seed = onehot_seed[np.newaxis , ...]

            probabilites = self.model.predict(onehot_seed)[0]

            output_int = self._sample_with_temperature(probabilites,temperature)

            seed.append(output_int)
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            # check whether we're at the end of a melody
            if output_symbol == "/":
                break

            # update melody
            melody.append(output_symbol)
        return melody

    
    def _sample_with_temperature(self,probabilites ,temperature ):
        predictions = np.log(probabilites) / temperature
        probabilites = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilites)) # [0, 1, 2, 3]
        index = np.random.choice(choices, p=probabilites)
        
        return index
    def save_melody(self, melody ,step_duration= 0.25,format = "midi" , filename="mel.midi"):
        # create a music21 stream
        stream = m21.stream.Stream()

        start_symbol = None
        step_counter = 1

        # parse all the symbols in the melody and create note/rest objects
        for i, symbol in enumerate(melody):

            # handle case in which we have a note/rest
            if symbol != "_" or i + 1 == len(melody):

                # ensure we're dealing with note/rest beyond the first one
                if start_symbol is not None:

                    quarter_length_duration = step_duration * step_counter # 0.25 * 4 = 1

                    # handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    # reset the step counter
                    step_counter = 1

                start_symbol = symbol

            # handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1

        # write the m21 stream to a midi file
        stream.write(format, filename)





if __name__ == "__main__":
    mg = MelodyGenerator()
    seed = "55 _ _ _ 60 _ _ _ 55 _ _ _ 55 _"
    seed2 = "55 _ 60 _ 60 _ 60 59 60 62 64 65 "
    melody = mg.generate_melody(seed2, 500, SEQUENCE_LENGHT, 0.4)
    print(melody)
    mg.save_melody(melody)