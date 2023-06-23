import os 
import json
import music21 as m21 
from tensorflow import keras 
import numpy as np 


KERN_DATASET_PATH = "D:/Sound_Generation/dataset/deutschl/erk"
SAVE_DIR = r"D:\Sound_Generation\Updated_dataset"
SINGLE_FILE_DATASET = "file_dataset"
SEQUENCE_LENGHT = 64 #when training our LSTM we need to pass them 
MAPPING_PATH = r"mapping.json"
ACCEPTABLE_DURATIONS = [
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    4,
]
def Load_Songs_In_Kern(dataset_path):
    #go through all the songs one by one and see them load in music21
    
    songs = []
    for path ,subdir ,files in os.walk(dataset_path):#basic recursion for subdirectories and files 
        for file in files:
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path,file))
                songs.append(song)
    return songs

def has_acceptable_durations(song , acceptable_duration) :
    #bool function i mean pretty clear 
    pass
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_duration:
            return False
    return True

def Transpose(song):
    #get the key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]
    #get/estimate key using the music21
    #we also need to know even if we have the key or not 
    if not isinstance(key ,m21.key.Key):
        key = song.analyse("key")#analyse is basically majik for us commeners
    

    #get the interval for transposition basically Bmaj --> cmaj (interval estimation or info find out basically)
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic , m21.pitch.Pitch("C"))#tonic is basically pitch object
    elif key.mode =="minor":
        interval = m21.interval.Interval(key.tonic , m21.pitch.Pitch("A"))
    #transpose song by calculated interval 
    transposed_song = song.transpose(interval)
    return transposed_song
    #reson for transposition - we dont want all the data , we dont want other key , we only want to learn about c and A major and minor resp 
    #we can use way less data 

def encode_song(song , time_step = 0.25):
    #p = 100 , d = 1.00 ---> [60 ,"_","_","_"] -- basic representation 
    encode_song = []
    for event in song.flat.notesAndRests:
        #handle notes first 
        if isinstance(event , m21.note.Note):
            symbol = event.pitch.midi #60 in our example 
        elif isinstance(event , m21.note.Note):
            symbol = "r"
        #convert into a time series convention or the RHS equation 
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encode_song.append(symbol)
            else:
                encode_song.append("_")
    #cast or convert to string 
    encode_song = " ".join(map(str , encode_song))
    return encode_song




def preprocessor(dataset_path):
    pass 
    #load the songs 
    #filtre ou the songs that have non acceptable durations 
    #main preprocessing is the amount of work done before we actually deploy this amount of data 
    #amount of data we collected is from a a source called KERN dataset and the language of choice is 
    #dutch
    print("loading songs")
    songs = Load_Songs_In_Kern(dataset_path)
    print(f"Loaded {len(songs)} songs .")

    for i ,song in enumerate(songs):
        #we need to filtre out acceptable range or not 
        if not has_acceptable_durations(song , ACCEPTABLE_DURATIONS):
            continue#skips the song in the preprocess range

        #transpose songs to Cmaj or C min 
        song = Transpose(song)

        #encode songs in music time series representation
        encoded_song = encode_song(song)

        #load songs into a text file to a dataset 
        save_path = os.path.join(SAVE_DIR ,str(i))
        with open(save_path, "w") as fp:

            fp.write(encoded_song)


def load(file_path):
    with open(file_path , "r") as fp:
        song = fp.read()
    return song



def create_single_file_dataset(dataset_path ,file_dataset_path , sequence_lenght):
    new_song_delimiter = "/ "*sequence_lenght
    songs = ""


    #load encoded songs and add the limiters 
    for path , _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path ,file)
            song = load(file_path)
            songs = songs + song +" " + new_song_delimiter

    songs = songs [:-1]
    #add or join the all dataset
    with open(file_dataset_path , "w") as fp:
        fp.write(songs)
    return songs 

def create_mapping(songs,mapping_path):
    mappings = {}

    #identify the vocabulary(disctionary) 
    songs = songs.split()
    vocabulary = list(set(songs))

    #create a mappings system 
    for i ,symbol in enumerate(vocabulary):
        mappings[symbol] = i

    #save the vocabulary to a json file format file 
    with open(mapping_path ,"w") as fp : 
        json.dump(mappings ,fp , indent=4)

def convert_songs_to_int(songs):
    int_songs = []
    # load the mapping and the mappings which are in json specifiaclly and then 
    with open(MAPPING_PATH , "r") as fp:
        mappings = json.load(fp)
    #cast songs string to a list
    songs = songs.split()
    #maps songs to int as the last step
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs


########################################generating training sequences ########################################
def generating_training_sequences(sequence_lenght):
    #[11, 12,13,14 ...........] ---> [11,12] , [13] and the next step [12 ,13] ,[14]

    #load songs and map them to int
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)

    #generate the training sequences 
    #100 symbols dataset , 64 sequence lenght , 100-64 = 36 items x64
    inputs = []
    targets = []
    num_sequences = len(int_songs) - sequence_lenght
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_lenght])
        targets.append(int_songs[i+sequence_lenght])


    #one-hot encode the sequences 
    vocabulary_size = len(set(int_songs))
    #inputs = keras.utils.to_categorical(in)
    inputs = keras.utils.to_categorical(inputs , num_classes=vocabulary_size)
    targets = np.array(targets)
    return inputs, targets




def main():
    preprocessor(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR , SINGLE_FILE_DATASET, SEQUENCE_LENGHT)  
    create_mapping(songs ,MAPPING_PATH )
   




if __name__ == "__main__":
    main()   
    








    