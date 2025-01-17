# Generate MIDI from a train NN

import glob
import pickle
import numpy
import sys
from music21 import instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import Dense, Dropout,LSTM, Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def generate():
    print('Generate a MIDI files', sep=' ', end='\n', file=sys.stdout, flush=False)
    #load the notes used to train the model
    with open('data/notes', mode='rb', buffering=-1,
    encoding=None, errors=None, newline=None, closefd=True) as filepath:
        notes = pickle.load(filepath)

    #get all pitch names
    pitchnames = sorted(set(item for item in notes))
    n_vocab = len(set(notes))

    network_input, normalize_input = prepare_sequences(notes,pitchnames,n_vocab)
    model = create_network(normalize_input, n_vocab)
    prediction_output = generate_notes(model, network_input,pitchnames,n_vocab)
    create_midi(prediction_output)

def prepare_sequences(notes, pitchnames, n_vocab):
    print('prepare sequences used by the NN', sep=' ',
    end='\n', file=sys.stdout, flush=False)
    #map between notes and integers and back
    note_to_int = dict((note,number)for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input = []
    output = []
    for i in range(0, len(notes)- sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i+ sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    #reshape the input into a format compatible with LSTM layers
    normalized_input = numpy.reshape(network_input,(n_patterns,sequence_length,1))
    #normalize input
    normalized_input = normalized_input/float(n_vocab)

    return(network_input, normalized_input)

def create_network(network_input, n_vocab):
    #Create the structure of the NN
    print("Creating NN \n")
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(
        network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
    metrics=None, loss_weights=None, sample_weight_mode=None,
    weighted_metrics=None, target_tensors=None)

    #load the weights to each node

    model.load_weights("weights-improvement-01-4.7702-bigger.hdf5",
    by_name=False, skip_mismatch=False, reshape=False)

    return model

def generate_notes(model, network_input, pitchnames, n_vocab):
    print('Generate notes from the NN', sep=' ', end='\n', file=sys.stdout, flush=False)
    #pick a random sequence from the input as a starting point for the prediction
    start = numpy.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note)for number, note in enumerate(pitchnames))
    pattern = network_input[start]
    prediction_output = []

    #generate 500 notes
    for note_index in range(500):
        prediction_input = numpy.reshape(pattern, (1, len(pattern),1))
        prediction_input = prediction_input/float(n_vocab)

        prediction = model.predict(prediction_input,batch_size=None,
        verbose=0, steps=None)

        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(index)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def create_midi(prediction_output):
    print('Convert the output from the prediction to a MIDI file',
    sep=' ', end='\n', file=sys.stdout, flush=False)
    offset = 0
    output_notes = []
    #create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        #pattern is chords
        print(pattern, sep=' ', end='\n\n', file=sys.stdout, flush=False)
        patt = str(pattern)
        if ('.' in patt) or patt.isdigit():
            notes_in_chord = patt.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)

            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        #pattern is a note
        else:
            new_note = note.Note(pitchName=pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        #increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi',fp ='test_output.mid')

if __name__ == '__main__':

    generate()
