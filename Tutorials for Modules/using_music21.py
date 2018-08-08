# test to get midi files and feed them to the NN

import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense, Dropout,LSTM, Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def train_network():
    # Train a NN to generate music21
    notes = get_notes()

    #get amount of pitch names
    n_vocab = len(set(notes))

    network_input , network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)


def get_notes():
    # Get all the notes and chords from the MIDI files in the ./midi directory
    print("Getting Notes")
    notes = []
    counter = 1

    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)
        print("N° {} Parsing {}".format(counter,file))
        notes_to_parse = None
        counter = counter+1
        try: #the file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n)for n in element.normalOrder))
    with open('data/notes','wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, n_vocab):
    #prepare the sequences used by te NN
    print("Preparing Sequences")
    sequence_length = 100
    #get all the pitch names
    pitchnames = sorted(set(item for item in notes))
    #Create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    #Create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    #reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    #normalize network_input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output, num_classes=None)

    return (network_input, network_output)


def create_network(network_input, n_vocab):
    #Create the structure of the NN
    print("Creating NN")
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

    return model

def train(model, network_input, network_output):
    #Train the NN
    print("Training NN")
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(filepath,
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        mode='auto',
        period=1)

    callbacks_list = [checkpoint]
    model.fit(x=network_input, y=network_output, batch_size=64,
        epochs=3, verbose=1, callbacks=callbacks_list,
        validation_split=0., validation_data=None,
        shuffle=True, class_weight=None, sample_weight=None,
        initial_epoch=0, steps_per_epoch=None, validation_steps=None)

if __name__ == '__main__':

    train_network()
