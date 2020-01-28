#! /usr/bin/python3
"""
with every frame that comes in:
    check if it is_speech 
    append to buffer - frame and True/False indicating if it contains speech
    if 90% of frames in buffer are speech then:
        get current time (then to be used in filename)
        start saving to a separate array

    when 90% not speech:
        save the array to a file with time as filename
        empty the array
        continue running
"""
import pyaudio
import collections
import contextlib
import wave
import webrtcvad
import datetime
from pathlib import Path
import yaml

with open("setup.yaml", "r") as f:
    setup = yaml.load(f, Loader=yaml.FullLoader)

# saves section with audio to file
# file name is the time when sombedy started speaking
def write_wave(path, audio, sample_rate):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)

def main():
    FOLDER = setup["audio_folder"]
    os.makedirs(FOLDER, exist_ok=True)

    FORMAT = pyaudio.paInt16
    CHANNELS = setup["A_CHANNELS"]
    RATE = setup["A_RATE"]
    FRAME_DURATION = setup["A_FRAME_DURATION"] #30 ms
    CHUNK = int(RATE*FRAME_DURATION/1000)
    PADDING_DURATION = setup["A_PADDING_DURATION"]
    num_padding_frames = int(PADDING_DURATION/FRAME_DURATION)
    FILE_PATTERN = FOLDER + '%s.wav'

    p = pyaudio.PyAudio()
    # ring buffer to store the last n seconds of audio
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # voice activity detector
    vad = webrtcvad.Vad(3)
    speech = False
    num_speech = 0
    voiced_frames = []
    time = ""

    # this starts audio recording
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    #print("recording")
    while True:
        # read a section of 30ms
        data = stream.read(CHUNK)
        is_speech = vad.is_speech(data, RATE)

        if not speech:
            ring_buffer.append((data, is_speech))
            num_speech = len([data for data, speech in ring_buffer if speech])
            # if more than 90% speech then get current time &
            # save contents of buffer to voiced_frames array
            if num_speech>0.9*num_padding_frames:
                speech = True
                time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S.%f")
                #print(time + "\n")
                #print("you started speaking")
                for data, speech in ring_buffer:
                    voiced_frames.append(data)
                ring_buffer.clear()
        else:
            ring_buffer.append((data, is_speech))
            voiced_frames.append(data)
            num_notspeech = len([data for data, speech in ring_buffer if not speech])
            # if more than 90% of frames are not speech
            # then save the speech to a file
            if num_notspeech>0.9*num_padding_frames:
                speech=False
                sentence = b''.join([f for f in voiced_frames])
                write_wave(FILE_PATTERN%time,sentence,RATE)
                ring_buffer.clear()
                voiced_frames=[]
                #print("you finished speaking")



if __name__ == '__main__':
    main()

