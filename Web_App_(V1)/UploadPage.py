from funcs2 import *
from pydub import AudioSegment
from pydub.playback import play
import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import time

model = mod()

def py(soundn):
    #sound = AudioSegment.from_wav('C:\\Users\\kulka\\NoteClasf_webapp\\newaudio\\' + str(soundn))
    #play(sound)
    audio_file = open('C:\\Users\\kulka\\NoteClasf_webapp\\Newaudio\\' + str(soundn) + ".wav",'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes)

def createSound(filename):
    fs = 48000  # Sample rate
    seconds = 10  # Duration of recording

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2,dtype=np.int16)
    sd.wait()  # Wait until recording is finished
    write('C:\\Users\\kulka\\NoteClasf_webapp\\Newaudio\\' + filename  + ".wav", fs, myrecording)  # Save as WAV file 


def p_page():
    
    st.markdown("<h1 style='text-align: center; color: white;'>Predict on a new recording!</h1>", unsafe_allow_html=True)

    
    textinput = st.text_input("Enter A Name For Your Recording","")
    crec = st.button("Record Audio Of A Guitar")

    if crec:
        
        st.write("Recording in progress!")

        createSound(str(textinput))

        time.sleep(2)

        st.write("Recording Complete!")

        py(textinput)

        #___________________________________________________

        plotSaveSound("C:\\Users\\kulka\\NoteClasf_webapp\\Newaudio\\" + str(textinput)  + ".wav",textinput)

        data = createData2("C:\\Users\\kulka\\NoteClasf_webapp\\imsnew\\"+str(textinput) + ".png",X_data=[])

        y_pred = model.predict(data)

        num_to_note = {0:"3G",9:"6E",10:"6F",11:"6F#",12:"6G",5:"5A",7:"5Bb",6:"5B",8:"5C",1:"4D",3:"4Eb",2:"4E",4:"4F"}

        out = np.argmax(y_pred)
        fin = num_to_note.get(out)
        
        st.write("The Predicted Note Is: " + str(fin))
    else:
        pass




        




        