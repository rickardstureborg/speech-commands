import speech_recognition as sr
from os import getcwd, listdir

def get_dataset(PATH_SPEAKERS,model,save_file=False):
    """Converts all wave files from SLU dataset into text,
    either yields the files or saves them in place with same
    structure as original dataset.
    
    Arguments:
        PATH_SPEAKERS {string} -- path to the speakers folder in dataset
        model {speech_recognition.Recognizer} -- Model for recognizing speech
    
    Keyword Arguments:
        save_file {bool} -- Determines whether text files will be 
                            saved or not (default: {False})
    """
    for speaker in listdir(PATH_SPEAKERS):
        for speech in listdir(PATH_SPEAKERS+speaker):
            # import speech file
            PATH_FILE = PATH_SPEAKERS+speaker+'/'+speech
            with sr.AudioFile(PATH_FILE) as source:
                audio = model.record(source)
            # convert to text
            text = model.recognize_google(audio)
            # save text file
            if save_file:
                with open(PATH_SPEAKERS+speaker+'/'+speech[:-4]+'.txt','w') as wf:
                    wf.write(text)


if __name__ == "__main__":

    r = sr.Recognizer()
    PATH = getcwd()
    PATH_SPEAKERS = PATH+'/../slu_data/wavs/speakers/'  # may need to edit this
    get_dataset(PATH_SPEAKERS,r,save_file=True)
