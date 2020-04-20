import speech_recognition as sr
from os import getcwd, listdir
from gensim.models import Word2Vec
import torch
import pickle
from gensim.test.utils import common_texts


def get_dataset():
    """Converts all wave files from SLU dataset into WEs"""
    r = sr.Recognizer()
    PATH = getcwd()
    PATH_SPEAKERS = PATH+'/slu_data/wavs/speakers/'  # may need to edit this

    w2v = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)

    for speaker in listdir(PATH_SPEAKERS):
        for speech in listdir(PATH_SPEAKERS+speaker):
            # import speech file
            PATH_FILE = PATH_SPEAKERS+speaker+'/'+speech
            with sr.AudioFile(PATH_FILE) as source:
                audio = r.record(source)
            # convert to text
            text = r.recognize_google(audio)
            print(text)
            # convert into word embeddings
            embeddings = []
            for word in text.split(' '):
                embeddings.append(w2v.wv['computer'])
            embeddings = torch.Tensor(embeddings)
            print(embeddings)
            #pickle.dump(embeddings, open(PATH_SPEAKERS+speaker+'/'+speech[:-4]+'.p', 'w'))

if __name__ == "__main__":
    get_dataset()
