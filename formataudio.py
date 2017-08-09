import librosa
path='data/test/' #default
# path='data/people/'
def formataudio(path2):
    """Formats wav file into 16 different speeds and writes them to path2 directory

        Parameters
        ----------
        path2 : file name, wav format"""

    y, sr = librosa.load(path+path2+'.wav.ig')
    for i in range(16):
        if i == 4:
            librosa.output.write_wav('{}_{}_{}.wav'.format(path,path2,i+1), y, sr)
        else:
            y_stretch = librosa.effects.time_stretch(y, (i*2/16)+.5)  # .5 to 2.5 factor
            librosa.output.write_wav('{}_{}_{}.wav'.format(path,path2,i+1), y_stretch, sr)
if __name__ == '__main__':
    path2 = input("Please enter the file to format\n")
    formataudio(path2)
