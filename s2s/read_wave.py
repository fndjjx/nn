import wave
import numpy as np
def read_wave_data(file_path):  
    #open a wave file, and return a Wave_read object  
    f = wave.open(file_path,"rb")  
    #read the wave's format infomation,and return a tuple  
    params = f.getparams()  
    #get the info  
    nchannels, sampwidth, framerate, nframes = params[:4]  
    #Reads and returns nframes of audio, as a string of bytes.   
    str_data = f.readframes(nframes)  
    #close the stream  
    f.close()  
    #turn the wave's data to array  
    wave_data = np.fromstring(str_data, dtype = np.short)  
    #for the data is stereo,and format is LRLRLR...  
    #shape the array to n*2(-1 means fit the y coordinate)  
    wave_data.shape = -1, 2  
    #transpose the data  
    wave_data = wave_data.T  
    #calculate the time bar  
    time = np.arange(0, nframes) * (1.0/framerate)  
    return wave_data, time  

a,b = read_wave_data("final.wav")
print(a.shape)
print(b.shape)
print(list(a[0][:10000]))
