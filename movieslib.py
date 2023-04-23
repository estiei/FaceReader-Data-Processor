import shutil
import os
import statistics 
import scipy.stats as stats
import math
from itertools import cycle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


COLSETS = {'eyes':['AU01_r', 'AU02_r', 'AU04_r'], 'midpart':['AU05_r', 'AU06_r', 'AU07_r', 'AU09_r'],
    'mouth':['AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r'], 
    'all':['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r']}

#walks through a geven directory and collects all files that contain the word file
#input: directory_name, file - a word that bust be included in the file name
def get_filenames(directory_name, file='frames'):
    files = os.listdir(directory_name)
    fnames = []
    for i in files:
        if file in i:
            fnames.append(i)
    return fnames

#creates a new folder on the path 'fold_path'
def new_folder(fold_path):
    if os.path.exists(fold_path):
        shutil.rmtree(fold_path)
    os.makedirs(fold_path)



################  CUTS  ########################




#inner function
#input: a pandas dataframe which represent one shot
#return: number of cuts; duration of each of them
#how to find a shot: 
# shots = df.face_id.unique() #find unique face_ids
# for id in shots:
#   shot = df[df['face_id'] == id]

def find_cats(shot):
    
    output = []
    shot = shot.reset_index()
    length = len(shot)
    init_frame = shot.iloc[0]['frame']
    current = init_frame
    init_time = shot.iloc[0]['timestamp']

    for i in range(length):
        temp = shot.iloc[i] 

        if temp['frame'] != current: #counting continous sequence of the frames
            output.append([init_frame, shot.iloc[i-1]['frame'], shot.iloc[i-1]['timestamp'] - init_time])
            init_frame = temp['frame']
            current = init_frame
            init_time = temp['timestamp']
        current += 1

    output.append([init_frame, shot.iloc[i]['frame'], shot.iloc[i]['timestamp'] - init_time])

    return output


def test():
    print('test')


#the function check if there are several cuts in one shot, and rename the shots
#it numerates cuts in the way: 27.0, 27.1, 27.2... or assign -1 to cuts lesser than 3sec
#output: dataframe with renamed face_id fields, number of cuts

def cuts_split(_df):
    df = _df
    shots = df.face_id.unique() #find unique face_ids
    cuts = 0
    cuts_logs = []

    for id in shots:
        shot = df[df['face_id'] == id]
        temp = find_cats(shot)

        if len(temp) > 1: #if there are cuts in the shot
            cuts += 1 #iterate the amount of shots with cuts
            print('The face_id that is changed:', id, temp, len(temp))
            cuts_logs.append(str(id) + ' ' + str(temp) + ' ' + str(len(temp)))
            for j in range(len(temp)):
                if temp[j][2] >= 3.0: #if a current cut is longer than 3 sec
                    df.loc[(df['frame'] >= temp[j][0]) & (df['frame'] <= temp[j][1]) & (df['face_id'] == id), 'face_id'] = id + j/10 #change face_id
                    #df[(df['frame'] >= temp[j][0]) & (df['frame'] <= temp[j][0])]['face_id'] = id + j/10 #change face_id
                else:
                    #df[(df['frame'] >= temp[j][0]) & (df['frame'] <= temp[j][0])]['face_id'] = -1 #mark as a negative number to delete later
                    df.loc[(df['frame'] >= temp[j][0]) & (df['frame'] <= temp[j][1]) & (df['face_id'] == id), 'face_id'] = -1 #mark as a negative number to delete later

    print('The amount of shots with cuts:', cuts)
    
    return df, cuts_logs




#transforms time in ssss.msmsms format to hh:mm:ss:ms
def time_transform(timestamp):

    time = timestamp.split('.')
    if len(time[1]) ==1:
        time[1] += '00'
    if len(time[1]) ==2:
        time[1] += '0'

    trial_t = int(time[0]) * 1000 + int(time[1])
    #print(trial_t)
    
    trial_h = (trial_t//3600000)%100 #in case an amount of hours is more than 99 hours
    trial_min = (trial_t % 3600000)//60000
    trial_s = (trial_t % 60000) //1000
    trial_ms = trial_t % 1000
    if trial_h > 9:
        time = str(trial_h) + ':'
    else:
        time = '0' + str(trial_h) + ':'
    if trial_min > 9:
        time = time + str(trial_min)
    else:
        time = time + '0' + str(trial_min)
    if trial_s > 9:
        time = time + ':' + str(trial_s)
    else:
        time = time + ':' + '0' + str(trial_s)
    if trial_ms > 0:
        time = time + '.' + str(trial_ms)
    else:
        time = time + '.' + '000'
    
    return time







####################   MEAN and SD #####################

#this function calculates mean and Standard deviation of one frame 
#input: shot: a dataframe; id: name of the frame (could be either number of a shot or a name of a movie)
#       facial act: a pandas dataframe with statistics for one shot of a movie

def mean_SD(shot, movie, id, stat_cols):

    df_output = pd.DataFrame()

    time = list(shot['timestamp'])

    df_output['movie'] = [movie]
    df_output['face_id'] = [id]
    df_output['time'] = time[-1] - time[0]

    for col in stat_cols:
        #mean
        df_output[col + '_mean'] = [ statistics.mean(shot[col]) ]
        #SD
        df_output[col + '_SD'] = [ statistics.stdev(shot[col]) ]

   
    return df_output





#this function calculates the mean and SD for the whole frame with multiple shots
# input: dataframe
#output: facial act: a pandas dataframe with statistics for the whole movie 
                 
def stat(df, movie, value='max', colSets=COLSETS):
    shots = list(df.face_id.unique()) #find unique face_ids in a dataframe

    #extract max values for each sets of AU
    if value == 'max':
        df['eyes'] = df[colSets['eyes']].max(axis=1)
        df['midpart'] = df[colSets['midpart']].max(axis=1)
        df['mouth'] = df[colSets['mouth']].max(axis=1)
    
    if value == 'sum':
        df['eyes'] = df[colSets['eyes']].sum(axis=1)
        df['midpart'] = df[colSets['midpart']].sum(axis=1)
        df['mouth'] = df[colSets['mouth']].sum(axis=1)

    if value == 'avg':
        df['eyes'] = (df[colSets['eyes']].sum(axis=1))/len(COLSETS['eyes'])
        df['midpart'] = (df[colSets['midpart']].sum(axis=1))/len(COLSETS['midpart'])
        df['mouth'] = (df[colSets['mouth']].sum(axis=1))/len(COLSETS['mouth'])


    #extract velocity for each sets of AU
    df['eyes_d'] = df['eyes'].diff()
    df['midpart_d'] = df['midpart'].diff()
    df['mouth_d'] = df['mouth'].diff()

    stat_cols = list(df.columns[-6:])


    output = pd.DataFrame(columns=['movie', 'face_id', 'time', 'eyes_mean', 'eyes_SD', 'midpart_mean', 'midpart_SD',
    'mouth_mean', 'mouth_SD', 'eyes_d_mean', 'eyes_d_SD', 'midpart_d_mean', 'midpart_d_SD', 'mouth_d_mean', 'mouth_d_SD'])

    
    for id in shots:
        shot = df[df['face_id'] == id] #select rows with a particular face_id
        #if 'Bright' in movie and id == 74.0: #костыль
            #continue

        output = pd.concat([output, mean_SD(shot, movie, id, stat_cols)], ignore_index=True)
    


    return output







####################   FILTER #####################

#input: au - a list of AU values; frac: SD will be divided by sqrt(frac); deg = degree of a polynom
def smooth_au(au, frac=4, deg=3):

    t = len(au)
    w = deg + 2 #window length has to be more than a polynom degree

    SD = statistics.stdev(au)
    stdev = SD/math.sqrt(frac)
    #stdev = SD/2

    #look for the best window; step=2; w has to be odd
    while w < t:
        smoothed = savgol_filter(au, w, deg)

        sum = 0
        for i in range(t):
            sum += abs(au[i] - smoothed[i])

        if sum/t > stdev: #interrupt a cycle 
            break

        w += 2

    #show output plot

    #plt.plot(au, color='b', label='AU')
    #plt.plot(smoothed, color='r', label='smoothed')
    #plt.show()
    #print('orig_mean:', statistics.mean(au), 'smooth_mean:', statistics.mean(smoothed))

    return smoothed



#filter all AUs using Savitsky-Golay filter with polynom degree=3
def normalize(df):
    shots = df.face_id.unique() #find unique face_ids
    for id in shots:
        for AU in COLSETS['all']:
            au = list(df[df['face_id'] == id][AU])
            minim = min(au)
            maxim = max(au)

            if maxim == 0: #for cases when no emotions are shown
                #print(id, AU, maxim)
                continue
            #normalize data
            for i in range(len(au)):
                au[i] = (au[i] - minim) / (maxim - minim)
            
            df.loc[df['face_id'] == id, AU] = au
    
    return df    



#filter all AUs using Savitsky-Golay filter with polynom degree=3
def sg_filter(df, frac=4):
    shots = df.face_id.unique() #find unique face_ids
    for id in shots:
        for AU in COLSETS['all']:
            au = list(df[df['face_id'] == id][AU])
            upd_au = smooth_au(au, frac, deg=3)
            df.loc[df['face_id'] == id, AU] = upd_au
    
    return df   








#return selected rows for all HM and AM movies
def hm_am(file):
    HM = ['MonsterInLaw', 'QuantumOfSolace', 'Click', 'PiratesOfCaribbean', 'TheDarkKnight', 'FastAndFurious']
    AM = ['Clean', 'Synecdoche', 'BrightStar', 'CertifiedCopy', 'Spider', 'AllOrNothing']

    COLS=['movie', 'face_id', 'time', 'eyes_mean', 'eyes_SD', 'midpart_mean', 'midpart_SD',
    'mouth_mean', 'mouth_SD', 'eyes_d_mean', 'eyes_d_SD', 'midpart_d_mean', 'midpart_d_SD', 'mouth_d_mean', 'mouth_d_SD']

    df = pd.read_csv(file, sep=',')

    
    #dset for Hollywood movies
    outputHM = pd.DataFrame(columns=COLS)
    for i in HM:
        outputHM = pd.concat([outputHM, df[df['movie'] == i]]) #select rows with a particular face_id

    #dset for Art movies
    outputAM = pd.DataFrame(columns=COLS)
    for i in AM:
        outputAM = pd.concat([outputAM, df[df['movie'] == i]]) #select rows with a particular face_id
    
    
    return outputHM, outputAM



def ttest():
    ttest_ind(data_group1, data_group2, equal_var=True/False)



#plots the graphs of given AUs
#input: df - shot, cols - columns of interest
def plot(df, cols):
    
    au_vectors = {}
    au_array = []
    color = cycle('bgrcmk')

    for au in cols:
        au_vectors[au] = list(df[au])
        au_array.append(au_vectors[au])

    for vec in au_vectors:
        plt.plot(au_vectors[vec], color=next(color), label=vec)
        plt.legend(au_vectors)
    plt.show()


    au_array = np.array(au_array)
    vector_sum = au_array.sum(axis=0)


    plt.plot(vector_sum, color=next(color), label=vec)
    plt.legend('sum')
    plt.show()