import shutil
import os
import statistics
import numpy as np

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



#####################    STATISTICS    ##########################

#calculates mean of one line values
#input: line: pandas dataframe line; cols:columns of interest
def line_mean(line, cols):
    values = list(line[cols])
    return statistics.mean(values)

#calculates standard deviation of one line values
#input: line: pandas dataframe line; cols:columns of interest
def line_SD(line, cols):
    values = list(line[cols])
    return statistics.stdev(values)









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

        if temp['frame'] != current:
            output.append([init_frame, shot.iloc[i-1]['frame'], shot.iloc[i-1]['timestamp'] - init_time])
            init_frame = temp['frame']
            current = init_frame
            init_time = temp['timestamp']
        current += 1

    output.append([init_frame, shot.iloc[i-1]['frame'], shot.iloc[i-1]['timestamp'] - init_time])

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
#       facial act: a dictionary in a format: facial_act = {'face_id':[], 'mean_eyes(max)':[], 'SD_eyes(max)':[], 'mean_mouth(max)':[], 'SD_mouth(max)':[], 'mean_AU_r(max)':[], 'SD_AU_r(max)':[],
                #'mean_eyes(average)':[], 'SD_eyes(average)':[], 'mean_mouth(average)':[], 'SD_mouth(average)':[], 'mean_AU_r(average)':[], 'SD_AU_r(average)':[],
                 #   'mean_blink':[], 'SD_blink':[]}
#output: updated facial_act (with one more line added)

def mean_SD(shot, id, facial_act):

    colSets = {'eyes':['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r'], 
    'mouth':['AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r'], 
    'blink':['AU45_r']}

    #blink is excluded
    eyes = {'max':[], 'mean':[], 'der':[]}
    mouth = {'max':[], 'mean':[], 'der':[]}
    AU = {'max':[], 'mean':[], 'der':[]}
    
    eyes_prev = max(shot.iloc[0][[colSets['eyes']]])
    mouth_prev = max(shot.iloc[0][[colSets['mouth']]])
    AU_prev = max(shot.iloc[0][colSets['eyes'] + colSets['mouth']])
    time_prev = shot.iloc[0]['timestamp']

    for i in range(len(shot)):

        temp = shot.iloc[i] #locate a current line

        #for eyes:
        values = list(temp[colSets['eyes']])
        eyes['max'].append(max(values)) #append max value from the row
        eyes['mean'].append(statistics.mean(values)) #append mean of the row
        eyes['der'].append((eyes_prev - eyes['max'][i]) / (temp['timestamp'] - time_prev)) #append velocity of the row

        #for mouth: 
        values = list(temp[colSets['mouth']])
        mouth['max'].append(max(values)) #append max value from the row
        mouth['mean'].append(statistics.mean(values)) #append mean of the row
        mouth['der'].append((mouth_prev - mouth['max'][i]) / (temp['timestamp'] - time_prev)) #append velocity of the row

        #for all AUs:
        values = list(temp[colSets['eyes'] + colSets['mouth']])
        AU['max'].append(max(values)) #append max value from the row
        AU['mean'].append(statistics.mean(values)) #append mean of the row
        AU['der'].append((AU_prev - AU['max'][i]) / (temp['timestamp'] - time_prev)) #append velocity of the row

    facial_act['face_id'].append(id)

    #for MAX values
    facial_act['mean_eyes(max)'].append(statistics.mean(eyes['max']))
    facial_act['SD_eyes(max)'].append(statistics.stdev(eyes['max']))
    facial_act['mean_mouth(max)'].append(statistics.mean(mouth['max']))
    facial_act['SD_mouth(max)'].append(statistics.stdev(mouth['max']))
    facial_act['mean_AU_r(max)'].append(statistics.mean(AU['max']))
    facial_act['SD_AU_r(max)'].append(statistics.stdev(AU['max']))

    #for derivativity (MAX values in a row)
    facial_act['mean_eyes(der)'].append(statistics.mean(eyes['der']))
    facial_act['SD_eyes(der)'].append(statistics.stdev(eyes['der']))
    facial_act['mean_mouth(der)'].append(statistics.mean(mouth['der']))
    facial_act['SD_mouth(der)'].append(statistics.stdev(mouth['der']))
    facial_act['mean_AU_r(der)'].append(statistics.mean(AU['der']))
    facial_act['SD_AU_r(der)'].append(statistics.stdev(AU['der']))

    #for average values
    facial_act['mean_eyes(average)'].append(statistics.mean(eyes['mean']))
    facial_act['SD_eyes(average)'].append(statistics.stdev(eyes['mean']))
    facial_act['mean_mouth(average)'].append(statistics.mean(mouth['mean']))
    facial_act['SD_mouth(average)'].append(statistics.stdev(mouth['mean']))
    facial_act['mean_AU_r(average)'].append(statistics.mean(AU['mean']))
    facial_act['SD_AU_r(average)'].append(statistics.stdev(AU['mean']))

    #for blink
    blink = list(shot['AU45_r'])
    facial_act['mean_blink'].append(statistics.mean(blink))
    facial_act['SD_blink'].append(statistics.stdev(blink))


    return facial_act




#this function calculates the mean and SD for the whole frame with multiple shots
# input: dataframe
#output: facial act: a dictionary in a format: facial_act = {'face_id':[], 'mean_eyes(max)':[], 'SD_eyes(max)':[], 'mean_mouth(max)':[], 'SD_mouth(max)':[], 'mean_AU_r(max)':[], 'SD_AU_r(max)':[],
                #'mean_eyes(average)':[], 'SD_eyes(average)':[], 'mean_mouth(average)':[], 'SD_mouth(average)':[], 'mean_AU_r(average)':[], 'SD_AU_r(average)':[],
                 #   'mean_blink':[], 'SD_blink':[]}
                 
def stat(df):
    shots = list(df.face_id.unique()) #find unique face_ids in a dataframe

    facial_act = {'face_id':[], 'mean_eyes(max)':[], 'SD_eyes(max)':[], 'mean_mouth(max)':[], 'SD_mouth(max)':[], 'mean_AU_r(max)':[], 'SD_AU_r(max)':[],
                'mean_eyes(der)':[], 'SD_eyes(der)':[], 'mean_mouth(der)':[], 'SD_mouth(der)':[], 'mean_AU_r(der)':[], 'SD_AU_r(der)':[],
                'mean_eyes(average)':[], 'SD_eyes(average)':[], 'mean_mouth(average)':[], 'SD_mouth(average)':[], 'mean_AU_r(average)':[], 'SD_AU_r(average)':[],
                    'mean_blink':[], 'SD_blink':[]}
    
    for id in shots:
        shot = df[df['face_id'] == id] #select rows with a particular face_id

        facial_act = mean_SD(shot, id, facial_act)
    
    return facial_act