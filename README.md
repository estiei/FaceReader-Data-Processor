# FaceReader-Data-Processor

Row data are processed in facereader.ipynb.


1. Look for all the frames in the raw data that last more than 3 sec
(two following files are the same)
facereader.ipynb
facereader_suitable_frames_search.ipynb - use this as the main file
Input (raw data): 12movies_init_FaceReader_output
Output folder: /12movies_selected_frames/ (there are still original FRAME!)

2. Look for frames where there was a camera view change (Keira - ships - Keira) and choose uninterrupted sequences:
split_cuts(KeiraKn).ipynb
INPUT = './12movies_selected_frames/'
OUTPUT = './12movies_cuts'

############# further work with this folder ########## frames are as original

3. Here we store functions for data processing:
movieslib.py

4. Calculating statistics (descriptive) - applying filters and not:
stat_calc_filtering.ipynb
stat_calc.ipynb
FOLDER_INPUT = './12movies_cuts/'
FOLDER_OUTPUT = './extr_stat_normalised'




These files are auxiliary or experimental for quick checking on something:

Change time to to human-readable format - for choosing frames from videos (inhouse use, to check manually)
12mov_timestamps_transform.ipynb

Calculating significance in statistical data between two movie types - applying filters and not (HM, AM) - inferential stat:
ttest_2samples.ipynb

Plotting graphs for initial AUs that came from FaceReader
plot_graphs.ipynb


Experiments: to see if there's a difference between filtering data before splitting into shots and after. Spoiler: yes, there is, and it's better to filter after splitting - otherwise the filter invent signals where there's zero in the initial data
filtering_experiment_graphical_compare_before-after_splitting.ipynb



File to delete:
experiments1.ipynb

