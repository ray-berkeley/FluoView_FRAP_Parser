#!/bin/python

import argparse
import numpy as np
import os
import pandas as pd
import pickle
import re

class FRAPData:
    def __init__(self):
        self.replicates = int
        self.maxtime = int
        self.D = int
        self.tau_half_dict = {}
        self.tau_half_zscores = {}
        self.outliers = int
        self.tau_half = int
        self.file_list = list
        self.all_data = pd.DataFrame()

    def describe(self, full):
        '''Prints a descrption of the FRAPData object in question'''
        print(f"Replicates Included: {self.replicates}")
        print(f"Time acquired: {self.maxtime} sec")
        # print(f"Diffusion Coefficient (D): {self.D}")
        print(f"Half Recovery (tau 1/2): {self.tau_half:.2f}")
        # print({self.file_list})
        print('\n')
        for reps, taus in self.tau_half_dict.items():
            zscore = self.tau_half_zscores.get(reps)
            print(f"{reps} ~ {taus:.2f} ~ {zscore:.2f}")

        print("###############################")
        if full==True:
            self.all_data.to_html('report.html')

    def bin_data(self, bin_size, cutoff_value):
        '''Bins timepoints to a bin size set by the user. This method also trims
        all replicates to the same length (by timepoint)'''

        # Bin timepoints based on the value defined in bin_size.
        self.all_data['bin_time'] = self.all_data['time'].div(bin_size)
        self.all_data['bin_time'] = self.all_data['bin_time'].round(0)
        self.all_data['bin_time'] = self.all_data['bin_time'].mul(bin_size)

        # Get the shortest timecourse and trim the results to that value. 
        maxtimes = self.all_data.groupby(['replicate']).max()['bin_time']
        maxtimes = maxtimes.nsmallest(cutoff_value)
        
        # Remove all rows where bin_time exceeds the cutoff_value.
        self.all_data = self.all_data[self.all_data['bin_time'] <= maxtimes.max()]
        self.all_data = self.all_data.reset_index(drop=True)
        
        self.maxtime = maxtimes.max()/1000

    def populate(self, normalization_frames, append_dir):
        '''Fills the FRAPData df with parsed, normalized data from the csvs in the file_list'''
        replicate = 0

        for csv in self.file_list:
            replicate+=1
            parsed_csv, tau = raw_LiveStop_parser(csv, normalization_frames)
            self.tau_half_dict.update({replicate:tau})
            parsed_csv['replicate'] = replicate

            if append_dir==True:
                directory = os.getcwd().split(os.sep)[-1]
                parsed_csv['dir'] = directory

            self.all_data = self.all_data.append(parsed_csv, ignore_index=True)

        self.replicates = replicate
    
    def compute_means(self):
        means = self.all_data.groupby('bin_time', as_index=False)['recovery'].mean()
        means.to_html('means.html')

    def compute_zscores(self, purge_outliers):
        taus = np.array([])
        
        for rep, tau in self.tau_half_dict.items():
            taus = np.append(taus, [tau])

        mean = np.mean(taus)
        stdev = np.std(taus)

        for rep, tau in self.tau_half_dict.items():
            zscore = abs((tau-mean)/stdev)
            self.tau_half_zscores.update({rep:zscore})

        self.tau_half = mean

        if purge_outliers==True:
            print("The purge method needs work")

            good_taus = np.array([])

            for rep, zscore in self.tau_half_zscores.items():
                if zscore > 2:
                    self.all_data = self.all_data[self.all_data.replicate != rep]
                    self.replicates -= 1
                else:
                    good_taus = np.append(good_taus, [self.tau_half_dict.get(rep)]) 

            self.tau_half = np.mean(good_taus)
        
def raw_LiveStop_parser(csv, normalization_frames):
    '''
    A general method that does the heavy lifting when it comes to parsing Olympus csv files.
    
    This takes in a filename (some .csv file) and returns a pandas data frame. It first reads
    the file line by line and identifies the number of channels (and the number of lines to 
    skip when ingesting the data into pandas). The parsed dataframe that is generated at this
    step is saved as raw_df.
    '''
    
    lines_in_header = 0

    # Trim header from input file. 
    with open(csv) as f:
        for line in f:
            lines_in_header+=1
            
            csv_line = line.split(',')
            csv_line = list(filter(None, csv_line))

            if lines_in_header == 1:
                pass
                # assert csv_line[0].startswith('LiveStop'), f'{csv} is not a raw Olympus csv file'
            if csv_line[0] == '\n':
                break
    
    # Create symmetrical dataframe and remove blank channels and NaN columns.
    raw_df = pd.read_csv(csv, sep=',', skiprows=lines_in_header)
    raw_df = raw_df.loc[:, (raw_df != 0).any(axis=0)]
    raw_df = raw_df.dropna(axis=1)
    assert len(raw_df.columns) == 3, f'There is an issue with the number of columns in {csv}'
    
    # Normalize to zero assuming the column length assertion passed.
    F_0 =  raw_df.iloc[:normalization_frames, 2]
    F_0_avg = sum(F_0/normalization_frames)

    # Get rid of the normalization frames.
    raw_df = raw_df[normalization_frames:]

    # Divide the raw intensity values by the initial intensity to yield a 
    # "% recovery value. 
    raw_df.iloc[:,2] = raw_df.iloc[:,2].div(F_0_avg)

    # Adjust times to "time after bleaching".
    raw_df.iloc[:,1] = raw_df.iloc[:,1].subtract(raw_df.iloc[0,1])
    
    # Rename the column headers to something meaningful.
    raw_df.columns = ['index', 'time', 'recovery']

    # Remove zero values from recovery frame.
    raw_df = raw_df[raw_df.recovery != 0]

    # Zero recovery values for each replicate. This allows us to take the 
    # standard deviation of our zeroed values instead of having to correct
    # using the coefficient of variance later.
    raw_df['recovery0'] = raw_df['recovery'] - raw_df.recovery.min()

    # Calculate the half life for this rep. This value is used to determine the final
    # recovery half life and can be printed using the --stats argument if desired. 
    half_recovery = raw_df.recovery0.max()/2
    tau_df = raw_df.iloc[(raw_df['recovery0']-half_recovery).abs().argsort()[:2]]

    tau_half = tau_df.time.mean()

    return raw_df, tau_half

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=(
            '''Read and manipulate lists of comma separated files that are produced by
             Olympus' FluoView software.'''))
    parser.add_argument('--cutoff-value', default = 0, type = int, help =
        '''Specifies the  value that will be used to truncate timeseries in 
        the final plot. Increasing this value allows the user to prevent early 
        truncation of the entire dataset if one time series was halted early 
        during the FRAP acquisition.''')
    parser.add_argument('--path', '-p', help = 'target path for csv aggregation')
    parser.add_argument('--append-dir', default = False, type = bool, help =
        '''Insert the name of the current directory as a frame.''')
    parser.add_argument('--bin-size', default=2700, type=int, help = 
        '''Window size for binning timepoints between different replicates.''')
    parser.add_argument('--normalization-frames', default = 4, type=int, help = 
    '''number of frames of baseline frames collected.''')
    parser.add_argument('--stats', choices=('short', 'full'), help = '''Summarize the data.''')
    parser.add_argument('--purge-outliers', type = bool, default = False)
    args = parser.parse_args()

    MyFRAPData = FRAPData()
    file_list = []

    ''' 
    The following block sets the directory to the path spectified in the '--path' argument (if present)
    and then retrieves the .csv file names therein. It is structured so that we can get more picky with
    the directories that are included if this is necessary in the future, or so that this program can be
    run on a bunch of directories from a bash script
    '''

    if args.path:
        if os.name=='nt':
            print("Include full path, starting from c:\\...")
        path = args.path
    else:
        path = os.getcwd()

    os.chdir(path)

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                file_list.append(file)
                
    MyFRAPData.file_list = file_list
    MyFRAPData.populate(args.normalization_frames, args.append_dir)
    MyFRAPData.bin_data(args.bin_size, args.cutoff_value)
    MyFRAPData.compute_means()
    MyFRAPData.compute_zscores(args.purge_outliers)


    if args.stats=='full':
        print(path)
        MyFRAPData.describe(True)
    elif args.stats:
        print(path)
        MyFRAPData.describe(False)

    directory = path.split(os.sep)[-2]
    condition = str([x for x in os.getcwd().split(os.sep) if x.startswith('Cond')])
    condition = re.sub("[^0-9]", "", condition)
    MyFRAPData.all_data.to_pickle(f"./all_data_{condition}_{directory}.pkl")