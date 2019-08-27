import argparse
import sys
import pandas as pd
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

fields = ['packetid', 'latency', 'unit']


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def generate_plot(method1, method2, method1_name, method2_name, unit, width):
    title = method1_name + " versus " + method2_name
    data_dict_new = dict()
    bins = list(range(1, len(method1) + 1))
    data_dict_new['bins'] = []
    data_dict_new['freq'] = []
    data_dict_new['method'] = []

    for i in range(len(bins)):
        data_dict_new['bins'].append(str(i * width) + "-" + str(width * (i + 1)) + " " + unit)
        data_dict_new['freq'].append(method1[i])
        data_dict_new['method'].append(method1_name)

    for i in range(len(bins)):
        data_dict_new['bins'].append(str(i * width) + "-" + str(width * (i + 1)) + " " + unit)
        data_dict_new['freq'].append(method2[i])
        data_dict_new['method'].append(method2_name)

    ax = sns.barplot(x='bins', y='freq', hue='method', data=data_dict_new)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()


def get_index_for_bin(value, range):
    if (range * 0) <= value < (range * 1):
        return 0
    elif (range * 1) <= value < (range * 2):
        return 1
    elif (range * 2) <= value < (range * 3):
        return 2
    elif (range * 3) <= value < (range * 4):
        return 3
    elif (range * 4) <= value < (range * 5):
        return 4
    elif (range * 5) <= value < (range * 6):
        return 5
    elif (range * 6) <= value < (range * 7):
        return 6
    elif (range * 7) <= value < (range * 8):
        return 7
    elif (range * 8) <= value < (range * 9):
        return 8
    elif (range * 9) <= value < (range * 10):
        return 9
    elif (range * 10) <= value < (range * 11):
        return 10
    elif (range * 11) <= value < (range * 12):
        return 11
    elif (range * 12) <= value < (range * 13):
        return 12
    elif (range * 13) <= value < (range * 14):
        return 13
    elif (range * 14) <= value < (range * 15):
        return 14
    elif value >= (range * 15):
        return 15


def display_file_dict(filedict):
    for key, value in filedict.items():
        print(key, value)


'''
Converts the nanoseconds to the micro seconds for the data frame passed
'''


def convert_to_micro_seconds(df):
    df[fields[1]] = df[fields[1]] / 1000
    df[fields[2]] = 'us'


'''
Returns the dictionary which has the filename as the key and absolute  path to the file as value.
'''


def getFileDict(path):
    filedict = dict()
    for file in os.listdir(path):
        absolute_path = os.path.join(path, file)
        if os.path.isdir(absolute_path) or file.startswith('.'):
            continue
        filedict[Path(absolute_path).stem] = absolute_path
    return filedict


def compute_stats(df):
    print("Mean {}".format(df[fields[1]].mean()))
    numparray = np.array(df[fields[1]].tolist())
    print("Std {}".format(numparray.std()))
    print("Min {}".format(df[fields[1]].min()))
    print("Max {}".format(df[fields[1]].max()))


'''
Reads the each file and gets the data frame.
'''


def read_log_file(logfile):
    return pd.read_csv(logfile)


def compute_bins(df, width=100):
    bins = [0 for i in range(0, 16)]

    for index, row in df.iterrows():
        ind = get_index_for_bin(row['latency'], 150)
        # print("Value is :{} and index is {}:".format(row['latency'], ind))
        bins[ind] = bins[ind] + 1
    return bins

    # nparray = np.array(bins)
    # nparray = nparray / np.sum(nparray)


def create_histogram(df):
    latencies = df['latency'].tolist()
    np.histogram(latencies, bins=16, density=True)
    plt.hist(latencies, bins=16, range=(0, 3000))  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()


def get_counter(width):
    bins = 16
    counter = []
    for val in range(bins):
        c = (val * width) + (width / 2)
        counter.append(int(c))
    return counter


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Latency Monitor Analyzer")
    parser.add_argument("--dirpath", help="Path to the CSV file")
    parser.add_argument("--debug", help="Debug mode", action='store_true')
    parser.add_argument("--stats", help="Compute Stats", action='store_true')
    parser.add_argument("--hist", help="Reads the logfile and create the bins", action='store_true')
    parser.add_argument("--width", help="Read the log file to get the histogram bins")
    parser.add_argument("--file", help="Read the log file to get the histogram bins")
    parser.add_argument("--plot", help="Plot the graph", action='store_true')
    parser.add_argument("--fake", help="Create fake data by using histograms values", action='store_true')
    parser.add_argument("--kl", help="KL divergence", action='store_true')

    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    # comparison-1 Bins taken by Spirent without having tap (loopback and with having tap)

    SpirentNoTapNoDut = [0, 1036185, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # comparison-2 Bins taken by Spirent and myStation (Setup2)
    spirentTapNoDut = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1036185, 0, 0, 0, 0, 0, 0]
    myStationTapNoDut = [0, 60, 992144, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # comparison-3 Bins taken by Spirent when DUT introduced without TAP and DUT introduced with TAP, All stats by Spirent

    SpirentNoTapDUT = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 394057, 480480, 30734, 12593, 7143, 111178]
    SpirentTapDUT = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 388405, 484733, 30326, 12542, 7242, 112937]

    # comparison-4 Bins taken by Spirent when DUT introduced without TAP and DUT introduced with TAP, All stats by Spirent

    myStationTapDut = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 390497, 481360, 30170, 12479, 7222, 112629]

    filedict = None

    if args.dirpath:
        filedict = getFileDict(args.dirpath)

    if args.debug:
        display_file_dict(filedict)

    if args.stats and args.file:
        df = read_log_file(args.file)
        convert_to_micro_seconds(df)
        compute_stats(df)

    if args.hist and args.file and args.width:
        df = read_log_file(args.file)
        convert_to_micro_seconds(df)
        myStationTaput = compute_bins(df, args.width)
        print(myStationTaput)

    if args.plot:
        generate_plot(SpirentNoTapNoDut, spirentTapNoDut, "(Spirent+NoTap)", "(Spirent+TAP)", "ns", 100)  # comp1
        generate_plot(spirentTapNoDut, myStationTapNoDut, "(Spirent+TAP)", "(MyStation+TAP)", "ns", 100)  # comp2
        generate_plot(SpirentNoTapDUT, SpirentTapDUT, "(Spirent+DUT)", "(Spirent+Tap+DUT)", "us", 150)  # comp3
        generate_plot(SpirentTapDUT, myStationTapDut, "(Spirent+Tap+DUT)", "(MyStation+Tap+DUT)", "us", 150)  # comp4

    if args.file and not args.stats:
        df = read_log_file(args.file)
        convert_to_micro_seconds(df)
        compute_stats(df)

    if args.fake:
        counter = get_counter(100)
        print(counter)

        data = []
        for c in range(len(SpirentTapDUT)):
            count = SpirentTapDUT[c]
            data.extend([counter[c]] * count)

        numparray = np.array(data)
        print(data)
        print(numparray.std())

    if args.kl:
        tempSpirent = []

        for i in range(len(SpirentTapDUT)):
            if SpirentTapDUT[i] == 0:
                tempSpirent.append(0.000001)
            else:
                tempSpirent.append(float(SpirentTapDUT[i]))

        tempMyStation = []
        for i in range(len(myStationTapDut)):
            if myStationTapDut[i] == 0:
                tempMyStation.append(0.000001)
            else:
                tempMyStation.append(float(myStationTapDut[i]))

        print(
            kl_divergence(np.array(tempSpirent) / np.sum(tempSpirent), np.array(tempMyStation) / np.sum(tempMyStation)))
