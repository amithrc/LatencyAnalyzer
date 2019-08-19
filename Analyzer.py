import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt

fields = ['packetid', 'latency', 'unit']


def convert_time_unit(df):
    df[fields[1]] = df[fields[1]] / 1000000
    df[fields[2]] = 'ms'


def read_log_file(logfile):
    df = pd.read_csv(logfile, skipinitialspace=True, usecols=fields)
    return df, df['unit'][0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Latency Monitor Analyzer")
    parser.add_argument("-f", "--logfile", help="Path to Log file", required=True)
    parser.add_argument("-t", "--timeunit", help="Convert the timeunit(ms,us,ns)", default='ms')

    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    logframe = None
    unit = None
    if args.logfile:
        print(args.logfile)
        logframe, unit = read_log_file(args.logfile)

    if args.timeunit != unit:
        convert_time_unit(logframe)

    print(logframe.tail())
    logframe.to_csv()

    plt.hist(logframe[fields[1]])
    plt.show()
