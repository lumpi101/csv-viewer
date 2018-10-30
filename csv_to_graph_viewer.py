# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:11:13 2017

@author: Kevin
"""

import multiprocessing
import sys
import matplotlib.pyplot as plt
import tkinter as tk
import numpy as np
import pandas as pd
import re
import dateutil as du
from csv import Sniffer

def getDelimiter(path):
    sniffer = Sniffer()
    with open(path, 'r') as rfile:
        header = rfile.readline()
        sample = header + rfile.readline() + rfile.readline()
    try:
        asniff = sniffer.sniff(sample, delimiters=";, ")
    except Exception:
        class tsniff(object):
            lineterminator = "\n"
            delimiter = ","
        asniff = tsniff()
        asniff.lineterminator = "\n"
    return asniff.delimiter, sniffer.has_header(sample)

class NoTimestampError(Exception):
    """Exception raised when no timestamp is found but was expected to."""
    pass

def timestampConversion(dataFrame, assumeTimestamp=False, inplace=False):
    if not inplace:
        dataFrame = dataFrame.copy()
    df = dataFrame
    accomplished = False

    if str(df.dtypes[0]) != 'datetime64[ns]' and (assumeTimestamp or re.match(r"timestamp|unix.*time.*|posix.*time.*", df.columns[0], re.IGNORECASE)):
        try:
            if 'int' in str(df.dtypes[0]):
                for unit in ['D', 's', 'ms', 'us', 'ns']:
                    try:
                        sample = pd.to_datetime(df.iloc[0,0], unit=unit)
                    except ValueError:
                        continue
                    factor = {'D': 0, 's': 1, 'ms': 10**3, 'us': 10**6, 'ns': 10**9}[unit]
                    msDiff = sample.replace(tzinfo=du.tz.tzutc()).astimezone(du.tz.gettz('Europe/Berlin')).utcoffset().total_seconds() * factor
                    df.iloc[:,0] = pd.to_datetime(df.iloc[:,0] + msDiff, unit=unit)
                    accomplished = True
                    break
            else:
                df.iloc[:,0] = pd.to_datetime(df.iloc[:,0])
                accomplished = True
        except:
            if assumeTimestamp:
                raise NoTimestampError
    
    if not inplace:
        return df, accomplished
    else:
        return accomplished

def getData(path, separator=None, convertTimestamps=False, assumeTimestamp=False):

    # Use the csv sniffer to find the delimiter. Pandas could do it, but cant use the c-routines to read the data then
    delimiter, _ = getDelimiter(path)

    try:
        df = pd.read_csv(path, sep=delimiter, na_values=['Infinity', '-Infinity'])
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=delimiter, na_values=['Infinity', '-Infinity'], encoding='latin-1')
    
    if convertTimestamps:
        return df, timestampConversion(df, assumeTimestamp=assumeTimestamp, inplace=True)

    return df

def discardInvalidDataPoints(x_data, y_data):
    x_mask = np.ones(len(x_data), dtype=bool) if str(x_data.dtype) in ['datetime64[ns]', 'object'] else np.isfinite(x_data)
    y_mask = np.ones(len(y_data), dtype=bool) if str(y_data.dtype) in ['datetime64[ns]', 'object'] else np.isfinite(y_data)
    mask = np.logical_and(x_mask, y_mask)
    return x_data[mask], y_data[mask]

def plot(x=None, y1=None, y2=None):
    """x is assumed to be a dictionary and y1/y2 are assumed to be lists of dictionaries
    every dictionary has to have 2 keys: 'label' -> contains the label of the date; 'data' -> contains the actual dataset"""

    x = {} if x is None else x
    y1 = [] if y1 is None else y1
    y2 = [] if y2 is None else y2
    if len(x) == 0:
        return
    if len(y1) == 0:
        y1 = y2
        y2 = []
    if len(y1) == 0:
        return
    
    ax2, leg1, leg2 = None, None, None
    num_colors = len(y1) + len(y2)
    cm = plt.get_cmap('gist_rainbow')
    cgen = (cm(1.*i/num_colors) for i in range(num_colors))
    
    fig, ax1 = plt.subplots()
    for i in range(len(y1)):
        x_data, y_data = discardInvalidDataPoints(x['data'], y1[i]['data'])
        line1, = ax1.plot(x_data, y_data, '.', color=next(cgen), label=y1[i]['label'])
    ax1.set_xlabel(x['label'])
    # ax1.xticks(rotation=45)
    
    if len(y2) > 0:
        ax2 = ax1.twinx()
        for i in range(len(y2)):
            x_data, y_data = discardInvalidDataPoints(x['data'], y2[i]['data'])
            line2, = ax2.plot(x_data, y_data, '.', color=next(cgen), label=y2[i]['label'])
    
    if len(y1) == 1:
        if len(y2) == 0:
            ax1.set_ylabel(y1[0]['label'])
        elif len(y2) == 1:
            ax1.set_ylabel(y1[0]['label'], color=line1.get_color())
            ax2.set_ylabel(y2[0]['label'], color=line2.get_color())
            ax1.tick_params('y', colors=line1.get_color())
            ax2.tick_params('y', colors=line2.get_color())
        else:
            leg1 = ax1.legend(loc='upper left')
            leg2 = ax2.legend(loc='lower right')
    else:
        leg1 = ax1.legend(loc='upper left')
        if len(y2) > 0:
            leg2 = ax2.legend(loc='lower right')
    
    if ax2 is not None and leg1 is not None:
        leg1.remove()
        ax2.add_artist(leg1)
    
    plt.xticks(rotation=45)
    plt.show()

def calculateIntegral(x_data, y_data, interpolate_method='constant'):
    x_data, y_data = discardInvalidDataPoints(x_data, y_data)
    integral = np.zeros(len(x_data), dtype=np.float64)
    running_integral = 0
    if len(x_data) != len(y_data) or len(x_data) < 2:
        print(0.)
        return
    last_x, last_y = x_data[0], y_data[0]
    convert_x = (lambda x: x/np.timedelta64(3600, 's')) if (x_data.dtype == 'datetime64[ns]') else (lambda x: x)
    for i in range(1, len(x_data)):
        running_integral += last_y*convert_x(x_data[i] - last_x)
        integral[i] = running_integral
        last_x, last_y = x_data[i], y_data[i]
    print("integral = {0}".format(running_integral))
    return integral

class App:
    def __init__(self, master, dataFrame, timestamped):
        self.noFeature = "-- no feature --"
        self.df = dataFrame
        self.timestamped = timestamped
        self.labels = list(self.df.columns)
        self.shortenLabels()
        self.features = [self.noFeature] + self.labels
        
        self.frame = tk.Frame(master)
        self.frame.pack()
        
        self.xTitle = tk.Label(self.frame, text="x - axis")
        self.xTitle.pack()
        self.xfVariable = tk.StringVar(self.frame)
        self.xfVariable.set(self.features[1])
        self.xFeature = tk.OptionMenu(self.frame, self.xfVariable, *self.features)
        self.xFeature.pack()

        self.y1Title = tk.Label(self.frame, text="y1 - axis")
        self.y1Title.pack()
        self.y1OptionsFrame = tk.Frame(self.frame)
        self.y1OptionsFrame.pack()
        self.y1fVariables = []
        self.y1OptionMenus = []
        self.more(self.y1OptionsFrame, self.y1fVariables, self.y1OptionMenus)
        self.y1ButtonFrame = tk.Frame(self.frame)
        self.y1ButtonFrame.pack()
        self.y1MoreButton = tk.Button(self.y1ButtonFrame, text="more", command=self.more1)
        self.y1MoreButton.pack(side=tk.LEFT)
        self.y1LessButton = tk.Button(self.y1ButtonFrame, text="less", command=self.less1)
        self.y1LessButton.pack(side=tk.LEFT)
        
        self.y2Title = tk.Label(self.frame, text="y2 - axis")
        self.y2Title.pack()
        self.y2OptionsFrame = tk.Frame(self.frame)
        self.y2OptionsFrame.pack()
        self.y2fVariables = []
        self.y2OptionMenus = []
        self.more(self.y2OptionsFrame, self.y2fVariables, self.y2OptionMenus)
        self.y2ButtonFrame = tk.Frame(self.frame)
        self.y2ButtonFrame.pack()
        self.y2MoreButton = tk.Button(self.y2ButtonFrame, text="more", command=self.more2)
        self.y2MoreButton.pack(side=tk.LEFT)
        self.y2LessButton = tk.Button(self.y2ButtonFrame, text="less", command=self.less2)
        self.y2LessButton.pack(side=tk.LEFT)
        
        self.mainButtonFrame = tk.Frame(self.frame)
        self.mainButtonFrame.pack()
        self.plotButton = tk.Button(self.mainButtonFrame, text="Plot", command=self.plot)
        self.plotButton.pack(side=tk.LEFT)
        self.integralButton = tk.Button(self.mainButtonFrame, text="Integral", command=self.integral)
        self.integralButton.pack(side=tk.LEFT)
        self.quitButton = tk.Button(self.mainButtonFrame, text="QUIT", fg="red", command=self.frame.quit)
        self.quitButton.pack(side=tk.LEFT)
        
    def more1(self):
        self.more(self.y1OptionsFrame, self.y1fVariables, self.y1OptionMenus)
    
    def more2(self):
        self.more(self.y2OptionsFrame, self.y2fVariables, self.y2OptionMenus)
    
    def less1(self):
        self.less(self.y1fVariables, self.y1OptionMenus)
    
    def less2(self):
        self.less(self.y2fVariables, self.y2OptionMenus)
        
    def more(self, frame, fVariables, fOptionMenus):
        fVariables.append(tk.StringVar(frame))
        fVariables[-1].set(self.noFeature)
        fOptionMenus.append(tk.OptionMenu(frame, fVariables[-1], *self.features))
        fOptionMenus[-1].pack()
    
    def less(self, fVariables, fOptionMenus):
        if len(fVariables) > 1:
            fVariables.pop()
            fOptionMenus[-1].pack_forget()
            fOptionMenus.pop()
    
    def shortenLabels(self):
        h = 1 if self.timestamped else 0
        longVersionLabels = self.labels[:]
        n = len(self.labels) - h
        if n > 1:
            i = 0
            for i, z in enumerate(zip(*self.labels[h:])):
                if z.count(z[0]) != n:
                    break
            if i > 0:
                # if len(self.labels[0]) > i and self.labels[0][:i] == self.labels[1][:i]:
                #     self.labels[0] = self.labels[0][i:]
                for j in range(h, len(self.labels)):
                    self.labels[j] = self.labels[j][i:]
        self.longVersionLabels = {z[0]:z[1] for z in zip(self.labels, longVersionLabels)}
    
    def plot(self):
        if self.xfVariable.get() == self.noFeature:
            xdata = np.arange(len(self.df[self.longVersionLabels[self.y1fVariables[0].get()]].values))
            x = {'label': 'empty', 'data': xdata}
        else:
            x = {'label': self.xfVariable.get(), 'data': self.df[self.longVersionLabels[self.xfVariable.get()]].values}
        y1 = []
        y2 = []
        for i in range(len(self.y1fVariables)):
            if self.y1fVariables[i].get() != self.noFeature:
                y1.append({'label': self.y1fVariables[i].get(), 'data': self.df[self.longVersionLabels[self.y1fVariables[i].get()]].values})
        for i in range(len(self.y2fVariables)):
            if self.y2fVariables[i].get() != self.noFeature:
                y2.append({'label': self.y2fVariables[i].get(), 'data': self.df[self.longVersionLabels[self.y2fVariables[i].get()]].values})
        multiprocessing.Process(target=plot, args=(x, y1, y2)).start()
    
    def integral(self):
        if self.y1fVariables[0].get == self.noFeature:
            return
        if self.xfVariable.get() == self.noFeature:
            x_data = np.arange(len(self.df[self.longVersionLabels[self.y1fVariables[0].get()]].values))
            x = {'label': 'empty', 'data': x_data}
        else:
            x_data = self.df[self.longVersionLabels[self.xfVariable.get()]].values
            x = {'label': self.xfVariable.get(), 'data': x_data}
        y_data = self.df[self.longVersionLabels[self.y1fVariables[0].get()]].values
        integral = calculateIntegral(x_data, y_data)
        y = [{'label': "int_" + self.y1fVariables[0].get(), 'data': integral}]
        multiprocessing.Process(target=plot, args=(x, y)).start()

def view(dataframeOrPath):
    if type(dataframeOrPath) is str:
        df, detectedTimestamp = getData(dataframeOrPath, convertTimestamps=True)
    else:
        df, detectedTimestamp = dataframeOrPath, False
    # print(df.dtypes)
    
    root = tk.Tk()
    App(root, df, detectedTimestamp)
    root.mainloop()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.argv.append(tk.filedialog.askopenfilename(filetypes=(("CSV File", "*.csv *.txt *.ascii"), ("All Files", "*.*")), title = "Choose a file"))
    if sys.argv[-1] == '':
        exit(0)
    view(sys.argv[1])
