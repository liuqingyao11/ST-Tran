# -*- coding: utf-8 -*-
"""
/*******************************************
**  license
********************************************/
"""
import pandas as pd
import numpy as np


class STMatrix(object):
    def __init__(self, data, timestamps, T=24, CheckComplete=True):
        super(STMatrix, self).__init__()
        assert len(data) == len(timestamps)
        self.data = data
        self.timestamps = timestamps #list[(index,timestamp)]
        self.T = T
        self.pd_timestamps = timestamps
        self.make_index()

    def make_index(self):
        self.get_index = dict()
        for i, ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i


    def get_matrix(self, timestamp, len_closeness, closeness):
        index = self.get_index[timestamp]
        if(closeness):
            return self.data[index]
        else:
            return self.data[np.arange(index,index+len_closeness)]


    def check_it(self, depends):
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True


    def create_dataset(self, len_closeness=3, len_trend=3, TrendInterval=7, len_period=3, PeriodInterval=1):
        offset_frame = pd.DateOffset(hours=1)
        XC = []
        XP = []
        XT = []
        Y = []
        timestamps_Y = []
        depends = [range(1, len_closeness+1),
                   [PeriodInterval * self.T * j for j in range(1, len_period+1)],
                   [TrendInterval * self.T * j for j in range(1, len_trend+1)]]

        i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness)
        while i < len(self.pd_timestamps)-len_closeness:
            Flag = True
            for depend in depends:
                if Flag is False:
                    break
                Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depend])

            if Flag is False:
                i += 1
                continue
            x_c = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame,len_closeness,closeness=True) for j in depends[0]]
            x_p = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame,len_closeness,closeness =False) for j in depends[1]]
            x_t = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame, len_closeness,closeness =False) for j in depends[2]]
            y = self.get_matrix(self.pd_timestamps[i],len_closeness,closeness=False)
            #print(x_c[0].shape,x_p[0].shape) #(2,400) (6,2,400)
            if len_closeness > 0:
                XC.append(np.stack(x_c))
            if len_period > 0:
                XP.append(np.stack(x_p))
            if len_trend > 0:
                XT.append(np.stack(x_t))
            Y.append(y)
            timestamps_Y.append(self.timestamps[i])
            i += 1
       # print(timestamps_Y)
        XC = np.asarray(XC)
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        Y = np.asarray(Y)
        print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
        return XC, XP, XT, Y, timestamps_Y


    # def create_dataset(self, len_closeness=3, len_trend=3, TrendInterval=7, len_period=3, PeriodInterval=1):
    #     offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
    #     XC = []
    #     XP = []
    #     XT = []
    #     Y = []
    #     timestamps_Y = []
    #     depends = [range(1, len_closeness+1),
    #                [PeriodInterval * self.T * j for j in range(1, len_period+1)],
    #                [TrendInterval * self.T * j for j in range(1, len_trend+1)]]

    #     i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness)
    #     while i < len(self.pd_timestamps):
    #         Flag = True
    #         for depend in depends:
    #             if Flag is False:
    #                 break
    #             Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depend])

    #         if Flag is False:
    #             i += 1
    #             continue
    #         x_c = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[0]]
    #         x_p = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[1]]
    #         x_t = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[2]]
    #         y = self.get_matrix(self.pd_timestamps[i])
    #         if len_closeness > 0:
    #             XC.append(np.vstack(x_c))
    #         if len_period > 0:
    #             XP.append(np.vstack(x_p))
    #         if len_trend > 0:
    #             XT.append(np.vstack(x_t))
    #         Y.append(y)
    #         timestamps_Y.append(self.timestamps[i])
    #         i += 1
    #     XC = np.asarray(XC)
    #     XP = np.asarray(XP)
    #     XT = np.asarray(XT)
    #     Y = np.asarray(Y)
    #     print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
    #     return XC, XP, XT, Y, timestamps_Y