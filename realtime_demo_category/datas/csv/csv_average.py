import csv
import numpy as np
def csv_average(name):
    with open(name + ".csv", "r") as f:
        reader = csv.reader(f)
        first_row = reader.__next__()
        print(first_row)
        ave = [0] * len(first_row)
        count = 0;
        for row in reader:
            if len(row) == 0:
                continue
            count += 1
            print(row)
            for i, d in enumerate(row):
                ave[i] += float(d)
        ave = list(map(lambda x: round(x / count, 4), ave))
        print(ave)
        with open("ave_" + name + ".csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(first_row)
            writer.writerow(ave)
    return
def csv_stat(name):
    with open(name + ".csv", "r") as f:
        reader = csv.reader(f)
        first_row = reader.__next__()
        print(first_row)
        stat = []
        data = []
        count = 0
        for i in range(5):
            stat.append([])
        for row in reader:
            if len(row) == 0:
                continue
            count += 1
            if count == 1:
                continue
            print(row)
            data_row = []
            for d in row:
                data_row.append(float(d))
            data.append(data_row)
        for i in range(len(data[0])):
            data_col = np.array([])
            for j in range(len(data)):
                data_col = np.append(data_col, data[j][i])
                #print(data[j][i])
            print(data_col)
            stat[0].append(round(np.min(data_col), 4))
            stat[1].append(round(np.percentile(data_col, 25), 4))
            stat[2].append(round(np.median(data_col), 4))
            stat[3].append(round(np.percentile(data_col, 75), 4))
            stat[4].append(round(np.max(data_col), 4))
            print(stat)


        with open("stat_" + name + ".csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(first_row)
            print(stat)
            for row in stat:
                writer.writerow(row)
if __name__ == '__main__':
    csv_average("1108d1")
    csv_average("1108d2")
