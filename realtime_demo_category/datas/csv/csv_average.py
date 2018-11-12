import csv
def csv_average(name):
    with open(name + ".csv", "r") as f:
        reader = csv.reader(f)
        first_row = reader.__next__()
        print(first_row)
        ave = [0] * len(first_row)
        for row in reader:
            for i, d in enumerate(row):
                ave[i] += float(d)
        ave = list(map(lambda x: round(x / (reader.line_num - 1), 4), ave))
        print(ave)
        with open("ave_" + name + ".csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(first_row)
            writer.writerow(ave)
    return
if __name__ == '__main__':
    csv_average("1108d1")
    csv_average("1108d2")
