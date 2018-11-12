import re
import csv
import csv_average as av
def time_write(name):
    csv_table = []
    with open(name + ".log", "r") as f:
        count = 0
        col = 0
        csv_line = []
        for line in f:
            if re.match("progress.+", line):
                col+=1
                if len(csv_line) > 0:
                    print(csv_line)
                    csv_table.append(csv_line)
                    csv_line = []

            m = re.split(",", line)
            if len(m) >= 2:
                print(m)
                if col == 1:
                    row.append(m[0])
                string = str(m[0])
                csv_line.append(float(m[1]))
                count += 1
        print(count)
    #for i in range(0, len(csv_table)):
    #    csv_table[i] = csv_table[i][2:]
    #    csv_table[i].pop(7)
    with open(name + ".csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(row)
        writer.writerows(csv_table)


# row = [u"1-1カメラから画像抽出", u"1-2認識エリアの切り抜き", u"画像抽出", \
#     u"認識：コピー",u"グレイ化",u"2値化",u"輪郭抽出",u"位置マーカ探索",u"付箋領域を認識"\
#     ,u"付箋位置の推定", u"id認識", u"後処理", u"all"]
#row = [u"画像抽出", \
#    u"認識：コピー",u"グレイ化",u"2値化",u"輪郭抽出",u"位置マーカ探索",u"付箋領域を認識"\
#    ,u"認識all",u"id認識", u"後処理"]
row = []
#name_list = ["keisoku0", "keisoku5", "keisoku10", "keisokump0", "keisokump5", "keisokump10", "ato10"]
#for name in name_list:
#    time_write(name)
name_list = ["hei1109d", "tyoku1109d"]
for name in name_list:
    row = []
    time_write(name)
    av.csv_average(name)
