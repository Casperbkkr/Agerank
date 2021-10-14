import csv

def huidhoudens_opbouw(csv_file):

    return


file = open("Huishoudens__samenstelling__grootte__regio__1_januari_11102021_105824.csv")
csvreader = csv.reader(file)
header = next(csvreader)
print(header)
rows = []
for row in csvreader:
    rows.append(row)
print(rows)
file.close()


