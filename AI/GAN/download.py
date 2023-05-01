import requests

data_dir = "dataset/"

ranges = []

for i in range(0, 160):
    if i < 10:
        ranges.append("000-00" + str(i))

    elif i < 100:
        ranges.append("000-0" + str(i))

    else:
        ranges.append("000-" + str(i))

print(ranges)