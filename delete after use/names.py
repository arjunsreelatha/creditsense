with open("names.csv", "r") as file:
    for line in file:
        row =  line.rstrip().split(",")
        print(f"{row[0]} is in {row[1]}")
