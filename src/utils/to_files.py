import csv


def write_to_file(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in data:
            if isinstance(row[0], list):
                for subrow in row:
                    writer.writerow(subrow)
                continue
            writer.writerow(row)
