import csv

def P3Dfilter(file_path):
    rows = []
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if not (row[0].startswith('e_') or row[0].lower().startswith('error')):
                rows.append(','.join(row))
    
    formatted_text = ',\n'.join(rows)
    return formatted_text

csv_file = 'PIPE3D titles.csv'
formatted_output = P3Dfilter(csv_file)
print(formatted_output)
