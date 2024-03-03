import csv


def replace_commas_with_semicolons(input_file, output_file):
    # Open the input CSV file in read mode
    with open(input_file, 'r', newline='') as csv_in:
        # Open the output CSV file in write mode
        with open(output_file, 'w', newline='') as csv_out:
            # Create CSV reader and writer objects
            reader = csv.reader(csv_in)
            writer = csv.writer(csv_out, delimiter=';')

            # Iterate over each row in the input CSV file
            for row in reader:
                # Replace commas with semicolons in each row
                modified_row = [cell.replace(',', ';') for cell in row]
                # Write the modified row to the output CSV file
                writer.writerow(modified_row)


# Replace commas with semicolons in input.csv and write the result to output.csv
#replace_commas_with_semicolons('B-data-mnist_test.csv', 'newB-data-mnist_test.csv')
#replace_commas_with_semicolons('F-data-mnist_test.csv', 'newF-data-mnist_test.csv')

import csv


def replace_commas_with_semicolons_and_convert_to_int(input_file, output_file):
    # Open the input CSV file in read mode
    with open(input_file, 'r', newline='') as csv_in:
        # Open the output CSV file in write mode
        with open(output_file, 'w', newline='') as csv_out:
            # Create CSV reader and writer objects
            reader = csv.reader(csv_in)
            writer = csv.writer(csv_out, delimiter=';')

            # Iterate over each row in the input CSV file
            for row in reader:
                # Check if all cells in the row contain only numbers
                if all(cell.replace('.', '', 1).isdigit() for cell in row):
                    # Convert appropriate numbers with decimal part .0 to integers
                    modified_row = [int(cell.split('.')[0]) if '.' in cell and cell.split('.')[1] == '0' else cell for
                                    cell in row]
                else:
                    # If row contains string or mixed data, leave it unchanged
                    modified_row = row
                # Write the modified row to the output CSV file
                writer.writerow(modified_row)


# Replace commas with semicolons and convert appropriate numbers to integers in input.csv and write the result to output.csv
#replace_commas_with_semicolons_and_convert_to_int('F-data-mnist_test.csv', 'newF-data-mnist_test.csv')

replace_commas_with_semicolons_and_convert_to_int('B-data-mnist_test.csv', 'newB-data-mnist_test.csv')
