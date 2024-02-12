import sys
import csv

# Check if the correct number of arguments are provided
if len(sys.argv) != 3:
    print("Usage: python script.py <number1> <number2>")
    sys.exit(1)

# Try to convert arguments to numbers
try:
    number1 = float(sys.argv[1])
    number2 = float(sys.argv[2])
except ValueError:
    print("Please provide two numbers.")
    sys.exit(1)

# Perform a calculation (in this case, addition)
result = number1 + number2

# Define the CSV file name
filename = "results.csv"

# Write the result to a CSV file
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Writing the headers
    writer.writerow(['Number 1', 'Number 2', 'Result'])
    # Writing the data
    writer.writerow([number1, number2, result])

print(f"Result of adding {number1} and {number2} is {result}. Saved to {filename}.")
