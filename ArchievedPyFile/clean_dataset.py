import os

input_path  = './adc.data'
tmp_path    = input_path + '.tmp'

lower_bound = 0
upper_bound = 64000
count = 0

valid_data = []

with open(input_path, 'r') as fin, open(tmp_path, 'w') as fout:
    for raw in fin:
        line = raw.strip()

        # Check if the line is a valid positive integer
        if line.isdigit():
            if 500 < int(line) < 3850 and (lower_bound < count < upper_bound):
                # Write the valid line to the temporary file
                valid_data.append(int(line))
                fout.write(line + '\n')
        count += 1


# for i in range(len(valid_data)):
#     if i == 0:
#         window = [valid_data[i], valid_data[i], valid_data[i+1]]
#     elif i == len(valid_data) - 1:
#         window = [valid_data[i-1], valid_data[i], valid_data[i]]
#     else:
#         window = [valid_data[i-1], valid_data[i], valid_data[i+1]]

#     median_value = sorted(window)[1]  # After sorting, the middle one is median
#     valid_data[i] = median_value  # Replace with median value
    
# Step 3: Save filtered result back into file
with open(tmp_path, 'w') as fout:
    for value in valid_data:
        fout.write(f"{value}\n")
        
# Replace original file with the cleaned one
os.replace(tmp_path, input_path)

print("Cleaning complete. Only valid positive integers remain in adc.data.")
