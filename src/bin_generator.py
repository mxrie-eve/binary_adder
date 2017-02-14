from sys import argv
import random as r

# Create an input and output file
input_file = open('../data/input.txt','w')
output_file = open('../data/output.txt','w')

# Convert a decimal to a binary string
def dec2bin (x):
 return str(bin(x)[2:])

# Transform a binary string to a 32 char long string
def bin2biggerbin(x):
 return ('0' * (NB_BIT - len(x)) ) + x

m = [0,0]

NB_BIT = 16
for i in range(100000):
  for j in range(2):
    # Generate a 31 char long binary in the decimal format
    number = r.getrandbits(NB_BIT -1)
    tmp = dec2bin(number)
    tmp=bin2biggerbin(tmp)
    m[j] = tmp
    input_file.write('%s' % tmp)
  # Add the 2 random binaries and write the result
  result = dec2bin(int(m[0],2)+int(m[1],2))
  output_file.write(bin2biggerbin(result)+'\n')
  input_file.write('\n')

# Close the files
input_file.close()
output_file.close()

