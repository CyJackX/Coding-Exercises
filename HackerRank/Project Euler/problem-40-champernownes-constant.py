# Enter your code here. Read input from STDIN. Print output to STDOUT
from functools import reduce
def d(n):
    i = 1
    lg = 0

    # Increase
    while n > 9 * 10 ** lg * (len(str(i))):
        n -= 9 * (10 ** lg) * len(str(i))
        i += 9 * 10 ** lg
        lg += 1

    digitsInI = len(str(i))

    # Calculate how many times we can subtract the current number of digits from n
    numbersToSkip = (n - 1) // digitsInI

    i += numbersToSkip
    n -= numbersToSkip * digitsInI
    return int(str(i)[n - 1])

inputs = []
for _ in range((int(input()))):
    inputs.append(input().split(' '))
    # print(row)
    # print(d(14))
for row in inputs:  
    print(reduce(lambda x, y: x * d(int(y)), row, 1))