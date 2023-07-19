from itertools import permutations
from itertools import combinations
from functools import reduce
import operator
'''
Essentially need all the ways to partition these groups of numbers.
'''
def partition_ordered(array):
    if not array:
        return [[]]

    first = array[0]
    remaining = array[1:]
    remaining_partitions = partition_ordered(remaining)

    new_partitions = []

    for rp in remaining_partitions:
        # 1. Add the current element as a new separate list
        new_partitions.append([[first]] + rp)

        # 2. Insert the current element at every possible position within each existing subset
        for i, subset in enumerate(rp):
            for j in range(len(subset) + 1):
                new_subset = subset[:j] + [first] + subset[j:]
                new_partitions.append(rp[:i] + [new_subset] + rp[i+1:])
                
        # 3. Add a new subset with the current element at every possible position between the existing subsets
        for i in range(1, len(rp) + 1):
            new_partitions.append(rp[:i] + [[first]] + rp[i:])

    return new_partitions

## Given an array, return a list of values using every operation between every element.
def operations(sub):

    #If the subarray is a singular element, return that
    if isinstance(sub,int):
        return [sub]
    
    #Else find all the possible values of the first element.
    leftValues = operations(sub[0])

    #Then, do all operations possible on all possible variations of the next element until the end.
    for i in range(1, len(sub)):
        rightValues = operations(sub[i])
        results = []
        for leftVal in leftValues:
            for rightVal in rightValues:
                results += [leftVal + rightVal, leftVal - rightVal, leftVal * rightVal]
                if rightVal != 0:
                    results += [leftVal / rightVal]
        leftValues = results

    return leftValues

#Rounding function for floating point errors
def round_if_close(num, tolerance=1e-9):
    nearest_int = round(num)
    if abs(num - nearest_int) < tolerance:
        return nearest_int
    return num

if __name__ == '__main__':
    # M = int(input().strip())
    # arr = [int(x) for x in input().split(' ')]
    combos = list(combinations([1,2,3,4,5,6,7,8,9],4))
    highScore = 0
    highScoreID = 0
    for arr_ in combos:
        arr = partition_ordered(arr_)
        vals = set()

        for part in arr:
            vals.update(operations(part))

        # Filter for natural numbers, rounding errors
        vals = {round(val) for val in vals if val > 0 and abs(val - round(val)) < 1e-9}

        missing = 1
        while(missing in vals):
            missing += 1
        
        if (missing - 1 > highScore):
            highScore = missing - 1
            highScoreID = ''.join([str(x) for x in arr_])
    
    print(highScoreID)
