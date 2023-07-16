from itertools import groupby
from itertools import combinations_with_replacement
from itertools import permutations
from collections import Counter

def digitSquareSum(n):
    sum = 0
    while n > 0:
        sum += (n % 10)**2
        n = n//10
    return sum

def squareDigitChains(k):
    # Initialize a list of the maximum square digit sum
    list_ = Counter([1,4,9,16,25,36,49,64,81])

    memory = Counter([1,4,9,16,25,36,49,64,81])

    for i in range(2, k + 1):
        nextMem = Counter()
        for mem, count in memory.items():
            # start = mem % 10
            for digit in range(10):
                
                list_[mem + digit ** 2] += count
                nextMem[mem + digit ** 2] += count
        memory = nextMem

    # printSorted(list_)
    print89s(list_)

'''def bruteSquareDigitChains(k):
    arr = list(combinations_with_replacement([0,1,2,3,4,5,6,7,8,9],k))
    arr.pop(0)
    collect = Counter()
    for combo in arr:
        num = 0
        for dig in combo:
            num = num*10 + dig
        collect[digitSquareSum(num)] += 1
    
    print(collect)
    count = 0
    for i, value in collect.items():
        n = i
        while n != 89 and n != 1:
            n = digitSquareSum(n)
        if n == 89:
            count = (count + value) % (10 ** 9 + 7)
    print(count)'''

def bruteSquareDigitChains(k):
    collection = Counter()
    for i in range(1, 10**k):
        collection[digitSquareSum(i)] += 1
    # printSorted(collection)
    
    print89s(collection)
    
def print89s(collection):
    count = 0
    for i, value in collection.items():
        if(i == 0):
          continue
        n = i
        while n != 89 and n != 1:
            n = digitSquareSum(n)
        if n == 89:
            count = (count + value) % (10 ** 9 + 7)
    print(count)
def printSorted(collection):
    for key in sorted(collection):
      print(f'{key}: {collection[key]}')  
# for N in range(1,201):
#   print(N)
#   squareDigitChains(N)
#   bruteSquareDigitChains(N)

squareDigitChains((200))
