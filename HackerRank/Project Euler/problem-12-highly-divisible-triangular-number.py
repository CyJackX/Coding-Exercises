# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import Counter
def divisible_triangle_number(n):
    triangle = 1
    count = 2
    while count_divisors(triangle) <= n:
        triangle += count
        count += 1
    print(triangle)


def count_divisors(n):
    primeFactors = Counter()

    while n % 2 == 0:
        primeFactors[2] += 1
        n = n // 2
    
    i = 3
    while i * i <= n:
        while n % i== 0:
            primeFactors[i] += 1
            n = n // i
        i += 2
        
    if n > 1:
        primeFactors[n]+=1
    prod = 1
    for value in primeFactors.values():
        prod *= value + 1
    return prod
divisible_triangle_number(4)

if __name__ == '__main__':
    t = int(input().strip())

    for t_itr in range(t):
        n = int(input().strip())
        divisible_triangle_number(n)
