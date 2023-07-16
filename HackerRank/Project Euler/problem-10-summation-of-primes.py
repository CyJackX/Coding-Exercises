#!/bin/python3

import math
import os
import random
import re
import sys

def sieve(N):
    primes = [True for i in range(N+1)]
    p = 2
    while p**2 <= N:
        if primes[p]:
            for i in range(p**2, N+1, p):
                primes[i] = False
        p += 1
    prime_nums = [p for p in range(2, N + 1) if primes[p]]
    
    # print(prime_nums)
    return prime_nums

def limitedSegmentedSieve(N):
    limit = int(N**.5) + 1
    start = limit
    end = start * 2
    primes = sieve(start)
    length = len(primes)

    while start <= N:
        if end > N:
            end = N
        mark = [True for i in range(end - start+1)]
        for i in range(length):
            loLim = int((start/primes[i])) * primes[i]
            if loLim < start:
                loLim += primes[i]
            for j in range(loLim, end + 1, primes[i]):
                mark[j-start] = False

        for i in range(start, end + 1):
            if mark[i-start]:
                primes.append(i)

        start += limit
        end += limit
    return primes

#Looks like Sieve won't work as is, but modify it for summing, eh?

if __name__ == '__main__':
    t = int(input().strip())
    primes = sieve(10**6)
    result = [0] * (10**6+1)
    current_sum = 0
    for i in range(1, 10**6+1):
        if primes and i >= primes[0]: # Check if i is larger than the smallest prime left in the list
            current_sum += primes.pop(0) # If so, remove the smallest prime and add it to the current sum
        result[i] = current_sum

    for t_itr in range(t):
        n = int(input().strip())
        print(result[n])

