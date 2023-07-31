'''
This one earns it's "advanced" difficulty.  It requires a lot of deep diving into algebra, Pell's Equations, Generalized Pell's Equations,
        information which is NOT easy to suss out on the internet even using ChatGPT; you have to do some research yourself!  But ChatGPT definitely helped understand the parts I could not.
        
        B = Blue Disks
        T = Totals
        P = Double Blue Outcome
        Q = Total Outcomes
        
        Fundamentally, we are asked to find T above a minimum D.  To do this without brute force, however, seems like it will require finding ALL solutions and iterating the next solution(s).  This involves algebra and more Diophantine Equations.
        This:
        B*(B-1)/(T*(T-1)) = P/Q
        ...can be converted into general Diophantine Equations via this link:
        https://math.stackexchange.com/questions/2186773/reduction-of-quadratic-diophantine-equation-to-pells-equation
        ...where it was shown that Legendre showed you could turn any Quadratic Diophantine into two Pell-type equations by completing the squares.  That math is beyond me, but you need a little algebra still to be able to manipulate it.
        ax^2+bxy+cy^2+dx+ey+f=0
        (Dy2ae+bd)^2D(2ax+by+d)^2=4a(ae2+cd2bde+Df)
        (Dx2cd+be)^2D(2cy+bx+e)^2=4c(ae2+cd2bde+Df)
        D = b^2 -4ac

        Do the algebra, create some Diophantine solvers and solution iterators, and you'll be on your way.
        Very helpful site(s)
        https://www.alpertron.com.ar/QUAD.HTM
        This site actually has a step-by-step calculator, if you can understand what it's doing!

        https://www.alpertron.com.ar/METHODS.HTM
        https://kconrad.math.uconn.edu/math3240s20/handouts/pelleqn1.pdf
        https://www.imsc.res.in/~knr/acadimsc13/mds.pdf
        http://www.numbertheory.org/PDFS/general_quadratic_solution.pdf
        http://www.numbertheory.org/PDFS/talk_2004.pdf

        An Introduction to Diophantine Equations
        https://www.isinj.com/mt-usamo/An%20Introduction%20to%20Diophantine%20Equations%20-%20A%20Problem-Based%20Approach%20-%20Andreescu,%20Andrica%20and%20Cucurezeanu%20(Birk,%202011).pdf

        Pell's Equation, alternative solution(s) besides the continued fraction method using log, because continued fractions don't seem to work for random large PQ, and thus, D of the Pell's equation
        https://www.ams.org/notices/200202/fea-lenstra.pdf

        Bhaskara II's method is mentioned in a Pell's solver online: https://en.wikipedia.org/wiki/Chakravala_method
        Though I don't know if you actually need it, depending on the actual testcases...

        Seems like both continued fractions and Chakravala time out...
        Alpertron's solutions seem like the ONLY way, his site is the only one that returns an instant answer!

'''
from functools import reduce
from itertools import product
import math
import random


def minDisks(P, Q, min_val):
    g = gcd(P,Q)
    P //= g
    Q //= g
    D = Q * P
    if is_perfect_square(D):
        return [0,0]

    K = Q * (Q - P)
    pell_solution = chakravala(D)

    # (x,y) and (x,-y) are both solutions which can generate positive values for future solutions
    # solution = solve_general_pells_brute(D, K)

    # # Turn them to blue disk and total disk count above min_val
    # solutions = [solution, [solution[0], -solution[1]], [-solution[0], -solution[1]], [-solution[0], solution[1]]]
    solutions = alpertronGenericSolve(D,K)    
    solutions = [x for x in solutions if x[0] > 0]
    solutions.sort(key = lambda x: abs(x[0]))

    answer = [0, float('inf')]

    for xy in solutions:
        T = 0
        B = 0
        solution = xy

        # If both even, can't work
        if xy[0] % 2 == 0 and xy[1] % 2 == 0:
            continue

        while abs(T) <= min_val and abs((solution[1] + 1) // 2) < 2**200:
            if (solution[0] + Q) % (2 * Q) == 0 and (solution[1] + 1) % 2 == 0:
                B = (solution[0] + Q) // (2 * Q)
                T = (solution[1] + 1) // 2

            solution = next_solution(pell_solution, solution)

        # Compare and pick the smaller one
        if T < answer[1] and B > 0 and T > 0 and B < T:
            answer = [B, T]

    return answer


# def is_perfect_square(n):
#     if n < 0:
#         return False
#     sqrt_n = int(n**0.5)
#     return sqrt_n * sqrt_n == n

# Floating point errors from other version I think?
def is_perfect_square(n):
    if n < 0:
        return False
    low, high = 0, n
    while low <= high:
        mid = (low + high) // 2
        mid_squared = mid * mid
        if mid_squared == n:
            return True
        elif mid_squared < n:
            low = mid + 1
        else:
            high = mid - 1
    return False

def next_solution(pell_solution, cur_solution):
    x1 = pell_solution[0]
    y1 = pell_solution[1]
    xk = cur_solution[0]
    yk = cur_solution[1]

    D = (x1 * x1 - 1) // (y1 * y1)

    return (x1 * xk + D * y1 * yk, x1 * yk + y1 * xk)


# def solve_general_pells(D, K):
#     pell_solution = solve_pells(D)
#     n = 1
#     x = pell_solution[0]
#     y = pell_solution[1]

#     x_power = x
#     y_power = y

#     while (x_power - D * y_power) % K != 0:
#         n += 1
#         x_power *= x
#         y_power *= y

#     multiplier = (x_power - D * y_power) // K

#     return (x * multiplier, y * multiplier)


def solve_general_pells_brute(D, K, min=0):

    x = 0
    y = min
    square = 0
    while True:
        y += 1
        square = K + D * y * y
        if square != 0 and is_perfect_square(square):
            break
    x = round(math.sqrt(K + D * y * y))

    return [(x, y),(-x,-y),[x,-y],[-x,y]]

def solve_pells(D):

    fractions = continued_fractions(D)
    i = 1
    x = 0
    y = 0
    while True:
        convergent = nth_convergent(i, fractions)
        x = convergent[0]
        y = convergent[1]
        i += 1
        if x * x - D * y * y == 1:
            break

    return (x, y)

# General quadratic continued fraction calculator, works for rationals as well

def continued_fractions(D, remainder=0, denominator=1):
    if is_perfect_square(D):
        remainder += round(D**.5)
        D = 0

    # Correct form
    if (D - remainder**2) % denominator != 0:
        D *= denominator**2
        remainder *= abs(denominator)
        denominator *= abs(denominator)

    integer_part = int(((D**.5) + remainder)/denominator)

    continued_fraction_list = [integer_part]

    sequence_value = integer_part

    seen = []

    while True:
        remainder = denominator * sequence_value - remainder
        if D - remainder**2 == 0:
            return [continued_fraction_list, []]
        denominator = (D - remainder**2) // denominator
        sequence_value = int((remainder + D**.5) / denominator)

        prevState = (sequence_value, remainder, denominator)
        if prevState in seen:
            repetendStart = 1 + list(seen).index(prevState)
            return [continued_fraction_list[0:repetendStart], continued_fraction_list[repetendStart:]]

        continued_fraction_list.append(sequence_value)
        seen.append((sequence_value, remainder, denominator))

def nth_convergent(n, continued_fractions):
    cur_num = continued_fractions[0][0]
    cur_denom = 1

    prev_num = 1
    prev_denom = 0

    period_length = len(continued_fractions[1])

    for i in range(1, n):

        if i < len(continued_fractions[0]):
            coefficient = continued_fractions[0][i]
        else:
            j = i - len(continued_fractions[0])
            coefficient = continued_fractions[1][(j) % period_length]

        cur_num, prev_num = cur_num * \
            coefficient + prev_num, cur_num
        cur_denom, prev_denom = cur_denom * \
            coefficient + prev_denom, cur_denom

    return [cur_num, cur_denom]

def next_convergent(prevConvergent, currentConvergent, continued_fractions, nextIndex):
    cur_num = currentConvergent[0]
    cur_denom = currentConvergent[1]

    prev_num = prevConvergent[0]
    prev_denom = prevConvergent[1]

    period_length = len(continued_fractions[1])

    i = nextIndex - 1

    if i < len(continued_fractions[0]):
        coefficient = continued_fractions[0][i]
    else:
        j = i - len(continued_fractions[0])
        coefficient = continued_fractions[1][(j) % period_length]

    cur_num, prev_num = cur_num * \
        coefficient + prev_num, cur_num
    cur_denom, prev_denom = cur_denom * \
        coefficient + prev_denom, cur_denom

    return [cur_num, cur_denom]

def gcd(a,b):
    while b != 0:
        a, b = b, a % b
    return a

def gcdExtended(a, b):
    x0, x1, y0, y1 = 1, 0, 0, 1

    while b != 0:
        q, a, b = a // b, b, a % b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1

    return a, x0, y0

def modInverse(A, M):

    g, x, y = gcdExtended(A, M)
    if (g != 1):
        print("Inverse doesn't exist")

    else:
        # m is added to handle negative x
        return (x % M + M) % M

# print(modInverse(50,71))


def chakravala(D):
    b = 1
    a = int(D**.5) + 1
    k = a*a - D*b*b

    while k != 1:
        # Determine m
        # m = int(abs(k) * (D**.5) // abs(k))
        # while (a + b*m) % abs(k) != 0:
        #     m -= 1
        # if abs((m+abs(k))**2 - D) < abs(m*m - D):
        #     m = m + abs(k)

        # Took a while to grok it but got the modular inverse and how to use it to my advantage instead of iterating towards m
        # since a + bm = 0 mod k, that means m = -a*b^-1 = m - j *k, where j is some integer
        # And if m is close to the square root of D as possible, it gives us a place to solve for the nearest j.
        binverse = modInverse(b, abs(k))
        j = (int(D**.5)+a*binverse)//k
        m2 = j*k-a*binverse

        m2 = min([m2, m2 + k, m2 - k], key=lambda x: abs(x**2 - D))
        if m2 < 0:
            m2 += k
        # print(m, m2, k)
        m = m2

        a, b, k = (a*m + D*b)//abs(k), (a+b*m)//abs(k), (m*m - D)//k

    return [a, b]

def minDisksBrute(P, Q, min_val):
    D = Q * P
    if is_perfect_square(D):
        return None
    B = 0
    T = min_val
    flip = True
    while B*(B-1)*Q != T*(T-1)*P:
        if flip:
            T += 1
            B = int(((4*P*(T-1)*T + Q)**.5 + Q**.5)/(2*Q**.5))
        else:
            B += 1
        flip = not flip
    return [B, T]


def primeFactors(n):
    i = 2
    factors = []
    while i <= n**.5:
        primeFactors = 1
        while n % i == 0:
            primeFactors *= i
            n //= i
        if primeFactors > 1:
            factors.append(primeFactors)
        i += 1
    if n > 1:
        factors.append(n)
    return factors


def chineseRemainder(integers, mods):
    M = reduce(lambda x, y: x * y, mods, 1)
    modGroup = [M//mod for mod in mods]

    inverses = []
    for i in range(len(mods)):
        inverses.append(modInverse(modGroup[i], mods[i]))

    answer = 0
    for i in range(len(mods)):
        answer += integers[i] * inverses[i] * modGroup[i]

    answer %= M
    return answer


'''
Finding solutions of the homogeneous equation Ax2 + Bxy + Cy2 + F = 0
x = sy - Fz 
-(As2 + Bs + C) y2 / F + (2As + B)yz - AFz2 = 1 
'''

def alpertronGenericSolve(D, K):
    # Now we must determine the values of s between 0 and F - 1 such that As2 + Bs + C _ 0 (mod F).
    # Using the Chinese Remainder Theorem
    # F = K
    # From Alpertron site.  Hismethods page and calculator page use different variables, a little confusing!

    # Have to find solutions for K but also every K / (squares)?
    xySolutions = []

    squareIdx = 1
    while squareIdx**2 < K / 2 or squareIdx == 1:
        if K % squareIdx**2 == 0:
            Kprime = K // squareIdx**2

            factors = primeFactors(Kprime)
            factorSolutions = dict()
            for factor in factors:
                factorSolutions[factor] = []
                for T in range(factor):
                    if (T**2 - D) % factor == 0:
                        factorSolutions[factor].append(T)

            # Get all combinations of solutions
            chineseIntegers = list(product(*factorSolutions.values()))
            chineseMods = list(factorSolutions.keys())

            # Get all solutions
            modFSolutions = []
            for integers in chineseIntegers:
                modFSolutions.append(chineseRemainder(integers, chineseMods))

            # Some Transformations I don't quite understand
            # But this is for solving P_Y2 + Q_Yk + R_ k2 = 1 (Different from P/Q in the main problem)
            # P = (a_T2 + b_T + c) / n
            # Q = _(2_a_T + b)
            # R = a_n
            # With our Pell-like equation being x^2 - Dy^2 - K = 0,
            # so a = 1, b = 0, c = -D, n = K = F?  Variables all over the place on his site
            C = -D
            F = -Kprime
            
            for T in modFSolutions:
                for plusMinus in [1, -1]:
                    T *= plusMinus
                    for sign in [1, -1]:
                        P = -(T**2 + C)//F
                        Q = 2*T
                        R = -F
                        if gcd(gcd(P,Q),R) > 1:
                            continue

                        cf = continued_fractions(D, Q//2*sign, Kprime*sign)
                        # z, y = convergents
                        [z, y] = nth_convergent(1, cf)
                        prevConvergent = [1,0]

                        # Convergent Solution requires -k?
                        for i in range(2, len(cf) + len(cf[1])*2 + 1):        
                            
                            if P*y**2 + Q*y*-z + R*(-z)**2 == 1:
                                # If a solution is found:
                                x = T*y - Kprime*z
                                x *= squareIdx
                                y *= squareIdx

                                xySolutions.append((x,y))
                                break

                            [z, y], prevConvergent = next_convergent(prevConvergent, [z,y], cf, i), [z,y]
        
        squareIdx += 1
        
    
    # duplicates
    xySolutions = list(set([tuple(x) for x in xySolutions]))

    return xySolutions

def solutions

def generate_random_inputs():
    PQmin = 10**7
    Dmin = 10**15
    P = random.randint(1, PQmin - 1)  # P is between 1 and 10**7 - 1
    Q = random.randint(P + 1, PQmin)  # Q is between P + 1 and 10**7
    D = random.randint(2, Dmin)    # D is between 2 and 10**15
    return P, Q, D


def test_minDisks():
    

    for Q in range(2, 200, 1):
        for P in range(1, Q, 1):
    
    # test_cases = 100  # Change this value based on how many tests you want to run
    # for _ in range(test_cases):
            P, Q, D = generate_random_inputs()
            if is_perfect_square(Q*P):
                continue
            # P = 30
            # Q = 40
            min_val = D  # or any other value you wish to set based on the context
            [B, T] = minDisks(P, Q, min_val)
            # [B2, T2] = minDisksBrute(P, Q, min_val)
            [B2, T2] = [B, T]
            
            
            # if T > 2**63:
            print(f"P={P}, Q={Q}, min_val={min_val}. Result: {B},{T}, {B2},{T2}")

            if B*(B-1)*Q == T*(T-1)*P and B2*(B2-1)*Q == T2*(T2-1)*P:
                pass
            else:
                print(f"P={P}, Q={Q}, min_val={min_val}. Result: {B},{T}, {B2},{T2}")
                print("error")
                # return

            if [B, T] != [B2, T2]:
                print("Not Minimum")
                return

            # else:
            #     print("FAIL")
            #     print(f"P={P}, Q={Q}, min_val={min_val}. Result: {B},{T}")
            #     break
import cProfile

def testCase(P,Q,min_val):
    D = P*Q
    K = Q*(Q-P)
    print(solve_pells(D))
    print(chakravala(D))
    [B, T] = minDisks(P, Q, min_val)
    [B2, T2] = [None, None]
    # [B2, T2] = minDisksBrute(P, Q, min_val)

    print(f"P={P}, Q={Q}, min_val={min_val}. Result: {B},{T}, {B2},{T2}")

profiler = cProfile.Profile()
profiler.enable()
testCase(2,7,2)
# test_minDisks()
profiler.disable()
profiler.print_stats(sort="cumulative")
# testCase(56,59,2)

# import math
# i = 1
# while is_perfect_square(i*i):
#     print(math.log10(i*i))
#     i *= 9

# print(i)

# inputs = []
# for _ in range(int(input())):
#     inputs.append([int(x) for x in input().split(' ')])

# for input in inputs:
#     P = input[0]
#     Q = input[1]
#     min_val = input[2]

#     answer = minDisks(P, Q, min_val)

#     if answer[0] > 0:
#         print(str(answer[0]) + " " + str(answer[1]))
#     else:
#         print("No solution")
