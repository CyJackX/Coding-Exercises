'''
This one earns it's "advanced" difficulty.  It requires a lot of deep diving into algebra, diophantine equations,
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

Solubility of Ax^2 +Bxy + Cy^2 == N
https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=ab576e7bfc9c90800d306aa3b3685b34c8de7523
http://www.numdam.org/article/JTNB_2002__14_1_257_0.pdf

An Introduction to Diophantine Equations
https://www.isinj.com/mt-usamo/An%20Introduction%20to%20Diophantine%20Equations%20-%20A%20Problem-Based%20Approach%20-%20Andreescu,%20Andrica%20and%20Cucurezeanu%20(Birk,%202011).pdf

Pell's Equation, alternative solution(s) besides the continued fraction method using log, because continued fractions don't seem to work for random large PQ, and thus, D of the Pell's equation
https://www.ams.org/notices/200202/fea-lenstra.pdf

Bhaskara II's method is mentioned in a Pell's solver online: https://en.wikipedia.org/wiki/Chakravala_method
Though I don't know if you actually need it, depending on the actual testcases...

Seems like both continued fractions and Chakravala time out...
Alpertron's solutions seem like the ONLY way, his site is the only one that returns an instant answer!

--
Almost a week later, and I think I took a wrong turn with Pell's equation.
I thought it'd be easier since I already had solutions for Pell's, but ultimately it may be more straightforward to solve:
B*(B-1)/(T*(T-1)) = P/Q
=> QB^2 - QB -PT^2 +PT = 0
than to solve:
(QB-Q)^2 - QP(T-1)^2 = Q(Q-P) as an X^2 - DY^2 = N, because then you have to do divisibility checks, and that winds up being annoying.
Ultimately the question generalizes from Pell's equation to some Diophantine quadratics in general.
I've read so many PDFs on number theory that it's confusing to find out which one's have useful information.
So far the most useful site has been Dario Alpertron's site; it has a teaching function as well as a methodology page; the most important parts are at the bottom for general solutions!

And also, after building a testing suite testing across the EXTREME range that HackerRank provides, I think it's also overkill; there are solutions with 20000 digits if you were to test randomly for P and Q inclusive of 10**7!

It says all solutions are less than 2**63, but i had some testing that said otherwise?  So who knows.

This provides some help on solving quadratic modular arithmetic directly; I started with brute force; knowing that there are 2**(coprime factors) of solutions helps
https://math.stackexchange.com/questions/4467793/can-we-generalize-the-quadratic-formula-to-modular-arithmetic
https://math.gordon.edu/ntic/ntic/section-square-roots.html
http://www.numbertheory.org/php/squareroot.html

Tonelli-Shanks Algorithm for solving r**2 = n (mod p)
https://en.wikipedia.org/wiki/Tonelli%E2%80%93Shanks_algorithm

Straightforward overview of quadratic congruences:
https://www.johndcook.com/blog/quadratic_congruences

Alperton referenced this for his calculator
https://nntdm.net/papers/nntdm-25/NNTDM-25-1-075-083.pdf

on Quadratics mod 2^n
https://editorialdinosaurio.files.wordpress.com/2012/03/itn-niven.pdf
https://math.stackexchange.com/questions/90692/hensels-lemma-fx-equiv-0-pmodp-case/90856#90856
https://nntdm.net/papers/nntdm-25/NNTDM-25-1-075-083.pdf
'''
from itertools import chain, combinations
import cProfile
from functools import reduce
from itertools import product
from collections import Counter
import math
import random

# def is_perfect_square(n):
#     if n < 0:
#         return False
#     sqrt_n = int(n**0.5)
#     return sqrt_n * sqrt_n == n

# Floating point errors from other version I think?


def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a


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
        raise ValueError("No Modular Inverse")
    else:
        return x % M


def powerset(lst):
    # chain() joins together all the combinations
    # combinations(lst, r) generates all the possible combinations of length r
    return list(chain.from_iterable(combinations(lst, r) for r in range(len(lst)+1)))


class FactoredNumber:
    """Creating a helper class to factor once and only once when finding all square divisors, etc."""

    def __init__(self, factors):
        self.primeCounts = Counter(factors)

    def squareDivisors(self):
        square_roots = []
        for key in self.primeCounts.keys():
            for _ in range(self.primeCounts[key] // 2):
                square_roots.append(key)

        square_roots = list(set(powerset(square_roots)))
        square_roots = [FactoredNumber(square_root)
                        for square_root in square_roots]
        return square_roots

    def getValue(self):
        acc = 1
        for key, value in self.primeCounts.items():
            acc *= key**value
        return acc

    def clone(self):
        clone = FactoredNumber([0])
        clone.primeCounts = self.primeCounts.copy()
        return clone

    def divideBy(self, num2):
        """
        Return a new FactoredNumber with the factors of Num2 removed from Num1
        """
        result = self.clone()

        for key, value in num2.primeCounts.items():
            # Input Validation
            if key in result.primeCounts.keys() and result.primeCounts[key] >= value:
                result.primeCounts[key] -= value

        return result

    def primesPowered(self):
        """
        Return the prime factors to their power.
        """
        return [prime**power for prime, power in self.primeCounts.items()]

    def __repr__(self):
        return "Factors of " + str(self.getValue()) + " " + str(self.primeCounts)


def legendre(a, p):
    """
    Assume p is an odd prime
    """
    if gcd(a, p) != 1:
        return 0
    if a**((p-1)//2) % p == 1:
        return 1
    return -1


def jacobi(a, n):
    """
    Jacobi Symbol: https://en.wikipedia.org/wiki/Jacobi_symbol
    """
    pass


def tonelliShanks(n, prime):
    """
    The Tonelli-Shanks algorithm (referred to by Shanks as the RESSOL algorithm) is used in modular arithmetic to solve for r in a congruence of the form r**2 ≡ n (mod p), where p is a prime: that is, to find a square root of n modulo p.
    https://en.wikipedia.org/wiki/Tonelli%E2%80%93Shanks_algorithm

    Prime Power Congruences:
    https://brilliant.org/wiki/hensels-lemma/
    https://sites.millersville.edu/bikenaga/number-theory/prime-power-congruences/prime-power-congruences.html
    """

    # Regular Tonelli Shanks

    S = 0
    Q = prime - 1
    while Q % 2 == 0:
        Q //= 2
        S += 1

    z = 2
    while legendre(z, prime) == 1:
        z += 1

    M = S
    c = pow(z, Q, prime)
    t = pow(n, Q, prime)
    R = pow(n, (Q+1)//2, prime)

    while t != 1 and t != 0:
        i = 1
        while pow(t, 2**i, prime) != 1:
            i += 1

        b = pow(c, 2**(M-i-1), prime)
        M = i
        c = pow(b, 2, prime)
        t = (t*b**2) % prime
        R = R*b % prime

    return R, (-R) % prime


def modRootPowersOfTwo(n, power):
    if power == 1:
        return [n % 2]

    if power == 2:
        if n % 4 == 1:
            return [1, 3]
        if n % 4 == 0:
            return [0, 2]
        raise ValueError("No Roots")

    if power >= 3 and n == 1:
        return [1 % (2**power), -1 % (2**power), 2**(power-1) - 1, 2**(power-1) + 1]
    



def modSquareRoot(n, prime, power=1):
    """
    An algorithm combining Tonelli-Shanks for finding the squareroot x:
    x^2 = n mod (prime**power)

    Then using Hensel's Lemma to lift that solution to the correct power.
    """
    legendreSymbol = legendre(n % prime, prime)

    if legendreSymbol == -1:
        raise ValueError("No Roots")

    if prime == 2:
        # Most straightforward I"ve seen on prime = 2:
        # https://www.johndcook.com/blog/quadratic_congruences/

        # Brute force until Hensel's Stronger Lemma understood
        # print("bruting", n, prime, power)
        root = [x for x in range(prime**power) if x**2 % prime**power == n]
        if not root:
            raise ValueError("No Roots")
        return root
        # return modRootPowersOfTwo(n, power)

    # If prime is not of form 4k+3, then use Tonelli Shanks
    if prime % 4 == 3:
        root = pow(n, (prime+1)//4, prime)
    else:
        root = min(tonelliShanks(n, prime))

    return root, -root % prime**power


def testModRootFunction():
    while True:
        power = random.randint(1, 5)
        prime = random.choice([3, 5, 7, 11, 13, 17, 19, 23, 29])
        num = 0  # random.randint(0, prime**power)
        roots = modSquareRoot(num, prime, power)
        if roots:
            print(f"Checking solutions for {num} mod {prime} to the {power}")
            for root in roots:
                if (root**2 % prime**power != num):
                    raise ValueError("Root wrong!")
            print("success")
        else:
            print(
                f"Checking no solutions for {num} mod {prime} to the {power}")
            for root in range(prime**power):
                if (root**2 % prime**power == num):
                    raise ValueError("Root exists!")
            print("success")

pass

def quadraticCongruenceSolver(A, B, C, prime, power):
    """
    A limited quadratic congruence solver of Ax^2 + Bx + C = 0 mod prime**power.
    Can only take a prime and its exponent, no other composites.
    First solves the mod prime**1, then lifts with Hensel to appropriate power.
    """

    D = B*B - 4*A*C

    def f(x): #Function, helpful
        return A*x**2 + B*x + C
    def df(x):
        return 2*A*x + B    

    if prime == 2:
        return [x for x in range(prime**power) if f(x) % prime**power == 0]
    else:
        roots = modSquareRoot(D, prime, power)
        roots = [(-B + root)*modInverse(2*A,prime) % prime
                for root in roots]

    # Hensel's Lemma, implemented only for squares (It generalizes to polynomials)
    for root in roots:
        if df(x) % prime != 0:
            root = (root - (root**2 - n)*modInverse(2*root, prime)) % prime**power
            
    if power > 1 and 2*root % prime == 0:
        # Brute force until Hensel's Stronger Lemma understood
        # print("bruting", n, prime, power)
        root = [x for x in range(prime**power) if x**2 % prime**power == n]
        if not root:
            raise ValueError("No Roots")
        return root
        # raise ValueError("f'(r) == 0, Hensel's Stronger Lemma Needed?")

    for m in range(1, power):
        root = (root - (root**2 - n)*modInverse(2*root, prime)) % prime**power

    return roots
    # # Hensel's Lemma, implemented only for squares (It generalizes to polynomials)
    # if power > 1 and 2*root % prime == 0:
    #     # Brute force until Hensel's Stronger Lemma understood
    #     # print("bruting", n, prime, power)
    #     root = [x for x in range(prime**power) if x**2 % prime**power == n]
    #     if not root:
    #         raise ValueError("No Roots")
    #     return root
    #     # raise ValueError("f'(r) == 0, Hensel's Stronger Lemma Needed?")

    # for m in range(1, power):
    #     root = (root - (root**2 - n)*modInverse(2*root, prime)) % prime**power
# def quadraticCongruenceSolver(A, B, C, prime, power):
#     """
#     A limited quadratic congruence solver of Ax^2 + Bx + C = 0 mod prime**power.
#     Can only take a prime and its exponent, no other composites.
#     Also I think will assume B^2-4AC >= 0, so two solutions or none?
#     """
#     mod = prime**power
#     A %= mod
#     B %= mod
#     C %= mod
#     D = B*B - 4*A*C
#     D %= mod
#     def f(x): #Function, helpful
#         return A*x**2 + B*x + C
#     def df(x):
#         return 2*A*x + B
    
#     if prime == 2: #Brute until I can figure it out :-/
#         return [x for x in range(prime**power) if (A*x**2 +B*x + C) % prime**power == 0]

#     roots = modSquareRoot(D, prime, power)

#     roots = [(-B + root)*modInverse(2*A,mod) % mod
#              for root in roots]

#     return roots
def testQuadraticCongruences():
    while True:
        while True:
            A = random.randint(-500, 500)
            B = random.randint(-500, 500)
            C = random.randint(-500, 500)
            power = random.randint(1, 10)
            prime = 2 #random.choice([2,3, 5, 7, 11, 13, 17, 19, 23, 29])
            if gcd(A,prime) == 1: #Edge cases, resolved by unimodular transformation in general solver...
                break

        try:
            roots = quadraticCongruenceSolver(A, B, C, prime, power)
        except ValueError as e:            
            if str(e) == 'No Roots':
                print(f"Checking NO solutions for {A}x^2 + {B}x + {C} = 0 mod {prime}^{power}")
                for root in range(prime**power):
                    if ((A*root**2 + B*root + C) % prime**power == 0):
                        raise ValueError("Root exists!")
                print("No solutions found")
            else:
                raise
        else:
            print(f"Checking solutions for {A}x^2 + {B}x + {C} = 0 mod {prime}^{power}")
            for root in roots:
                if ((A*root**2 + B*root + C) % prime**power != 0):
                    raise ValueError("Root wrong!")
                print(root,"is a root")
            
            print("exhausting all other roots")
            for root in range(prime**power):
                    if ((A*root**2 + B*root + C) % prime**power == 0) and root not in roots:
                        raise ValueError("Another root exists!")
            print("   success")

            
testQuadraticCongruences()
exit()

def continued_fractions(D, remainder=0, denominator=1):
    """
    General quadratic continued fraction calculator, works for rationals as well
    """
    if is_perfect_square(D):
        remainder += round(D**.5)
        D = 0

    # Correct form for algorithm
    if (D - remainder**2) % denominator != 0:
        D *= denominator**2
        remainder *= abs(denominator)
        denominator *= abs(denominator)

    sequence_value = math.floor(((D**.5) + remainder)/denominator)

    continued_fraction_list = [sequence_value]

    seen = dict()  # Use a dictionary keys to store the indexes for faster lookups for a repeated state

    index = 1
    while True:
        # Continued Fraction Logic
        remainder = denominator * sequence_value - remainder
        if D - remainder**2 == 0:
            return [continued_fraction_list, []]
        denominator = (D - remainder**2) // denominator
        sequence_value = math.floor((remainder + D**.5) / denominator)

        # Repetition Check
        prevState = (sequence_value, remainder, denominator)
        if prevState in seen.keys():
            # Termination, return two arrays, the first with non-repeating, the second the repetend.
            repetendStart = seen[prevState]
            return [continued_fraction_list[0:repetendStart], continued_fraction_list[repetendStart:]]

        # Store values
        continued_fraction_list.append(sequence_value)
        seen[(sequence_value, remainder, denominator)] = index
        index += 1

# Returns convergents


def convergentFunc(continued_fractions):
    cur_num = continued_fractions[0][0]
    cur_denom = 1

    prev_num = 1
    prev_denom = 0

    period_length = len(continued_fractions[1])

    i = 1
    while True:
        yield (cur_num, cur_denom)
        if i < len(continued_fractions[0]):
            coefficient = continued_fractions[0][i]
        else:
            j = i - len(continued_fractions[0])
            coefficient = continued_fractions[1][(j) % period_length]

        cur_num, prev_num = cur_num * \
            coefficient + prev_num, cur_num
        cur_denom, prev_denom = cur_denom * \
            coefficient + prev_denom, cur_denom

        i += 1


def prime_factors(n):
    factors = []
    while n % 2 == 0:
        factors.append(2)
        n //= 2

    i = 3
    while i*i <= n:
        while n % i == 0:
            factors.append(i)
            n //= i
        i += 2

    if n > 2:
        factors.append(n)

    return factors


def chineseRemainder(integers, mods):
    """
    Array of integers, array of their mods
    """
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


def solutionsABC(A, B, C, N):
    """
    Delivers solutions to Ax^2 +Bxy + Cy^2 = N
    """

    D = B*B - 4*A*C

    solutions = []

    # Special class to minimize factoring; do it only once and pass around and manipulate those powers.
    factorsOfN = FactoredNumber(prime_factors(N))

    # For every square divisor of N...
    for square_root in factorsOfN.squareDivisors():
        # Get the value of the squareroot of the square
        squareIdx = square_root.getValue()
        # Divide N by the square, maintaining a factor count
        factors = factorsOfN.divideBy(square_root).divideBy(
            square_root)

        # Find all solutions to quadratic congruences of the prime**powers.
        factorSolutions = dict()
        try:
            for prime, power in factors.primeCounts.items():
                factorSolutions[prime**power] = quadraticCongruenceSolver(A,B,C,prime,power)
        except ValueError as e:
            # If there are no roots discovered, continue to the next square divisor of N
            if str(e) == 'No Roots':
                continue
            raise

        # Get all combinations of solutions for Chinese Remainder
        chineseIntegers = list(product(*factorSolutions.values()))
        chineseMods = list(factorSolutions.keys())
        modSolutions = []
        for integers in chineseIntegers:
            modSolutions.append(chineseRemainder(integers, chineseMods))

        modulus = N // squareIdx**2
        for T in modSolutions:
            for sign in [1, -1]:
                P = (A*T**2 + B*T + C)//modulus
                Q = -(2*A*T+B)
                R = A*modulus
                if gcd(gcd(P, Q), R) > 1:
                    continue

                cfs = continued_fractions(D, Q*sign, 2*R*sign)
                convergents = convergentFunc(cfs)

                # Only evaluate one period if continued fraction period is even, twice if odd.
                period = len(cfs[1])
                if period % 2 != 0:
                    period *= 2

                for _ in range(len(cfs[0]) + period):
                    [z, y] = next(convergents)

                    if P*y**2 + Q*y*-z + R*(-z)**2 == 1:
                        x = T*y - modulus*-z
                        x *= squareIdx
                        y *= squareIdx
                        solutions.extend([(x, y), (-x, -y)])

    solutions = list(set(solutions))
    return solutions

# Creates a recurrence function for a quadratic, yielding subsequent values if inserting previous values.


def recurrenceFunc(A, B, C, D, E, F):
    def recurrence(XYn):
        Xn = XYn[0]
        Yn = XYn[1]
        nextX = P*Xn + Q*Yn + K
        nextY = R*Xn + S*Yn + L
        return (nextX, nextY)

    # find r and s, where r^2 +Brs + ACs^2 == 1
    rsSolutions = solutionsABC(1, B, A*C, 1)
    # Only need the first positive solution?, because all others can be derived from it?
    rsSolutions = [x for x in rsSolutions if x[0] > 0 and x[1] > 0]
    rsSolutions.sort()
    r = rsSolutions[0][0]
    s = rsSolutions[0][1]

    denom = 4*A*C - B*B
    flipSigns = True
    # do while
    while True:
        P = r
        Q = -C*s
        R = A*s
        S = r + B*s
        Knum = C*D*(P+S - 2) + E*(B-B*r-2*A*C*s)
        Lnum = D*(B-B*r-2*A*C*s) + A*E*(P+S-2)

        # End condition
        if Knum % denom == 0 and Lnum % denom == 0:
            K = Knum // denom
            L = Lnum // denom + D*s
            yield recurrence

        # Try -r,-s first, then go to next (r,s)
        r = -r
        s = -s
        if flipSigns:
            flipSigns = False
        else:
            r, s = r**2 - A*C*s**2, 2*r*s + B*s**2
            flipSigns = True


# Ax^2 + Bxy + Cy^2 +Dx + Ex + F ==0
# Even if I don't use everything for now...


def genericQuadSolve(a, b, c, d, e, f):
    """
    Just the implementation to solve this:
    B(B-1)/(T(T-1)) = P/Q => QB^2 - QB -PT^2 + PT = 0
    """

    D = b*b - 4*a*c
    alpha = 2*c*d - b*e
    beta = 2*a*e - b*d

    # Post-legendre transformation, Dx = X + alpha, Dy = Y + beta, eliminates later terms
    #  AX^2 + CY^2 = -D*(a*e**2 - b*e*d + c*d**2 + f*D)

    # Make coefficient of X^2 and right hand side co-prime using unimodular transformation
    # X = mU +(m-1)V, Y = U + V
    RHS = -D*(a*e**2 - b*e*d + c*d**2 + f*D)
    m = 1
    while gcd(a*m**2 + c, RHS) != 1:
        m += 1

    # Convert to AU^2 + BUV + CV^2 = RHS
    A = a*m**2 + c
    B = 2*(a*m*(m-1) + c)
    C = a*(m-1)**2 + c

    UVsolutions = solutionsABC(A, B, C, RHS)
    xysolutions = []
    # U and V are the solutions of this equation, so convert back to X,Y, then x, y
    for solution in UVsolutions:
        U = solution[0]
        V = solution[1]
        X = m*U + (m-1)*V
        Y = U + V
        if (X + alpha) % D == 0 and (Y + beta) % D == 0:
            x = (X + alpha) // D
            y = (Y + alpha) // D
            xysolutions.append((x, y))

    return xysolutions

# Final algorithm using info gleaned from alperton's site for solving QB^2 - QB -PT^2 +PT = 0


def minDisk2(P, Q, min_val):
    g = gcd(P, Q)
    P //= g
    Q //= g
    if is_perfect_square(Q*P):
        return (None, None)

    xysolutions = genericQuadSolve(Q, 0, -P, -Q, P, 0)
    # xysolutions = [(1,1),(1,0),(0,1),(0,0)] # Heuristic, is it possible?
    recursiveIterator = recurrenceFunc(Q, 0, -P, -Q, P, 0)
    # for _ in range(2)] # Get both recursive functions?
    recursiveFunctions = [next(recursiveIterator)]
    B, T = 0, float('inf')
    for recursiveFunc in recursiveFunctions:
        for root in xysolutions:
            solution = root
            while solution[1] <= min_val:
                solution = recursiveFunc(solution)
                if solution[0] < 0 or solution[1] < 0:
                    break

            if solution[1] < T and solution[0] > 0 and solution[1] > 0:
                (B, T) = solution

    return (B, T)

# print(minDisk2(1,4,1000))
# solutionsABC(1, 14, 18, 1)
# genericQuadSolve(27, 0, -2, -27, 2, 0)
# recurrenceFunc(3,14,6,-17,-23,-505)


def minDisksBrute(P, Q, min_val, end = 100000):

    B = 0
    T = min_val
    if end == None:
        end = 1000000

    for T in range(min_val + 1, end + 1):
        B = round(((4*P*(T-1)*T + Q)**.5 + Q**.5)/(2*Q**.5))
        if B*(B-1)*Q == T*(T-1)*P:
            return [B, T]        
    
    return ('Over', 'flow')
    


def generate_random_inputs():
    PQmin = 10**5
    Dmin = 10**15
    P = random.randint(1, PQmin - 1)  # P is between 1 and 10**7 - 1
    Q = random.randint(P + 1, PQmin)  # Q is between P + 1 and 10**7
    D = random.randint(2, Dmin)    # D is between 2 and 10**15
    return P, Q, D


def test_minDisks():

    for Q in range(2, 10, 1):
        for P in range(1, Q, 1):
            min_val = 3
    # for _ in range(10):
            P, Q, D = generate_random_inputs()
            min_val = D

            
            [B, T] = minDisk2(P, Q, min_val)
            # [B2, T2] = minDisksBrute(P, Q, min_val, T)
            [B2, T2] = [B, T]

            # if T > 2**63:
            print(f"P={P}, Q={Q}, min_val={min_val}. Result: {B},{T}, {B2},{T2}")

            if B2 != 'Over' and [B, T] != [B2, T2]:
                print("Doesn't Match")
                return
            # if B == None:
            #     continue

            if B*(B-1)*Q == T*(T-1)*P and B2*(B2-1)*Q == T2*(T2-1)*P:
                pass
            else:
                print(
                    f"P={P}, Q={Q}, min_val={min_val}. Result: {B},{T}, {B2},{T2}")
                print("error")
                # return





# inputs = []
# for _ in range(int(input())):
#     inputs.append([int(x) for x in input().split(' ')])

# for input in inputs:
#     P = input[0]
#     Q = input[1]
#     min_val = input[2]

#     answer = minDisk2(P, Q, min_val)

#     if answer[0] != None:
#         print(str(answer[0]) + " " + str(answer[1]))
#     else:
#         print("No solution")

def next_solution(pell_solution, cur_solution):
    x1 = pell_solution[0]
    y1 = pell_solution[1]
    xk = cur_solution[0]
    yk = cur_solution[1]

    D = (x1 * x1 - 1) // (y1 * y1)

    return (x1 * xk + D * y1 * yk, x1 * yk + y1 * xk)


def solve_general_pells_brute(D, K, min=0):

    x = 0
    y = min
    square = 0
    while square == 0 or not is_perfect_square(square):
        y += 1
        square = K + D * y * y
    x = round(math.sqrt(K + D * y * y))

    return [(x, y), (-x, -y), [x, -y], [-x, y]]


def solve_pells(D):

    cfs = continued_fractions(D)
    convergentGenerator = convergentFunc(cfs)
    x = 0
    y = 0
    while x * x - D * y * y != 1:
        convergent = next(convergentGenerator)
        x = convergent[0]
        y = convergent[1]

    return (x, y)


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


def minDisks(P, Q, min_val):
    g = gcd(P, Q)
    P //= g
    Q //= g
    D = Q * P
    if is_perfect_square(D):
        return [None, None]

    K = Q * (Q - P)
    pell_solution = chakravala(D)

    # (x,y) and (x,-y) are both solutions which can generate positive values for future solutions
    # solution = solve_general_pells_brute(D, K)

    # # Turn them to blue disk and total disk count above min_val
    # solutions = [solution, [solution[0], -solution[1]], [-solution[0], -solution[1]], [-solution[0], solution[1]]]
    solutions = solutionsABC(1, 0, -D, K)
    solutions = [x for x in solutions if x[0] > 0]
    solutions.sort(key=lambda x: abs(x[0]))

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


# def alpertronGenericSolve(D, K):
#     # Now we must determine the values of s between 0 and F - 1 such that As2 + Bs + C _ 0 (mod F).
#     # Using the Chinese Remainder Theorem
#     # F = K
#     # From Alpertron site.  Hismethods page and calculator page use different variables, a little confusing!

#     # Have to find solutions for K but also every K / (squares)?
#     xySolutions = []

#     squareIdx = 1
#     while squareIdx**2 < K / 2 or squareIdx == 1:
#         if K % squareIdx**2 == 0:
#             Kprime = K // squareIdx**2

#             factors = primeFactors(Kprime)
#             factorSolutions = dict()
#             for factor in factors:
#                 factorSolutions[factor] = []
#                 for T in range(factor):
#                     if (T**2 - D) % factor == 0:
#                         factorSolutions[factor].append(T)

#             # Get all combinations of solutions
#             chineseIntegers = list(product(*factorSolutions.values()))
#             chineseMods = list(factorSolutions.keys())

#             # Get all solutions
#             modFSolutions = []
#             for integers in chineseIntegers:
#                 modFSolutions.append(chineseRemainder(integers, chineseMods))

#             # Some Transformations I don't quite understand
#             # But this is for solving P_Y2 + Q_Yk + R_ k2 = 1 (Different from P/Q in the main problem)
#             # P = (a_T2 + b_T + c) / n
#             # Q = _(2_a_T + b)
#             # R = a_n
#             # With our Pell-like equation being x^2 - Dy^2 - K = 0,
#             # so a = 1, b = 0, c = -D, n = K = F?  Variables all over the place on his site
#             C = -D
#             F = -Kprime

#             for T in modFSolutions:
#                 for plusMinus in [1, -1]:
#                     T *= plusMinus
#                     for sign in [1, -1]:
#                         P = -(T**2 + C)//F
#                         Q = 2*T
#                         R = -F
#                         if gcd(gcd(P, Q), R) > 1:
#                             continue

#                         cfs = continued_fractions(D, Q*sign, 2*Kprime*sign)
#                         convergents = convergentFunc(cfs)
#                         # z, y = convergents
#                         [z, y] = next(convergents)

#                         # Convergent Solution requires -k?
#                         for i in range(2, len(cfs[0]) + len(cfs[1])*2 + 1):

#                             if P*y**2 + Q*y*-z + R*(-z)**2 == 1:
#                                 # If a solution is found:
#                                 x = T*y - Kprime*z
#                                 x *= squareIdx
#                                 y *= squareIdx

#                                 xySolutions.extend([(x, y), (x, -y)])
#                                 break

#                             [z, y] = next(convergents)

#         squareIdx += 1

#     # duplicates
#     xySolutions = list(set([tuple(x) for x in xySolutions]))

#     return xySolutions

# Discovered YIELD, which can probably save a boatload of memory by doing convergences lone at a time, without needing to save states manually...
# All my convergents and continued fractions at once!
# (remainder + sqrt(D))/denominator
# def convergentFunc(D, remainder=0, denominator=1):

#     # To accommodate rational numbers, squares
#     if is_perfect_square(D):
#         remainder += round(D**.5)
#         D = 0

#     # Correct form for algorithm
#     if (D - remainder**2) % denominator != 0:
#         D *= denominator**2
#         remainder *= abs(denominator)
#         denominator *= abs(denominator)

#     continuedFraction = math.floor(((D**.5) + remainder)/denominator)

#     cur_num = continuedFraction
#     cur_denom = 1
#     prev_num = 1
#     prev_denom = 0

#     while True:
#         yield (cur_num, cur_denom)

#         remainder = denominator * continuedFraction - remainder
#         while D - remainder**2 == 0:
#             # Catches rational numbers
#             yield (cur_num, cur_denom)
#         denominator = (D - remainder**2) // denominator
#         continuedFraction = int((remainder + D**.5) / denominator)

#         cur_num, prev_num = cur_num * \
#             continuedFraction + prev_num, cur_num
#         cur_denom, prev_denom = cur_denom * \
#             continuedFraction + prev_denom, cur_denom

# def primePowers(n):
#     """
#     Factor, but grouping primes
#     """
#     factors = []

#     factor = 1
#     while n % 2 == 0:
#         n //= 2
#         factor *= 2
#     factors.append(factor)

#     i = 3
#     while i*i <= n:
#         factor = 1
#         if n % i == 0:
#             while n % i == 0:
#                 n //= i
#                 factor *= i
#             factors.append(factor)
#         i += 2

#     if n > 1:
#         factors.append(n)

#     return factors

def testCase(P, Q, min_val):
    # D = P*Q
    # K = Q*(Q-P)
    # print(solve_pells(D))
    # print(chakravala(D))
    [B, T] = minDisk2(P, Q, min_val)
    # [B2, T2] = [None, None]
    [B2, T2] = minDisks(P, Q, min_val)

    print(f"P={P}, Q={Q}, min_val={min_val}. Result: {B},{T}, {B2},{T2}")


profiler = cProfile.Profile()
profiler.enable()
# testCase(1,5, 2)
test_minDisks()
# func = recurrenceFunc(17,0,-5,-17,5,0)
# for _ in range(1000):
# next(func)
#     D = random.randint(2, 1000000)
#     if is_perfect_square(D):
#         continue
#     chakravala(D)
#     solve_pells(D)
#     solutionsABC(1,0,-D,1)
profiler.disable()
profiler.print_stats(sort="tottime")
# testCase(56,59,2)