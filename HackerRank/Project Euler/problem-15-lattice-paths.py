def gcd(x, y):
    while y:
        x, y = y, x % y
    return x

        
def combinations(n, r):
    answer = 1
    d = n - r
    denom = 1
    for i in range(r+1, n+1):
        answer *= i
        denom *= d
        g = gcd(answer,denom)
        if(g > 1):
            answer //= g
            denom //= g
        d -= 1
    return answer

if __name__ == '__main__':
    t = int(input().strip())

    for t_itr in range(t):
        NM = input().split(' ')
        N = int(NM[0])
        M = int(NM[1])
        print(combinations(M+N,N) % (10**9 + 7))
        
