if __name__ == '__main__':
    t = int(input().strip())

    for t_itr in range(t):
        N = int(input())
        sum = 0
        num = 2**N
        while num > 0:
            sum += num % 10
            num //= 10
        print(sum)
        