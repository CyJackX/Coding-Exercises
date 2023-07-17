if __name__ == '__main__':
    T = int(input().strip())

    for t_itr in range(T):
        N = int(input().strip())
        arr = []
        for n_itr in range(N):
            arr.append([int(x) for x in input().split(' ')])
        # print(arr)

        for row in range(N - 2, -1, -1):
            for col in range(row + 1):
                arr[row][col] += max(arr[row + 1][col],arr[row + 1][col + 1])
        
        print(arr[0][0])