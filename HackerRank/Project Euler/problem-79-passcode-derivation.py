from itertools import permutations
def checkValidity(password, log):
    return password.index(log[1]) > password.index(log[0]) and password.index(log[2]) > password.index(log[1])

if __name__ == '__main__':
    # T = int(input().strip())
    # logs = set()
    # for t_itr in range(T):
    #     logs.add(input().strip())
    logs = set(['SMH', 'TON', 'RNG', 'WRO', 'THG'
                ])
    allChars = set(ch for s in logs for ch in s)
    allPasswords = list(permutations(allChars))
    for log in logs:
        allPasswords = [pword for pword in allPasswords if checkValidity(pword,log)]
    
    if allPasswords:
        allPasswords.sort()
        print(''.join(allPasswords[0]))
    else:
        print('SMTH WRONG')
    
