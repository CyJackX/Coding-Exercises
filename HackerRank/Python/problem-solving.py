from itertools import combinations

def compareHotelDistances(combos, roads):
    ways = 0
    for i in range(len(combos)):
        combo = combos[i]
        a = distanceBetween(combo[0], combo[1], roads)
        b = distanceBetween(combo[1], combo[2], roads)
        c = distanceBetween(combo[0], combo[2], roads)
        if a == b and b == c and a == c:
            ways += 1
    return ways
        
def numberOfWays(roads):
    numCities = len(roads) + 1
    cities = list(range(1,numCities + 1))
    combos = list(combinations(cities,3))
    return compareHotelDistances(combos,roads)

def distanceBetween(a, b, roads, distanceTraveled = 0, pastPath = 0):
    if a == b:
        return distanceTraveled
    
    sum = 0
    for road in roads:
        if road[0] == a and road[1] != pastPath:
            result = distanceBetween(road[1], b, roads, distanceTraveled +1, a)
            if result:
                return result
        if road[1] == a and road[0] != pastPath:
            result = distanceBetween(road[0],b, roads, distanceTraveled +1, a)
            if result:
                return result
        
print(numberOfWays([[1, 2], [2, 5], [3, 4], [4, 5], [5, 6], [7, 6]]))

