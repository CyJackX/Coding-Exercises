using System;
using System.Linq;
using System.Diagnostics;

public class Solution2594
{

    public long RepairCars(int[] ranks, int cars)
    {
        long minTime = 0;
        int mechanics = ranks.Length;
        double[] squareroots = ranks.Select(x => 1 / Math.Sqrt(x)).ToArray();
        double sum = squareroots.Sum();
        List<int[]> mechanicsList = new List<int[]>();
        int carsRemaining = cars;

        //Distribute initial set of cars based on floor of proportions algebraically.
        for (int i = 0; i < mechanics; i++)
        {
            int carsToAssign = (int)(cars * squareroots[i] / sum);
            carsRemaining -= carsToAssign;
            int[] mech = new int[] { ranks[i], carsToAssign };
            mechanicsList.Add(mech);
            long time = RepairCalc(mech[0], mech[1]);
            if (time > minTime)
            {
                minTime = time;
            }
        }
        mechanicsList = mechanicsList.OrderBy(x => RepairCalc(x[0], x[1] + 1)).ToList();
        while (carsRemaining > 0)
        {
            int[] mech = mechanicsList[0];
            mech[1]++;
            carsRemaining--;
            long time = RepairCalc(mech[0], mech[1]);
            if (time > minTime)
            {
                minTime = time;
            }

            // Binary reinsert
            mechanicsList.RemoveAt(0);
            int index = mechanicsList.BinarySearch(mech, Comparer<int[]>.Create((x, y) => RepairCalc(x[0], x[1] + 1).CompareTo(RepairCalc(y[0], y[1] + 1))));
            if (index < 0)
            {
                index = ~index;
            }
            mechanicsList.Insert(index, mech);
        }
        return minTime;
    }

    public long RepairCarsOG(int[] ranks, int cars)
    {

        int mechanics = ranks.Length;
        Array.Sort(ranks);
        List<Mechanic> mechanicsList = new List<Mechanic>();

        // Initialize based on proportions, based on 
        int evenSplit = cars / mechanics;
        for (int i = 0; i < mechanics; i++)
        {
            mechanicsList.Add(new Mechanic(ranks[i], evenSplit));
            if (i < cars % mechanics)
            { // Remainder
                mechanicsList[i].UpdateCars(1);
            }
        }

        // Sort by repairTimes, sort min to max.
        mechanicsList.Sort(new MechanicTimeComparer());
        long minTime = mechanicsList[mechanics - 1].Time();

        //Also get list of all mechanics with less than minTime capacity
        List<Mechanic> freeMechanics = mechanicsList.Where(x => x.timePotential <= minTime).ToList();

        //Loop that minimizes minTime until it can't and updates lists as it goes.
        while (freeMechanics.Count > 0)
        {
            Mechanic slowest = mechanicsList[mechanics - 1];
            Mechanic fastest = mechanicsList[0];

            slowest.UpdateCars(-1);
            fastest.UpdateCars(1);
            mechanicsList.RemoveAt(mechanics - 1);
            mechanicsList.RemoveAt(0); // Binary Search for removal?            

            BinaryInsert(mechanicsList, slowest, new MechanicTimeComparer());
            BinaryInsert(mechanicsList, fastest, new MechanicTimeComparer());

            minTime = mechanicsList[mechanics - 1].Time();
            freeMechanics = freeMechanics.Where(x => x.timePotential <= minTime).ToList();
        }


        return minTime;
    }

    public static long Time(int[] rankAndCars)
    {
        return (long)(rankAndCars[0] * Math.Pow(rankAndCars[1], 2));
    }

    public static void BinaryInsert<T>(List<T> list, T item, IComparer<T> comparer = null)
    {
        if (comparer == null)
            comparer = Comparer<T>.Default;

        int binarySearchIndex = list.BinarySearch(item, comparer);

        if (binarySearchIndex < 0)
            binarySearchIndex = ~binarySearchIndex;

        list.Insert(binarySearchIndex, item);
    }

    public class MechanicTimeComparer : IComparer<Mechanic>
    {
        public int Compare(Mechanic x, Mechanic y)
        {
            if (x == null || y == null)
                throw new ArgumentException("Arguments can't be null!");

            if (x.Time() > y.Time()) return 1;
            if (x.Time() < y.Time()) return -1;

            return 0;
        }
    }

    public class MechanicPotentialComparer : IComparer<Mechanic>
    {
        public int Compare(Mechanic a, Mechanic b)
        {
            // As you wanted to compare with "b.Time() - a.Time()"
            return a.timePotential.CompareTo(b.timePotential);
        }
    }


    public static long RepairCalc(long rank, long cars)
    {
        return rank * cars * cars;
    }

    [DebuggerDisplay("ID: {ID}, Rank: {rank}, Cars: {carsAssigned}, Time: {time}, Delta: {delta}, TP: {timePotential}")]
    public class Mechanic
    {
        private static int IDGen = 0;
        public int rank;
        public int carsAssigned;
        private long time;
        public long timePotential;
        public int ID;
        public long delta;

        public Mechanic(int rank, int carsAssigned = 0)
        {
            this.rank = rank;
            this.carsAssigned = carsAssigned;
            ID = IDGen++;
            UpdateTime();
        }

        public void UpdateCars(int update)
        {
            carsAssigned += update;
            UpdateTime();
        }

        public void UpdateTime()
        {
            long cars = carsAssigned;
            time = rank * cars * cars;
            delta = rank * (2 * cars + 1);
            timePotential = time + delta;
        }

        public long Time()
        {
            return time;
        }

        public int Rank()
        {
            return rank;
        }

        public int Cars()
        {
            return carsAssigned;
        }

    }


}

public class Program2594
{
    public static void Main(string[] args)
    {
        int[] ranks = new int[] { 3, 1, 3, 1, 4, 7, 4, 6, 5, 5, 1, 2, 1, 2, 1, 7, 2, 6, 3, 7, 1, 1, 2, 4, 6, 2, 4, 5, 4, 6, 5, 7, 5, 7, 3, 1, 5, 6, 7, 5 };
        int cars = 2234;

        long result = (new Solution2594()).RepairCars(ranks, cars);
        long result2 = 0; //(new Solution2594()).RepairCarsOG(ranks, cars);

        Console.WriteLine($"{result}, vs {result2}");
    }
}