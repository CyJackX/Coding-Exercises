using System;
using System.Collections.Generic;
using System.IO;
class Solution
{
    static void Main(String[] args)
    {
        /* Enter your code here. Read input from STDIN. Print output to STDOUT. Your class should be named Solution */

        int cases = Convert.ToInt32(Console.ReadLine());
        long[][] inputs = new long[cases][];  // Declare inputs as a jagged array of long

        for (int i = 0; i < cases; i++)
        {
            inputs[i] = Console.ReadLine().Split(' ').Select(long.Parse).ToArray();
            // Console.WriteLine(i);
        }

        /*
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

        Pell's Equation
        https://www.ams.org/notices/200202/fea-lenstra.pdf
        
        */
        // static int[] difs = new int[] {1,-1};
        checked
        {
            for (int i = 0; i < cases; i++)
            {
                long P = inputs[i][0];
                long Q = inputs[i][1];
                long min = inputs[i][2];

                long[] answer = minDisks(P, Q, min);

                if (answer != null)
                {
                    Console.WriteLine($"{answer[0]} {answer[1]}");
                }
                else
                {
                    Console.WriteLine("No solution");
                }

            }
        }
    }

    public static long[] minDisks(long P, long Q, long min)
    {
        long D = Q * P;
        if (IsApproximatelySquare(D))
        {
            return null;
        }
        long K = Q * (Q - P);
        long[] pellSolution = solvePells(D);

        // (x,y) and (x,-y) are both solutions which can generate positive values for future solutions
        long[] solution1 = solveGeneralPellsBrute(D, K);
        long[] solution2 = new long[] { solution1[0], -solution1[1] };

        //Turn them to blue disk and total disk count above min
        long[][] solutions = { solution1, solution2 };
        long[] answer = new long[] { 0, long.MaxValue };
        foreach (long[] xy in solutions)
        {
            long T = 0;
            long B = 0;
            long[] solution = xy;
            do
            {
                if ((solution[0] + Q) % (2 * Q) != 0 || (solution[1] + 1) % 2 != 0)
                {
                    solution = nextSolution(pellSolution, solution);
                    continue;
                }
                B = (solution[0] + Q) / (2 * Q);
                T = (solution[1] + 1) / 2;
                solution = nextSolution(pellSolution, solution);
            } while (T <= min);

            // Compare and pick the smaller one
            if (T < answer[1])
            {
                answer = new long[] { B, T };
            }

        }
        return answer;
    }


    public static long[] ContinuedFractions(long number)
    {
        // Initial integer part of the square root
        long integerPartOfSqrt = (long)Math.Sqrt(number);

        // If the number is a perfect square, then it doesn't have a non-trivial expansion
        if (integerPartOfSqrt * integerPartOfSqrt == number)
            return new long[] { integerPartOfSqrt };

        List<long> continuedFractionList = new List<long> { integerPartOfSqrt };

        // Initialization of variables
        long previousRemainder = 0;
        long denominator = 1;
        long sequenceValue = integerPartOfSqrt;

        // The loop calculates the continued fraction expansion
        // The limit assumes the worst-case scenario for periodicity
        while (sequenceValue != 2 * integerPartOfSqrt)
        {
            previousRemainder = denominator * sequenceValue - previousRemainder;
            denominator = (number - previousRemainder * previousRemainder) / denominator;
            sequenceValue = (integerPartOfSqrt + previousRemainder) / denominator;

            continuedFractionList.Add(sequenceValue);
        }

        return continuedFractionList.ToArray();
    }

    public static long[] nthConvergent(int n, long[] continuedFractions)
    {
        // The zeroth convergent is usually the integer part of the square root.
        long currentNumerator = continuedFractions[0];
        long currentDenominator = 1;

        //Return if Square
        if (continuedFractions.Length == 1)
        {
            return new long[] { currentNumerator * currentNumerator, 1 };
        }

        // The initial values for the previous convergent's numerator and denominator.
        long prevNumerator = 1;
        long prevDenominator = 0;

        // The period is the repeating part of the continued fraction.
        int periodLength = continuedFractions.Length - 1;

        // Calculate the nth convergent.
        for (int i = 1; i <= n; i++)
        {
            // Get the continued fraction coefficient.
            long coefficient = continuedFractions[(i - 1) % periodLength + 1];

            // Update the current convergent's numerator and denominator using 
            // the formula for the nth convergent.
            (currentNumerator, prevNumerator) =
                (currentNumerator * coefficient + prevNumerator, currentNumerator);

            (currentDenominator, prevDenominator) =
                (currentDenominator * coefficient + prevDenominator, currentDenominator);
        }

        // Return the nth convergent's numerator and denominator.
        return new long[] { currentNumerator, currentDenominator };
    }

    public static long[] solvePells(long D)
    {
        long[] fractions = ContinuedFractions(D);
        int i = 1;
        long x = 0;
        long y = 0;
        do
        {
            long[] convergent = nthConvergent(i, fractions);
            x = convergent[0];
            y = convergent[1];
            i++;
        } while (x * x - D * y * y != 1);

        return new long[] { x, y };
    }

    public static long[] nextSolution(long[] pellSolution, long[] curSolution)
    {
        long x1 = pellSolution[0];
        long y1 = pellSolution[1];
        long xk = curSolution[0];
        long yk = curSolution[1];
        long D = (x1 * x1 - 1) / (y1 * y1);
        return new long[] { x1 * xk + D * y1 * yk, x1 * yk + y1 * xk };
    }

    public static long[] solveGeneralPells(long D, long K)
    {
        long[] pellSolution = solvePells(D);
        int n = 1;
        long x = pellSolution[0];
        long y = pellSolution[1];

        long xPower = x;
        long yPower = y;

        while ((xPower - D * yPower) % K != 0)
        {
            n++;
            xPower *= x;
            yPower *= y;
        }

        long multiplier = (xPower - D * yPower) / K;

        return new long[] { x * multiplier, y * multiplier };
    }
    public static long[] solveGeneralPellsBrute(long D, long K)
    {

        long x = 0;
        long y = 0;
        long square = 0;
        do
        {
            y++;
            square = K + D * y * y;

        } while (square == 0 || !IsApproximatelySquare(square));
        x = (long)Math.Round(Math.Sqrt(K + D * y * y));

        return new long[] { x, y };
    }


    public static bool IsApproximatelySquare(double num)
    {
        if (num < 0) return false;  // Negative numbers aren't squares

        double sqrt = Math.Sqrt(num);
        double nearestInt = Math.Round(sqrt);

        // Compute the difference between the squared rounded value and the original number
        double difference = Math.Abs(nearestInt * nearestInt - num);

        // Tolerance can be adjusted based on the precision you need
        double tolerance = 1e-10;

        return difference <= tolerance;
    }

    public static void testingSuite()
    {
        long P = 0;
        long Q = 0;
        long min = 0;
        long[] solution = new long[2];
    }

    public long GenerateRandomLong(long minValue, long maxValue)
    {
        Random random = new Random();
        byte[] buffer = new byte[8];
        random.NextBytes(buffer);
        long longRand = BitConverter.ToInt64(buffer, 0);

        return Math.Abs(longRand % (maxValue - minValue)) + minValue;
    }
}

// 