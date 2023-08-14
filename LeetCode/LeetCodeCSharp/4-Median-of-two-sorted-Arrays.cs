using System;
public class Solution4 {
    public static double FindMedianSortedArrays(int[] nums1, int[] nums2) {
        int[] merged = Merge(nums1,nums2);
        return Median(merged);
    }

    public static int[] Merge(int[] nums1, int[] nums2){
        int[] merged = new int[nums1.Length + nums2.Length];
        int i = 0;
        int j = 0;
        int k = 0;
        // Step 1: Loop until you've gone through one of the arrays entirely.
        while(j < nums1.Length && k < nums2.Length) {
            // Step 2: For every iteration, compare the current elements.
            if(nums1[j] <= nums2[k]) {
                // Step 3: Place it in the merged array and move to the next element in nums1.
                merged[i++] = nums1[j++];
            } else {
                // Place the current element of nums2 in the merged array.
                merged[i++] = nums2[k++];
            }
        }

        // Step 4: After the loop, if there are still elements left in either array.
        while(j < nums1.Length) {
            merged[i++] = nums1[j++];
        }

        while(k < nums2.Length) {
            merged[i++] = nums2[k++];
        }

        return merged;
    }

    public static double Median(int[] arr){
        int mid = arr.Length/2;
        if(arr.Length % 2 == 0){
            return (arr[mid] + arr[mid - 1])/2.0;
        }
        return arr[mid];
    }
}


// public class Program4 {
//     public static void Main(string[] args) {
//         int[] nums1 = {1, 2};
//         int[] nums2 = {3, 4};

//         double result = Solution4.FindMedianSortedArrays(nums1, nums2);

//         Console.WriteLine($"The median is: {result}");
//     }
// }
