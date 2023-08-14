using System.Diagnostics;
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int val=0, TreeNode left=null, TreeNode right=null) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
public class Solution113
{
    public IList<IList<int>> PathSum(TreeNode root, int targetSum)
    {
        if (root == null){
            return new List<IList<int>>();
        }
        var nodeQueue = new Queue<(TreeNode, List<int>, int)>();
        nodeQueue.Enqueue((root, new List<int>() { root.val }, root.val));
        var solutions = new List<IList<int>>();

        while (nodeQueue.Count > 0)
        {
            var nodeData = nodeQueue.Dequeue();
            (var node, var path, var sum) = nodeData;

            bool isLeaf = true;
            foreach (var child in new[] { node.left, node.right })
            {
                if (child != null)
                {
                    isLeaf = false;
                    var childPath = path.ToList();
                    childPath.Add(child.val);
                    nodeQueue.Enqueue((child, childPath, sum + child.val));
                }
            }

            if (isLeaf && sum == targetSum)
            {
                solutions.Add(path);
            }
        }

        return solutions;

    }
}
[DebuggerDisplay("{val}")]
public class TreeNode
{
    public int val;
    public TreeNode? left;
    public TreeNode? right;
    public TreeNode(int val = 0, TreeNode? left = null, TreeNode? right = null)
    {
        this.val = val;
        this.left = left;
        this.right = right;
    }

    public static TreeNode CreateTree(int[] nodes)
    {   
        if(nodes.Length == 0){
            return null;
        }
        var root = new TreeNode(nodes[0], null, null);
        var layer = new List<TreeNode>() { root };
        for (int i = 1; i < nodes.Length; i++)
        {
            TreeNode node = layer[0];
            layer.RemoveAt(0);

            if (nodes[i] != -1001)
            {
                var leftChild = new TreeNode(nodes[i], null, null);
                node.left = leftChild;
                layer.Add(leftChild);
            }

            i++;

            if (i < nodes.Length && nodes[i] != -1001)
            {
                var rightChild = new TreeNode(nodes[i], null, null);
                node.right = rightChild;
                layer.Add(rightChild);
            }
        }
        return root;
    }
}



// public class Program113
// {
//     public static void Main(string[] args)
//     {
//         // int[] nodes = new int[] { 5, 4, 8, 11, -1001, 13, 4, 7, 2, -1001, -1001, 5, 1 };
//         int[] nodes = new int[]{};

//         var root = TreeNode.CreateTree(nodes);

//         var results = (new Solution113()).PathSum(root, 22).ToList();

//         foreach (List<int> innerList in results)
//         {
//             Console.WriteLine(string.Join(", ", innerList));
//         }
//     }
// }