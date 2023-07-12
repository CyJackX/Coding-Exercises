/*
Problem 93: Arithmetic expressions
By using each of the digits from the set, {1, 2, 3, 4}, exactly once, and making use of the four arithmetic operations (+, −, *, /) and brackets/parentheses, it is possible to form different positive integer targets.

For example,

8 = (4 * (1 + 3)) / 2
14 = 4 * (3 + 1 / 2)
19 = 4 * (2 + 3) − 1
36 = 3 * 4 * (2 + 1)
Note that concatenations of the digits, like 12 + 34, are not allowed.

Using the set, {1, 2, 3, 4}, it is possible to obtain thirty-one different target numbers of which 36 is the maximum, and each of the numbers 1 to 28 can be obtained before encountering the first non-expressible number.

Find the set of four distinct digits, a < b < c < d, for which the longest set of consecutive positive integers, 1 to n, can be obtained, giving your answer as a string: abcd.
--
An interesting question, but it seems like eval() might be something new for me to use?
It's the brackets that are the hardest part, the operators should be easy enough.

Okay, the digits can be in any order.
The operators can be repeated.
Parentheses variations:
(ab)cd
ab(cd)
a(bc)d
a(bcd)
(abc)d
(ab)(cd)
((ab)c)d
(a(bc))d
a((bc)d)
a(b(cd))
Those seem most annoying!
I could probably just iterate whether any given number has a left or right parentheses next to it?
Is this going to explode in O(n)?
4! permutations of abcd times 4^3 operator options times 7? parentheses options * 6^4 options for a<b<c<d = 13934592
Makes more sense to tally up all versions, then compare, right?  I don't think evaluating every combination for whether it can progress linearly makes any sense.
--
Haven't thought about how all the combinations when reversed together would be redundant.  So, half aren't necessary?  But not EACH half, just one of the halves...?
Considering that I am lumping together these different sets of permutations.
Maybe I can just hardcode the 5 parentheticals, so then when everything else reverses it's different?  But, subtraction is a thing, isn't it!  That works differently left to right than otherwise.

--
hardcoding a list is fine when it's a const
--
Of course, instant time out.
Every combination generates a TON of redundancies
Maybe there's a way to work backwards?
There are many arrangements that make duplicate values, and anything that searches through all of them is going to be too slow.
*/
const ops = generatePermutations(['*', '+', '-', '/'], 3);
const parentheticals = [
  'd1 op1 d2 op2 d3 op3 d4',
  '(d1 op1 d2) op2 d3 op3 d4',
  'd1 op1 (d2 op2 d3) op3 d4',
  'd1 op1 d2 op2 (d3 op3 d4)',
  '(d1 op1 d2 op2 d3) op3 d4',
  'd1 op1 (d2 op2 d3 op3 d4)',
  '(d1 op1 d2) op2 (d3 op3 d4)',
  '((d1 op1 d2) op2 d3) op3 d4',
  '(d1 op1( d2 op2 d3)) op3 d4',
  'd1 op1 ((d2 op2 d3) op3 d4)',
  'd1 op1 (d2 op2( d3 op3 d4))'
];
function arithmeticExpressions() {
  let abcd;
  let max = 0;
  for (let a = 1; a <= 9; a++) {
    for (let b = a + 1; b <= 9; b++) {
      for (let c = b + 1; c <= 9; c++) {
        for (let d = c + 1; d <= 9; d++) {
          let count = countMax(expressionsChain(a, b, c, d));
          if (count > max) {
            max = count
            abcd = '' + a + b + c + d;
          }

        }
      }
    }
  }
  console.log(abcd);
  return abcd;
}

//Return a set of all the positive integer values.
function expressionsChain(a, b, c, d) {
  let permutations = getPermutations([a, b, c, d]);
  let values = new Set();
  let debug = [];

  for (let perms of permutations) {
    for (let op of ops) {
      for (let p of parentheticals) {
        let [d1, d2, d3, d4] = perms.split('');
        let [op1, op2, op3] = op.split('');
        let str = p.replace('op1', op1)
          .replace('op2', op2)
          .replace('op3', op3)
          .replace('d1', d1)
          .replace('d2', d2)
          .replace('d3', d3)
          .replace('d4', d4);
        let value = eval(str);
        //console.log(d1, d2, d3, d4, str, "=", value);
        debug.push([str, value]);

        if (Number.isInteger(value) && value > 0) {
          values.add(value);

        }
      }
    }
  }
  debug.sort(a,b => a[1] - b[1]);
  console.log(debug.join('\n'));
  return values;
}
console.log("(1,2,3,4)", countMax(expressionsChain(1, 2, 3, 4)));
function countMax(set) {
  let count = 0;
  while (set.has(count + 1)) {
    count++
  }
  return count;
}

//Generates permutations of size n from an array where repeats are allowed.
function generatePermutations(arr, n) {
  let list = [];

  function build(str = '') {
    if (str.length == n) {
      list.push(str);
      return
    }

    for (let elem of arr) {
      build(str + elem);
    }
  }

  build();
  return list;
}

//Generates permutations of an array.
function getPermutations(options) {
  let results = [];

  function permute(list, prefix = '') {
    if (!list.length) {
      results.push(prefix);
    }
    let dupeCheck = []; //Cull duplicates early?
    for (let i = 0; i < list.length; i++) {
      let next = prefix.concat(list[i]);
      if (dupeCheck.indexOf(next) < 0) {
        let newList = [...list];
        newList.splice(i, 1);
        permute(newList, next);
        dupeCheck.push(next);
      }
    }
  }

  permute(options);
  //console.log(options, results, calcs);
  return results;
}

//arithmeticExpressions();
