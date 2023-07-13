/*
Hm, time to crib form the previous chain problem?  Seems pretty similar?
--
Okay, timed out on just the last one, huh.  More dynamic programming needed methinks.
--
Okay, nowhere close to passing last test?  New dynamic stuff needed.  Probably need to pad 0s and do permutations on every step of the chain, okay!  I grok it at least.
--
Damn, still timed out.  All the padded zeros are probably so redundant...
--
Getting rid of the zeros entirely didn't work either.  Maybe an in-between way to do it?  Not to mention even something like 9999999 will generate a lot of redundant overhead :-/
--
Okay, just got to do it with a method that culls the permutations early.
--
Okay, so now it doesn't do a lot of dumb redundant calcs but it still isn't getting past 1M...  Maybe a map is inefficient?  All their keys are either 89 or 1!
--
Something's weird...messing up test results depending on how I test?  Some sort of memory issue?
--
Annoying that pure brute force gets to 1.6M compared to dynamic thing?
--
Maybe...optimize for numbers instead of using strings?
--
Nah, I got to figure out how to generate all the unique combinations and only test those numbers.
--
Still timing out?
--
Man, forget the memory array, doing it without gets to 6.8M.  I bet if I can just get a good permutation algorithm that'll be it.
--
Alright, testing the permutationCount.
--
Okay, on freeCodeCamp this was only K = 7.  K = 200 on HackerRank?  Hardcore!
Going to mod the permutations to be not long strings?  Possibly.  But then that requires BigInt, and forums say it can be done with 32-bits...
--
What if each number is represented as an array of the digit counts?  This skips straight to representing the combination, since we only really care how many 1s,2s, etc!
No need to generate the list of permutations, we can iterate straight through that, right?
--
Getting the hint that something uber-dynamic is necessary.
for an n-digit number, the highest digit square sum is n*9^2.  This is pretty manageable if I can find a way to count how many numbers sum to a given number, but based on the mere additition of digits instead of having to iterate through 10^n.
--
For each result of 1,4,9..., you can add the results again.
--
New solution, close!
--
Ah, I'm missing the tens...or I guess every trailing 0 situation.
*/
let k = 3

function squareDigitChains(k) {

  const list = Array.from({ length: k * 81 + 1 }, () => 0);
  var memory = [0];

  //Insert incredible dynamic programming solution here:
  for (let i = 1; i <= k; i++) { //This loop for # of digits...
    //this step needs the dynamic part...?
    let nextMem = [];
    for (let mem of memory) {
      for (let digit = 1; digit < 10; digit++) {

        list[mem + digit ** 2] += k - i + 1;
        nextMem.push(mem + digit ** 2);

      }
    }
    memory = nextMem;
  }

  //Process the array afterwards.
  let count = 0;
  for (let i = 1; i < list.length; i++) {
    let n = i;
    while (n != 89 && n != 1) {
      n = digitSquareSum(n);
    }
    if (n == 89) {
      count = (count + list[i]) % (10 ** 9 + 7)
    }
  }
  console.log(list);

  return count;
}

// console.log(digitSquareSum(44));
function digitSquareSum(n) {
  let sum = 0;
  while (n > 0) {
    sum += (n % 10) ** 2;
    n = Math.floor(n / 10);
  }
  return sum;
}
//console.log(permutationCount('Mississippi'));
function permutationCount(str) {
  str = str.split('').sort();
  let answer = 1
  let char, count;
  for (let i = 0; i < str.length; i++) {
    answer *= i + 1
    if (str[i] == char) {
      count++;
      answer /= count;
    } else {
      char = str[i];
      count = 1;
    }
  }
  return answer;
}

function factorial(n) {
  let prod = 1;
  while (n > 1) {
    prod *= n;
    n--;
  }
  return prod
}
//console.log(generateDigitCombinations(20).length);
function generateDigitCombinations(digits) {
  let list = [];

  function generate(arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], depth = 0, index = 1) {

    if (depth == digits) {
      return
    }

    for (let i = index; i <= 9; i++) {
      let newArr = [...arr];
      newArr[i]++;
      list.push(newArr);
      generate(newArr, depth + 1, i);
    }
  }

  generate();
  return list;
}

// console.log(permutationCount('aaabbbccc'))
// //Returns all sets of r from the options, recursive
// function getPermutations(options) {
//   let results = [];

//   function permute(list, prefix = '') {
//     if (!list.length) {
//       results.push(prefix);
//     }
//     let dupeCheck = []; //Cull duplicates early?
//     for (let i = 0; i < list.length; i++) {
//       let next = prefix.concat(list[i]);
//       if (dupeCheck.indexOf(next) < 0) {
//         let newList = [...list];
//         newList.splice(i, 1);
//         permute(newList, next);
//         dupeCheck.push(next);
//       }
//     }
//   }

//   permute(options);
//   //console.log(options, results, calcs);
//   return results.map(x => parseInt(x));
// }
let list2 = getDigitCombinations(10**2);

function getDigitCombinations(limit) {
  //const digits = [0,1,2,3,4,5,6,7,8,9];
  let results = [];

  function combosWithReplacement(prev = '', index = 0) {
    if (prev.length == Math.log10(limit)) {
      return
    }

    for (let i = index; i < 10; i++) {
      let next = prev.concat(i);
      results.push(next);
      combosWithReplacement(next, i);
    }
  }

  combosWithReplacement();
  results = results.map(x => parseInt(x));
  results = Array.from(new Set(results));
  return results;
}

function processData(input) {
  return squareDigitChains(input)
}
console.log(processData(k))