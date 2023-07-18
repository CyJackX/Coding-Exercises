function processData(input) {
  input = input.split(/\n+/)
  let T = parseInt(input.shift())
  let arr = Array.from({ length: 10 ** 5 + 1 }, _ => 0);
  sumAmicableNum(arr);
  let sum = 0;
  arr = arr.map((value) => {
    let temp = sum;
    sum += value;
    return temp;
  });
  for (let i = 0; i < T; i++) {
    console.log(arr[input[i]]);
  }
}
process.stdin.resume();
process.stdin.setEncoding("ascii");
_input = "";
process.stdin.on("data", function (input) {
  _input += input;
});

process.stdin.on("end", function () {
  processData(_input);
});

function sumAmicableNum(arr) {

  for (let a = 220; a < arr.length; a++) {
    if (arr[a]) {
      continue;
    }
    let b = sumOfProperDivisors(a);
    if (a != b && sumOfProperDivisors(b) == a) {
      arr[a] = a
      arr[b] = b
    }
  }
}
function sumOfProperDivisors(n) {
  let sum = 1;
  for (let i = 2; i * i <= n; i++) {
    if (n % i == 0) {
      sum += i;
      if (i * i != n) {
        sum += n / i;
      }
    }
  }
  //   console.log("Sum of Proper Divisors of " + n + " is " +sum);
  return sum;
}

processData('1\n300')