function processData(input) {
    input = input.split('\n').map(x => parseInt(x))
    input.shift()
    for(let inp of input){
        sumFactorialDigits(inp)
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

function sumFactorialDigits(n) {
  let int = BigInt(factorial(n)).toString();
//   console.log(int);
  let sum = 0;
  for(let i = 0 ; i < int.length; i++){
    sum += parseInt(int[i]);
  }
  console.log(sum);
  return sum;
}
//Need big integer to make it all work...
function factorial(n){
  let product = 1n;
  for(let i = 1n ; i <= n ; i++){
    product*=i;
  }
  return product;
}
