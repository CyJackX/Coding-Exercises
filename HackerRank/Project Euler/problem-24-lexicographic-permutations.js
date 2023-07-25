function processData(input) {
    input = input.split('\n').map(x => parseInt(x))
    input.shift()
    for (let x of input) {
        console.log(lexicographicPermutations(x - 1))
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

function lexicographicPermutations(n) {
    let perm = 'abcdefghijklm'.split('')
    let ret = [];
    for (let i = perm.length - 1; i >= 0; i--) {
        let f = factorial(i)
        let j = 0;
        while (n >= f) {
            n -= f;
            j++;
        }
        ret.push(perm.splice(j, 1));
        //console.log(ret);
    }

    return ret.join('');
}
function factorial(n) {
    let prod = 1;
    for (let i = 2; i <= n; i++) {
        prod *= i;
    }
    return prod;
}