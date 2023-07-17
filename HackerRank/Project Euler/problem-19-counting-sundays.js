

// var annualSums = Array.from({ length: 400 }, x => 0)


function processData(input) {
  input = input.split(/\n+/)
  let T = parseInt(input.shift())
  // for (let i = 0; i < 400; i++) {
  //   annualSums[i] = countingSundays([1900 + i, 1, 1], [1900 + i, 12, 31]);
  // }

  for (let i = 0; i < T; i++) {
    let date1 = input[2 * i].split(/\s+/)
    let date2 = input[2 * i + 1].split(/\s+/)

    // Very annoying DATA VALIDATION STEP.
    while (BigInt(date1[0]) > BigInt(Number.MAX_SAFE_INTEGER) || BigInt(date2[0]) > BigInt(Number.MAX_SAFE_INTEGER)) {
      date1[0] = BigInt(date1[0]) - 4000000000000000n
      date2[0] = BigInt(date2[0]) - 4000000000000000n
    }
    date1 = date1.map(x => parseInt(x));
    date2 = date2.map(x => parseInt(x));
    console.log(countingSundays(date1, date2))
  }
  return
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
yar = (10 ** 16).toString()
data = '10'
for (let T = 1; T <= parseInt(data); T++) {
  data += '\n' + (10 ** 16).toString() + ' 1 1\n' + (10 ** 16 + 1000).toString() + ' 12 31'
}
processData(data)

function countingSundays(firstDate, lastDate) {
  let dayOfMonth = firstDate[2]
  let month = firstDate[1]
  let year = firstDate[0]
  let dayOfWeek = zellersCongruence(dayOfMonth, month, year);
  let count = 0;

  if (dayOfMonth != 1) {
    [dayOfMonth, month, year, dayOfWeek] = nextMonth([1, month, year, dayOfWeek]);
  }
  // //Advance to January  1st
  // while (dayOfMonth != 1 && month != 1) {   
  //   if (dayOfWeek == 7 && dayOfMonth == 1) {
  //     count++;
  //   }
  //   [dayOfMonth, month, year, dayOfWeek] = nextDay([dayOfMonth, month, year, dayOfWeek]);
  //   //console.log(dayOfWeek, dayOfMonth);
  // }
  //Skip Years based on annualSums array
  // while (year < lastDate[0] - 1) {
  //   count += annualSums[(year - 1900) % 400]
  //   year++
  // }

  //Advance to Last Day
  while (compareDates([year, month, dayOfMonth], lastDate) <= 0) {
    if (dayOfWeek == 7) {
      count++;
    }
    [dayOfMonth, month, year, dayOfWeek] = nextMonth([dayOfMonth, month, year, dayOfWeek]);

  }
  //console.log(count);
  return count;
}
function zellersCongruence(day, month, year) {
  if (month < 3) {
    month += 12;
    year -= 1;
  }

  var K = year % 100;
  var J = Math.floor(year / 100);
  var h = (day + Math.floor((13 * (month + 1)) / 5) + K + Math.floor(K / 4) + Math.floor(J / 4) - 2 * J) % 7;

  h -= 1

  // Adjust for negative result
  while (h <= 0) {
    h += 7;
  }

  return h;
}

function compareDates(day1, day2) {
  firstYear = day1[0];
  firstMonth = day1[1];
  firstDay = day1[2]
  secondYear = day2[0];
  secondMonth = day2[1];
  secondDay = day2[2]
  if (firstYear < secondYear) return -1
  if (firstYear > secondYear) return 1
  if (firstMonth < secondMonth) return -1
  if (firstMonth > secondMonth) return 1
  if (firstDay < secondDay) return -1
  if (firstDay > secondDay) return 1
  return 0
}
function nextDay(date) {
  let [dayOfMonth, month, year, dayOfWeek] = date;

  dayOfWeek = (dayOfWeek % 7) + 1;
  //I'd like to do something uber nice looking over repetitive, but also intuitive?  Are there simply too many edge cases?  I guess just IMPLEMENTATION first then!  Or embedded switches, etc?

  if (isEndOfMonth(dayOfMonth, month, year)) {
    dayOfMonth = 1;
    if (month == 12) {
      month = 1;
      year++;
    } else {
      month++;
    }
  } else {
    dayOfMonth++;
  }
  //console.log("Month:" + month + " Day:" + dayOfMonth + " Year:" + year + " was a " + dayOfWeek);
  return [dayOfMonth, month, year, dayOfWeek];
}

function nextMonth(date) {
  let [dayOfMonth, month, year, dayOfWeek] = date;

  month++
  if (month == 13) {
    year++
    month = 1
  }
  dayOfWeek = zellersCongruence(dayOfMonth, month, year)
  return [dayOfMonth, month, year, dayOfWeek];
}

function nextYear(date) {
  let [dayOfMonth, month, year, dayOfWeek] = date;

  year++;
  dayOfWeek = zellersCongruence(dayOfMonth, month, year)
  return [dayOfMonth, month, year, dayOfWeek];
}

function endOfYear(date) {
  return [31, 12, year, zellersCongruence(31, 12, year)];
}


//This seems like a good use of abstraction...
function isEndOfMonth(day, month, year) {
  switch (day) {
    case 28:
      return month == 2 && !isLeapYear(year);
    case 29:
      return month == 2;
    case 30:
      switch (month) {
        case 4:
        case 6:
        case 9:
        case 11:
          return true;
        default: return false;
      }
    case 31: return true;
    default:
      return false;
  }
}

function isLeapYear(year) {
  if (year % 400 == 0) {
    return true;
  }
  if (year % 100 == 0) {
    return false;
  }
  return year % 4 == 0;
}
