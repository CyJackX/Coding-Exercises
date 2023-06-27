window.onload = function () {
  document.getElementById("new-quote").addEventListener("click", update);
  console.log("loaded");
  update();
};

function update() {
  let quote = randomQuote();
  let color = randomDarkColor();
  document.getElementById("body").style.backgroundColor = color;
  document.getElementById("text").style.color = color;
  document.getElementById("text").innerHTML = '"' + quote.text + '"';
  document.getElementById("author").innerHTML = "-" + quote.author;
  document.getElementById("tweet-quote").href =
    "https://twitter.com/intent/tweet?text=" +
    encodeURIComponent('"' + quote.text + '"\n' + "-" + quote.author);
}
function randomDarkColor() {
  var r = Math.floor(Math.random() * 128); // Random between 0 and 127
  var g = Math.floor(Math.random() * 128); // Random between 0 and 127
  var b = Math.floor(Math.random() * 128); // Random between 0 and 127

  return "rgb(" + r + "," + g + "," + b + ")";
}
function randomQuote() {
  let idx = Math.floor(Math.random() * quotes.length);
  return quotes[idx];
}

const quotes = [
  {
    text:
      "When you realize nothing is lacking, the whole world belongs to you.",
    author: "Lao Tzu"
  },
  {
    text:
      "If you are exploring truth of life, explore the present moment. Living in the present moment is the only truth of life.",
    author: "Invajy"
  },
  { text: "When you sit, everything sits with you.", author: "Shunryu Suzuki" },
  {
    text:
      "Learning the art of unlearning and replacing old learning with a new learning is the true learning.",
    author: "Invajy"
  },
  {
    text: "Muddy water is best cleared by leaving it alone.",
    author: "Alan Watts"
  }
];
