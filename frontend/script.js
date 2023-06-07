const old = document.getElementById("old-img");
const contours = document.getElementById("contours-img");
const numbers = document.getElementById("numbers-img");
const equation = document.getElementById("equation");
const result = document.getElementById("result");
const solve = document.getElementById("solve");
const clear = document.getElementById("clear");
const upload = document.getElementById("upload");
let c = false;
let myCanvas;
function setup() {
  myCanvas = createCanvas(500, 500);
  myCanvas.parent("canvas-container");
  background(255);
  stroke(0);
  strokeWeight(1);
  fill(0);
  scale(5);
}

function draw() {
  if (c) {
    background(255);
    c = false;
  }
}
let mouseUp = true;
function mouseReleased() {
  mouseUp = true;
}
function mouseDragged() {
  if (mouseUp) {
    mouseUp = false;
    return;
  }
  // don't draw if mouse is going too fast
  if (dist(mouseX, mouseY, pmouseX, pmouseY) > 25) return;
  line(pmouseX, pmouseY, mouseX, mouseY);
}

clear.addEventListener("click", () => {
  c = true;
});

solve.addEventListener("click", async () => {
  const dataURL = canvas.toDataURL();
  const response = await fetch("http://127.0.0.1:5000/", {
    method: "POST",
    body: JSON.stringify({ dataURL }),
    headers: {
      "Content-Type": "application/json",
    },
  });
  const data = await response.json();
  try {
    equation.innerHTML = JSON.parse(data.parsedExpression).join(" ");
  } catch (err) {
    equation.innerHTML = data.parsedExpression.replaceAll('"', "");
  }
  // turn solved into int and round to 2 decimal places
  result.innerHTML = Math.round(data.solvedExpression * 1000) / 1000;
  old.src = data.oldurl;
  contours.src = data.contoursurl;
  numbers.src = data.segmentedurl;
  numbers.style["max-width"] = "480px";
  // alert(data);
});

const toBase64 = (file) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
  });
const input = document.getElementById("myFile");
let file;
if (input.files.length > 0) {
  file = input.files[0];
}
const pic = document.getElementById("pic");
upload.addEventListener("click", () => {
  toBase64(file).then(async (dataURL) => {
    const response = await fetch("http://127.0.0.1:5000/", {
      method: "POST",
      body: JSON.stringify({ dataURL }),
      headers: {
        "Content-Type": "application/json",
      },
    });
    pic.src = dataURL;

    const data = await response.json();
    try {
      equation.innerHTML = JSON.parse(data.parsedExpression).join(" ");
    } catch (err) {
      equation.innerHTML = data.parsedExpression.replaceAll('"', "");
    }
    // turn solved into int and round to 2 decimal places
    result.innerHTML = Math.round(data.solvedExpression * 1000) / 1000;
    old.src = data.oldurl;
    contours.src = data.contoursurl;
    numbers.src = data.segmentedurl;
    numbers.style["max-width"] = "480px";
  });
});
input.addEventListener("change", () => {
  file = input.files[0];
});
