ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
ort.env.wasm.numThreads = 1;
const sessionOptions = { executionProviders: ["wasm"] };

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

ctx.lineWidth = 20;
ctx.lineCap = "round";
ctx.strokeStyle = "black";

let drawing = false;

canvas.addEventListener("mousedown", (e) => {
    drawing = true;
    const rect = canvas.getBoundingClientRect();
    ctx.beginPath();
    ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
});
canvas.addEventListener("mouseup", () => {
    drawing = false;
    ctx.beginPath();
});
canvas.addEventListener("mouseout", () => {
    drawing = false;
    ctx.beginPath();
});
canvas.addEventListener("mousemove", draw);

function draw(e) {
    if (!drawing) return;
    const rect = canvas.getBoundingClientRect();
    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
}

// Clear button 
document.getElementById("clear").onclick = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    document.getElementById("result").textContent = "?";
};

document.getElementById("predict").onclick = async () => {

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const inputTensor = preprocess(imageData);

    const arr = Array.from(inputTensor.data).slice(0, 20);
    console.log("Primières valeurs du tensor (0..1) :", arr);

    const session = await ort.InferenceSession.create("model.onnx", sessionOptions);

    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];
    console.log("Input name:", inputName, "Output name:", outputName);

    const feeds = {};
    feeds[inputName] = inputTensor;

    const results = await session.run(feeds);

    const output = results[outputName].data;
    console.log("Output probabilities:", output);

    // 3) Interpréter prédiction
    const prediction = output.indexOf(Math.max(...output));
    console.log("Predicted digit:", prediction);
    document.getElementById("result").textContent = prediction;
};

function preprocess(imageData) {
    const tmp = document.createElement("canvas");
    tmp.width = 28;
    tmp.height = 28;
    const tctx = tmp.getContext("2d");


    const src = document.createElement("canvas");
    src.width = imageData.width;
    src.height = imageData.height;
    const sctx = src.getContext("2d");
    sctx.putImageData(imageData, 0, 0);

    tctx.fillStyle = "white";
    tctx.fillRect(0, 0, 28, 28);
    tctx.drawImage(src, 0, 0, 28, 28);

    const small = tctx.getImageData(0, 0, 28, 28).data;

    const arr = new Float32Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
        const r = small[i * 4 + 0];
        const g = small[i * 4 + 1];
        const b = small[i * 4 + 2];
        const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        const inverted = 255 - lum;       
        arr[i] = inverted / 255.0;        
    }

    return new ort.Tensor("float32", arr, [1, 1, 28, 28]);
}
