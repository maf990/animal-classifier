let model; // Variabel untuk menyimpan model

// Load model saat halaman dibuka
async function loadModel() {
    model = await tf.loadGraphModel('http://127.0.0.1:5500/model/tfjs_model/model.json');
    console.log("✅ Model Loaded!");
}
loadModel();

// Preview gambar saat diunggah
document.getElementById("imageUpload").addEventListener("change", function(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById("imagePreview").src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
});

// Preprocessing gambar sebelum dikirim ke model
async function preprocessImage(image) {
    let tensor = tf.browser.fromPixels(image)
        .resizeNearestNeighbor([64, 64]) // Sesuaikan dengan ukuran input model
        .toFloat()
        .div(tf.scalar(255)) // Normalisasi ke [0,1]
        .expandDims(); // Tambah dimensi batch
    return tensor;
}

// Fungsi untuk melakukan prediksi
async function predictImage() {
    const image = document.getElementById("imagePreview");
    if (!image.src || image.src === window.location.href) {
        alert("Please upload an image first!");
        return;
    }

    const tensor = await preprocessImage(image);
    const prediction = model.predict(tensor);
    const predictedClass = prediction.argMax(1).dataSync()[0];

    const labels = ["Dog", "Cat", "Wild"]; // Sesuaikan dengan labels.txt
    document.getElementById("predictionResult").innerText = `Prediction: ${labels[predictedClass]}`;
}
