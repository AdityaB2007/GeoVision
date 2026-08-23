const fileInput = document.getElementById("fileInput");
const browseButton = document.getElementById("browseButton");
const dropZone = document.getElementById("dropZone");

const previewContainer = document.getElementById("previewContainer");
const imagePreview = document.getElementById("imagePreview");

const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");

const removeButton = document.getElementById("removeButton");
const classifyButton = document.getElementById("classifyButton");

const emptyState = document.getElementById("emptyState");
const results = document.getElementById("results");

const buttonText = document.getElementById("buttonText");
const loader = document.getElementById("loader");


browseButton.addEventListener("click", () => {
    fileInput.click();
});


fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
        handleFile(fileInput.files[0]);
    }
});


dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropZone.classList.add("dragover");
});


dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
});


dropZone.addEventListener("drop", (event) => {
    event.preventDefault();

    dropZone.classList.remove("dragover");

    const files = event.dataTransfer.files;

    if (files.length > 0) {
        handleFile(files[0]);
    }
});


function handleFile(file) {

    if (!file.type.startsWith("image/")) {
        alert("Please upload an image file.");
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        alert("File size must be below 10 MB.");
        return;
    }

    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);

    const reader = new FileReader();

    reader.onload = (event) => {
        imagePreview.src = event.target.result;

        dropZone.classList.add("hidden");
        previewContainer.classList.remove("hidden");

        classifyButton.disabled = false;
    };

    reader.readAsDataURL(file);
}


function formatFileSize(bytes) {

    if (bytes < 1024) {
        return `${bytes} B`;
    }

    if (bytes < 1024 * 1024) {
        return `${(bytes / 1024).toFixed(1)} KB`;
    }

    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}


removeButton.addEventListener("click", () => {

    fileInput.value = "";
    imagePreview.src = "";

    previewContainer.classList.add("hidden");
    dropZone.classList.remove("hidden");

    classifyButton.disabled = true;

    results.classList.add("hidden");
    emptyState.classList.remove("hidden");
});


classifyButton.addEventListener("click", () => {

    classifyButton.disabled = true;

    buttonText.textContent = "Analyzing...";
    loader.classList.remove("hidden");

    /*
     * Temporary mock prediction.
     *
     * This will eventually be replaced with:
     *
     * fetch("/predict", {
     *     method: "POST",
     *     body: formData
     * })
     *
     * once the Python backend is implemented.
     */

    setTimeout(() => {

        buttonText.textContent = "Classify Image";
        loader.classList.add("hidden");

        emptyState.classList.add("hidden");
        results.classList.remove("hidden");

        classifyButton.disabled = false;

    }, 1200);
});