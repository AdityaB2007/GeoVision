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
const errorMessage = document.getElementById("errorMessage");

const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");

const predictionClass = document.getElementById("predictionClass");
const predictionDescription = document.getElementById("predictionDescription");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceFill = document.getElementById("confidenceFill");
const probabilityList = document.getElementById("probabilityList");

const modelNameValue = document.getElementById("modelNameValue");
const inputSpecValue = document.getElementById("inputSpecValue");
const modelStatusValue = document.getElementById("modelStatusValue");

// Track the active File object explicitly so drag-and-drop works identically
// to the native file picker (event.dataTransfer.files is not placed in fileInput).
let selectedFile = null;
let apiBase = "";

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove("hidden");
}

function clearError() {
    errorMessage.textContent = "";
    errorMessage.classList.add("hidden");
}

async function resolveApiBase() {
    if (window.location.protocol === "file:") {
        return "http://127.0.0.1:8000";
    }
    try {
        const res = await fetch("/health");
        if (res.ok) {
            return "";
        }
    } catch (_) {
        // Fallback when HTML is served from a separate static server port.
    }
    return "http://127.0.0.1:8000";
}

async function checkBackendHealth() {
    apiBase = await resolveApiBase();
    try {
        const response = await fetch(`${apiBase}/health`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        if (data.model_loaded) {
            statusDot.classList.remove("offline");
            const accLabel =
                typeof data.test_accuracy === "number"
                    ? ` · ${(data.test_accuracy * 100).toFixed(1)}% Test Acc`
                    : "";
            statusText.textContent = `Model Ready (${data.backbone || data.model_name})${accLabel}`;
            modelNameValue.textContent = data.backbone || data.model_name || "GeoVision";
            inputSpecValue.textContent = `Satellite RGB (${data.input_size || 64}x${data.input_size || 64})`;
            modelStatusValue.textContent = "● Ready";
            modelStatusValue.classList.add("online");
        } else {
            statusDot.classList.add("offline");
            statusText.textContent = "Backend Online (No Model Loaded)";
            modelStatusValue.textContent = "● No Checkpoint";
            modelStatusValue.classList.remove("online");
        }
    } catch (_) {
        statusDot.classList.add("offline");
        statusText.textContent = "Backend Offline";
    }
}

document.addEventListener("DOMContentLoaded", () => {
    checkBackendHealth();
});

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
    clearError();

    if (!file.type.startsWith("image/")) {
        showError("Please upload a valid image file (PNG, JPG, or JPEG).");
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        showError("File size must be below 10 MB.");
        return;
    }

    selectedFile = file;
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
    selectedFile = null;
    fileInput.value = "";
    imagePreview.src = "";
    clearError();

    previewContainer.classList.add("hidden");
    dropZone.classList.remove("hidden");

    classifyButton.disabled = true;

    results.classList.add("hidden");
    emptyState.classList.remove("hidden");
});

function renderPrediction(data) {
    predictionClass.textContent = data.display_name || data.predicted_class;
    predictionDescription.textContent = data.description || "";

    const pct = Number(data.confidence_percentage).toFixed(1);
    confidenceValue.textContent = `${pct}%`;
    confidenceFill.style.width = `${Math.min(100, Math.max(0, data.confidence_percentage))}%`;

    probabilityList.innerHTML = "";
    (data.probabilities || []).forEach((item, index) => {
        const row = document.createElement("div");
        row.className = index === 0 ? "probability-row is-top" : "probability-row";

        const label = document.createElement("span");
        label.textContent = item.display_name || item.class_name;

        const track = document.createElement("div");
        track.className = "probability-track";

        const fill = document.createElement("div");
        fill.className = "probability-fill";
        const widthPct = Math.min(100, Math.max(0, Number(item.percentage)));
        fill.style.width = `${widthPct}%`;
        track.appendChild(fill);

        const value = document.createElement("strong");
        value.textContent = `${Number(item.percentage).toFixed(1)}%`;

        row.appendChild(label);
        row.appendChild(track);
        row.appendChild(value);
        probabilityList.appendChild(row);
    });

    modelNameValue.textContent = data.backbone || data.model_name;
    inputSpecValue.textContent = `Satellite RGB (${data.input_size}x${data.input_size})`;
    modelStatusValue.textContent = `● ${Number(data.inference_time_ms).toFixed(1)} ms`;
    modelStatusValue.classList.add("online");

    emptyState.classList.add("hidden");
    results.classList.remove("hidden");
}

classifyButton.addEventListener("click", async () => {
    if (!selectedFile) {
        showError("Please select a satellite image before classifying.");
        return;
    }

    clearError();
    classifyButton.disabled = true;
    buttonText.textContent = "Analyzing...";
    loader.classList.remove("hidden");

    try {
        if (!apiBase) {
            apiBase = await resolveApiBase();
        }
        const formData = new FormData();
        formData.append("file", selectedFile);

        const response = await fetch(`${apiBase}/predict`, {
            method: "POST",
            body: formData,
        });

        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(payload.detail || `Prediction failed (HTTP ${response.status}).`);
        }

        renderPrediction(payload);
    } catch (error) {
        showError(
            error.message ||
                "Could not reach the GeoVision FastAPI server. Make sure uvicorn is running."
        );
    } finally {
        buttonText.textContent = "Classify Image";
        loader.classList.add("hidden");
        classifyButton.disabled = !selectedFile;
    }
});
