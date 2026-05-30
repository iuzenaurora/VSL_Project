const video = document.getElementById("video")
const canvas = document.getElementById("canvas")
const ctx = canvas.getContext("2d")
const socket = io();

// ===========================
// START CAMERA
// ===========================
async function startCamera(){
    const stream = await navigator.mediaDevices.getUserMedia({
        video: {
            width: { ideal: 1280 },
            height: { ideal: 720 }
        }
    })
    video.srcObject = stream
}
startCamera()

// ===========================
// SEND FRAME
// ===========================
let isProcessing = false;
let isVideoUploading = false;

async function sendFrame(){
    if(isVideoUploading || isProcessing) return;
    if(video.videoWidth === 0){
        setTimeout(sendFrame, 100);
        return;
    }

    isProcessing = true;
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    ctx.drawImage(video, 0, 0)

    let imageData = canvas.toDataURL("image/jpeg", 0.75);
    let drawSkeleton = document.getElementById("skeletonToggle").checked;

    socket.emit("process_frame", {
        image: imageData,
        draw_skeleton: drawSkeleton
    });
}

// ===========================
// LẮNG NGHE WEBSOCKET
// ===========================
socket.on("frame_result", (result) => {
    if(result.success){
        document.getElementById("word").innerText = result.word || "---";
        document.getElementById("confidence").innerText = result.confidence;

        if(result.sentence){
            document.getElementById("sentence").innerText = result.sentence;
        }
        document.getElementById("buffer").innerText = result.buffer_size;

        if(result.frame_buffer_size !== undefined) {
            let pb = document.getElementById("frameProgress");
            let pct = (result.frame_buffer_size / 60) * 100;
            pb.style.width = pct + "%";
            
            if (result.frame_buffer_size === 60) {
                pb.innerText = "60/60 Frames (AI Đang nhận diện...)";
                pb.classList.replace("bg-info", "bg-success");
            } else {
                pb.innerText = result.frame_buffer_size + "/60 Frames (Đang thu thập...)";
                pb.classList.replace("bg-success", "bg-info");
            }
        }
        
        if(result.skeleton_img) {
            document.getElementById("skeletonOverlay").src = result.skeleton_img;
            document.getElementById("skeletonOverlay").style.display = "block";
        } else {
            document.getElementById("skeletonOverlay").style.display = "none";
        }
    }
    isProcessing = false;
    setTimeout(sendFrame, 15); 
});

setTimeout(sendFrame, 100);

// ===========================
// TRANSLATE
// ===========================
async function translateNow(){
    let response = await fetch("/translate_now", { method:"POST" })
    let result = await response.json()
    if(result.success){
        document.getElementById("sentence").innerText = result.sentence
    }
}

// ===========================
// RESET
// ===========================
async function resetAll(){
    await fetch("/reset", { method:"POST" })
    document.getElementById("word").innerText = "---"
    document.getElementById("sentence").innerText = ""
    document.getElementById("buffer").innerText = "0"

    let pb = document.getElementById("frameProgress")
    pb.style.width = "0%"
    pb.innerText = "0/60 Frames (Đang chờ...)"
}

// ===========================
// UPDATE THRESHOLD
// ===========================
async function updateThreshold(){
    let slider = document.getElementById("thresholdSlider")
    let val = slider.value
    document.getElementById("thresholdValue").innerText = val

    await fetch("/update_threshold", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({ threshold: val })
    })
}

// ===========================
// TOGGLE SKELETON
// ===========================
function toggleSkeleton(){
    let drawSkeleton = document.getElementById("skeletonToggle").checked
    if(!drawSkeleton){
        document.getElementById("skeletonOverlay").style.display = "none"
    }
}

// ===========================
// VIDEO UPLOAD
// ===========================
async function uploadVideo(){
    isVideoUploading = true;
    let file = document.getElementById("videoUpload").files[0]

    if(!file){
        alert("Chọn video trước")
        isVideoUploading = false;
        return
    }

    let formData = new FormData()
    formData.append("video", file)

    document.getElementById("videoResult").innerHTML = "Đang xử lý..."

    let response = await fetch("/predict_video", {
        method:"POST",
        body:formData
    })

    let result = await response.json()
    if(result.success){
        document.getElementById("videoResult").innerHTML = `
        <h5>Từ nhận diện</h5>
        <p>${result.words.join(" → ")}</p>
        <h5>Câu dịch</h5>
        <p>${result.sentence}</p>
        `
    }
    isVideoUploading = false;
}

// ===========================
// DICTIONARY
// ===========================
let vocabularyList = [];
async function loadVocabulary() {
    try {
        let response = await fetch("/api/vocabulary");
        let result = await response.json();
        if (result.success) {
            vocabularyList = result.vocabulary;
            let select = document.getElementById("vocabSelect");
            select.innerHTML = "";
            vocabularyList.forEach((item, index) => {
                let option = document.createElement("option");
                option.value = index;
                option.text = item.label;
                select.appendChild(option);
            });
        }
    } catch (e) {
        console.error("Lỗi khi tải từ điển: ", e);
    }
}

function playVocabVideo() {
    let select = document.getElementById("vocabSelect");
    let index = select.value;
    if (index !== "") {
        let item = vocabularyList[index];
        let videoPlayer = document.getElementById("vocabVideo");
        videoPlayer.src = "/dataset_video/" + item.video;
        document.getElementById("vocabLabel").innerText = item.label;
    }
}

loadVocabulary();
