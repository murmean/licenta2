document.getElementById("fileInput").addEventListener("change", function (event) {
    let file = event.target.files[0];
    let fileInfo = document.getElementById("fileInfo");
    let fileNameDisplay = document.getElementById("fileName");

    if (file) {
        let allowedTypes = ["text/plain", "text/csv"];
        if (!allowedTypes.includes(file.type)) {
            alert("Unsupported file type! Please upload a .txt or .csv file.");
            event.target.value = ""; // Clear file input
            fileInfo.style.display = "none";
            return;
        }

        fileNameDisplay.textContent = file.name;
        fileInfo.style.display = "block";
    }
});

// ✅ Fix: Make remove file button work properly
document.getElementById("removeFileBtn").addEventListener("click", function () {
    let fileInput = document.getElementById("fileInput");
    fileInput.value = ""; // Clear file input
    document.getElementById("fileInfo").style.display = "none"; // Hide file info
});

// ✅ Function to clean processed text by removing HTML tags before download
function stripHtmlTags(text) {
    let tempElement = document.createElement("div");
    tempElement.innerHTML = text;
    return tempElement.textContent || tempElement.innerText || "";
}

// ✅ Function to Download Processed Text as .txt File (Without HTML tags)
function downloadProcessedText(text) {
    let cleanText = stripHtmlTags(text); // Remove HTML tags
    let filename = "processed_text_" + new Date().toISOString().replace(/:/g, "-") + ".txt";
    let blob = new Blob([cleanText], {type: "text/plain"});
    let link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// ✅ Fix: Ensure downloaded file has clean text
document.getElementById("textForm").addEventListener("submit", async function (event) {
    event.preventDefault();

    let formData = new FormData();
    let textInput = document.getElementById("textInput").value.trim();
    let fileInput = document.getElementById("fileInput").files[0];

    if (fileInput) {
        formData.append("file", fileInput);
    } else if (textInput) {
        formData.append("text", textInput);
    } else {
        alert("Please enter text or upload a file.");
        return;
    }

    formData.append("lowercase", document.getElementById("lowercase").checked ? "true" : "false");
    formData.append("remove_punct", document.getElementById("remove_punct").checked ? "true" : "false");
    formData.append("remove_stopwords", document.getElementById("remove_stopwords").checked ? "true" : "false");
    formData.append("lemmatize", document.getElementById("lemmatize").checked ? "true" : "false");

    try {
        let response = await fetch("/analyze", {
            method: "POST",
            body: formData
        });

        let data = await response.json();
        if (!data || data.error) {
            alert("Error processing the request: " + (data.error || "Unknown error"));
            return;
        }

        // Display sentiment
        document.getElementById("sentiment").textContent = data.sentiment.toFixed(2);

        // Display extracted topics
        let topicsList = document.getElementById("topics");
        topicsList.innerHTML = "";
        data.topics.forEach(topic => {
            let li = document.createElement("li");
            li.textContent = topic;
            topicsList.appendChild(li);
        });

        //  Highlight keywords in processed text
        let processedText = data.processed_text;
        let keywords = data.keywords || [];

        keywords.forEach(keyword => {
            let regex = new RegExp(`\\b${keyword}\\b`, "gi");
            processedText = processedText.replace(regex, `<span class="highlight">${keyword}</span>`);
        });

        //  Display processed text
        document.getElementById("processedText").innerHTML = processedText;
        document.getElementById("results").style.display = "block";

        //  Show the Download button
        let downloadBtn = document.getElementById("downloadTextBtn");
        downloadBtn.style.display = "block";
        downloadBtn.onclick = function () {
            downloadProcessedText(processedText);
        };

    } catch (error) {
        console.error("Error during analysis:", error);
        alert("An error occurred while processing the text.");
    }
});

document.getElementById("summarizeBtn").addEventListener("click", async function () {
    let textInput = document.getElementById("textInput").value.trim();

    if (!textInput) {
        alert("Please enter text before summarizing.");
        return;
    }

    let formData = new FormData();
    formData.append("text", textInput);

    try {
        let response = await fetch("/summarize", {
            method: "POST",
            body: formData
        });

        let data = await response.json();
        if (!data || data.error) {
            alert("Error generating summary: " + (data.error || "Unknown error"));
            return;
        }

        // ✅ Display Summary
        document.getElementById("summaryText").textContent = data.summary;
        document.getElementById("summaryContainer").style.display = "block";

    } catch (error) {
        console.error("Error during summarization:", error);
        alert("An error occurred while generating the summary.");
    }
});

