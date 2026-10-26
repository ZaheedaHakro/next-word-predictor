
const inputText = document.getElementById("inputText");
const ghostText = document.getElementById("ghostText");

let bestWord = "";
let debounceTimer = null;
let requestNumber = 0;

function renderGhost(text, suggestion = "") {
    ghostText.replaceChildren();

    const typedText = document.createElement("span");
    typedText.textContent = text;
    ghostText.appendChild(typedText);

    if (suggestion) {
        const ghostSuggestion = document.createElement("span");
        ghostSuggestion.className = "ghost-suggestion";

        // Add a separating space unless one already exists.
        ghostSuggestion.textContent =
            (/\s$/.test(text) ? "" : " ") + suggestion;

        ghostText.appendChild(ghostSuggestion);
    }

    // Keep the overlay synchronized with textarea scrolling.
    ghostText.scrollTop = inputText.scrollTop;
    ghostText.scrollLeft = inputText.scrollLeft;
}

async function fetchSuggestion(text, currentRequest) {
    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                text: text,
                top_k: 1,
                generate: false,
                num_words: 1
            })
        });

        const data = await response.json();

        // Ignore responses for older text.
        if (
            currentRequest !== requestNumber ||
            inputText.value !== text
        ) {
            return;
        }

        if (!response.ok) {
            bestWord = "";
            renderGhost(text);
            return;
        }

        if (data.predictions && data.predictions.length > 0) {
            bestWord = data.predictions[0].word;
            renderGhost(text, bestWord);
        } else {
            bestWord = "";
            renderGhost(text);
        }
    } catch (error) {
        bestWord = "";
        renderGhost(inputText.value);
        console.error("Next-word prediction failed:", error);
    }
}

// Predict automatically while the user types.
inputText.addEventListener("input", () => {
    clearTimeout(debounceTimer);

    // Invalidate any outstanding prediction request.
    requestNumber++;

    bestWord = "";

    const text = inputText.value;
    renderGhost(text);

    if (!text.trim()) {
        return;
    }

    const currentRequest = requestNumber;

    // Wait 300 ms to avoid sending a request after every keystroke.
    debounceTimer = setTimeout(() => {
        fetchSuggestion(text, currentRequest);
    }, 300);
});

// Enter accepts the predicted word.
inputText.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || !bestWord) {
        return;
    }

    event.preventDefault();

    const text = inputText.value;
    const word = bestWord;

    clearTimeout(debounceTimer);
    requestNumber++;
    bestWord = "";

    inputText.value =
        text + (/\s$/.test(text) ? "" : " ") + word + " ";

    renderGhost(inputText.value);

    // Immediately request the next suggestion.
    inputText.dispatchEvent(
        new Event("input", { bubbles: true })
    );
});

// Keep the prediction overlay aligned when scrolling.
inputText.addEventListener("scroll", () => {
    ghostText.scrollTop = inputText.scrollTop;
    ghostText.scrollLeft = inputText.scrollLeft;
});

// Show the initial empty input.
renderGhost("");
