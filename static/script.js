async function sendMessage() {

    const input = document.getElementById("user-input");
    const sendButton = document.getElementById("send-button");
    const message = input.value.trim();

    if (message === "") {
        return;
    }

    removeWelcome();

    addMessage(message, "user");

    input.value = "";

    input.disabled = true;
    sendButton.disabled = true;

    try {

        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message
            })
        });

        if (!response.ok) {
            throw new Error("Server error");
        }

        const data = await response.json();

        addMessage(
            data.response || "Sorry, I could not understand that.",
            "bot"
        );

        if (data.sentiment) {
            addSentiment(data.sentiment);
        }

    } catch (error) {

        console.error("Chat error:", error);

        addMessage(
            "Sorry, something went wrong. Please try again.",
            "bot"
        );

    } finally {

        input.disabled = false;
        sendButton.disabled = false;
        input.focus();
    }
}


/* =========================
   ADD MESSAGE
========================= */

function addMessage(message, sender, save = true) {

    const chatBox = document.getElementById("chat-box");

    const messageDiv = document.createElement("div");

    messageDiv.classList.add(
        "message",
        sender + "-message"
    );

    const messageSpan = document.createElement("span");

    messageSpan.textContent = message;

    messageDiv.appendChild(messageSpan);

    chatBox.appendChild(messageDiv);

    chatBox.scrollTop = chatBox.scrollHeight;


    if (save) {

        let history = JSON.parse(
            localStorage.getItem("smartchat_history")
        ) || [];

        history.push({
            message: message,
            sender: sender
        });

        localStorage.setItem(
            "smartchat_history",
            JSON.stringify(history)
        );
    }

    return messageDiv;
}


/* =========================
   SENTIMENT
========================= */

function addSentiment(sentiment) {

    const chatBox = document.getElementById("chat-box");

    const sentimentDiv = document.createElement("div");

    sentimentDiv.classList.add("sentiment");

    let emoji = "😐";

    if (sentiment === "Positive") {
        emoji = "😊";
    } else if (sentiment === "Negative") {
        emoji = "😟";
    }

    sentimentDiv.textContent =
        `Sentiment: ${sentiment} ${emoji}`;

    chatBox.appendChild(sentimentDiv);

    chatBox.scrollTop = chatBox.scrollHeight;
}


/* =========================
   REMOVE WELCOME
========================= */

function removeWelcome() {

    const welcome = document.querySelector(".welcome");

    if (welcome) {
        welcome.remove();
    }
}


/* =========================
   QUICK PROMPT
========================= */

function usePrompt(prompt) {

    const input = document.getElementById("user-input");

    input.value = prompt;

    input.focus();

    sendMessage();
}


/* =========================
   CLEAR CHAT
========================= */

async function clearChat() {

    try {

        await fetch("/clear", {
            method: "POST"
        });

    } catch (error) {

        console.error(
            "Could not clear backend memory:",
            error
        );
    }


    localStorage.removeItem("smartchat_history");


    const chatBox = document.getElementById("chat-box");

    chatBox.innerHTML = `

        <div class="welcome">

            <div class="welcome-icon">
                ✦
            </div>

            <h2>
                How can I help you today?
            </h2>

            <p>
                Ask me about AI, programming,
                machine learning, data science
                and more.
            </p>

            <div class="suggestions">

                <button onclick="usePrompt('What is AI?')">
                    <strong>What is AI?</strong>
                    <small>Learn the basics of AI</small>
                </button>

                <button onclick="usePrompt('Explain machine learning')">
                    <strong>Machine Learning</strong>
                    <small>Understand ML simply</small>
                </button>

                <button onclick="usePrompt('What is Python?')">
                    <strong>Python</strong>
                    <small>Learn Python programming</small>
                </button>

                <button onclick="usePrompt('What is Data Science?')">
                    <strong>Data Science</strong>
                    <small>Explore data and analytics</small>
                </button>

            </div>

        </div>
    `;

    document.getElementById("user-input").focus();
}


/* =========================
   LOAD SERVER HISTORY
========================= */

async function loadServerHistory() {

    try {

        const response = await fetch("/history");

        if (!response.ok) {
            throw new Error("Could not load history");
        }

        const data = await response.json();

        if (!data.history || data.history.length === 0) {
            return;
        }

        removeWelcome();

        data.history.forEach(item => {

            if (item.user) {

                addMessage(
                    item.user,
                    "user",
                    false
                );
            }

            if (item.bot) {

                addMessage(
                    item.bot,
                    "bot",
                    false
                );
            }

        });

        const chatBox = document.getElementById("chat-box");

        chatBox.scrollTop = chatBox.scrollHeight;

    } catch (error) {

        console.error(
            "Could not load server history:",
            error
        );
    }
}


/* =========================
   ENTER KEY
========================= */

document
    .getElementById("user-input")
    .addEventListener("keypress", function(event) {

        if (event.key === "Enter") {

            event.preventDefault();

            sendMessage();
        }
    });


/* =========================
   LOAD HISTORY
========================= */

window.addEventListener("load", loadServerHistory);