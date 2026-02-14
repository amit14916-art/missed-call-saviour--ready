const chatStyles = `
    <style>
        .chat-widget-launcher {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 60px;
            height: 60px;
            background: linear-gradient(45deg, #00d2ff, #3a7bd5);
            border-radius: 50%;
            box-shadow: 0 4px 15px rgba(0, 210, 255, 0.4);
            cursor: pointer;
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: transform 0.3s ease;
        }

        .chat-widget-launcher:hover {
            transform: scale(1.1);
        }

        .chat-widget-launcher svg {
            width: 30px;
            height: 30px;
            fill: white;
        }

        .chat-window {
            position: fixed;
            bottom: 90px;
            right: 20px;
            width: 350px;
            height: 500px;
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            z-index: 9999;
            display: none;
            flex-direction: column;
            overflow: hidden;
            font-family: 'Segoe UI', sans-serif;
            opacity: 0;
            transform: translateY(20px);
            transition: all 0.3s ease;
        }

        .chat-window.open {
            display: flex;
            opacity: 1;
            transform: translateY(0);
        }

        .chat-header {
            background: linear-gradient(45deg, #0f172a, #1e293b);
            padding: 20px;
            border-bottom: 1px solid #334155;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .chat-header h3 {
            margin: 0;
            color: #00d2ff;
            font-size: 1.1em;
        }

        .chat-header span {
            font-size: 0.8em;
            color: #34d399;
        }

        .chat-close {
            color: #94a3b8;
            cursor: pointer;
            font-size: 1.2em;
        }

        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #0b0f19;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .message {
            max-width: 80%;
            padding: 10px 15px;
            border-radius: 15px;
            font-size: 0.95em;
            line-height: 1.4;
        }

        .message.bot {
            background: #1e293b;
            color: #e2e8f0;
            border-bottom-left-radius: 2px;
            align-self: flex-start;
        }

        .message.user {
            background: linear-gradient(45deg, #00d2ff, #3a7bd5);
            color: white;
            border-bottom-right-radius: 2px;
            align-self: flex-end;
        }

        .typing-indicator {
            display: none;
            padding: 10px;
            background: #1e293b;
            border-radius: 15px;
            width: fit-content;
            margin-bottom: 10px;
        }

        .typing-dot {
            width: 6px;
            height: 6px;
            background: #94a3b8;
            border-radius: 50%;
            display: inline-block;
            margin: 0 2px;
            animation: typing 1.4s infinite ease-in-out;
        }

        .typing-dot:nth-child(1) { animation-delay: 0s; }
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }

        @keyframes typing {
            0%, 100% { transform: scale(0.6); opacity: 0.6; }
            50% { transform: scale(1); opacity: 1; }
        }

        .chat-input-area {
            padding: 15px;
            background: #1e293b;
            border-top: 1px solid #334155;
            display: flex;
            gap: 10px;
        }

        .chat-input {
            flex: 1;
            background: #0b0f19;
            border: 1px solid #334155;
            padding: 10px;
            border-radius: 20px;
            color: white;
            outline: none;
        }
        
        .chat-input:focus {
            border-color: #00d2ff;
        }

        .chat-send {
            background: transparent;
            border: none;
            color: #00d2ff;
            cursor: pointer;
            font-size: 1.2em;
        }

        .quick-replies {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 5px;
        }

        .quick-reply-btn {
            background: rgba(0, 210, 255, 0.1);
            border: 1px solid #00d2ff;
            color: #00d2ff;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.8em;
            cursor: pointer;
            transition: all 0.2s;
        }

        .quick-reply-btn:hover {
            background: #00d2ff;
            color: white;
        }
    </style>
`;

const chatStructure = `
    <div class="chat-widget-launcher" onclick="toggleChat()">
        <svg viewBox="0 0 24 24">
            <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z"/>
        </svg>
    </div>

    <div class="chat-window" id="chatWindow">
        <div class="chat-header">
            <div>
                <h3>Assistant</h3>
                <span>● Online</span>
            </div>
            <div class="chat-close" onclick="toggleChat()">×</div>
        </div>
        <div class="chat-messages" id="chatMessages">
            <div class="message bot">
                Hello! I'm your Missed Call Saviour assistant. How can I help you recover lost sales today?
            </div>
            <div class="quick-replies">
                <button class="quick-reply-btn" onclick="sendQuickReply('Pricing')">Pricing</button>
                <button class="quick-reply-btn" onclick="sendQuickReply('How does it work?')">How it works?</button>
                <button class="quick-reply-btn" onclick="sendQuickReply('Support')">Human Support</button>
            </div>
        </div>
        <div class="typing-indicator" id="typingIndicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
        <form class="chat-input-area" onsubmit="handleUserSubmit(event)">
            <input type="text" class="chat-input" id="chatInput" placeholder="Type a message...">
            <button type="submit" class="chat-send">➤</button>
        </form>
    </div>
`;

// Inject Styles and HTML
document.head.insertAdjacentHTML('beforeend', chatStyles);
document.body.insertAdjacentHTML('beforeend', chatStructure);

function toggleChat() {
    const window = document.getElementById('chatWindow');
    window.classList.toggle('open');
}

function sendQuickReply(text) {
    addMessage(text, 'user');
    processBotResponse(text);
}

function handleUserSubmit(e) {
    e.preventDefault();
    const input = document.getElementById('chatInput');
    const text = input.value.trim();
    if (text) {
        addMessage(text, 'user');
        input.value = '';
        processBotResponse(text);
    }
}

function addMessage(text, sender) {
    const messagesDiv = document.getElementById('chatMessages');
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${sender}`;
    msgDiv.innerText = text;
    messagesDiv.appendChild(msgDiv);

    // Auto scroll to bottom
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function showTyping() {
    const indicator = document.getElementById('typingIndicator');
    indicator.style.display = 'block';
    const messagesDiv = document.getElementById('chatMessages');
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function hideTyping() {
    document.getElementById('typingIndicator').style.display = 'none';
}

function processBotResponse(userText) {
    showTyping();

    // Simulate AI delay
    setTimeout(() => {
        hideTyping();
        let response = "";
        const lowerText = userText.toLowerCase();

        if (lowerText.includes('pricing') || lowerText.includes('cost')) {
            response = "We have plans starting at $29/mo (Solo) up to $99/mo (Agency). You can see full details on our Pricing page.";
        } else if (lowerText.includes('work') || lowerText.includes('feature')) {
            response = "I automatically text back anyone who calls you when you're busy. I can also use AI to answer their questions and book appointments!";
        } else if (lowerText.includes('support') || lowerText.includes('human') || lowerText.includes('help')) {
            response = "Our team is available 24/7. You can email us at support@missedcallsaviour.com or call +1-555-0199.";
        } else if (lowerText.includes('demo')) {
            response = "You can try the live demo on our homepage! Just enter your number and I'll send you a simulation text.";
        } else {
            response = "That's a great question! I'm still learning, but our team can answer that in more detail. Would you like to connect with a support agent?";
        }

        addMessage(response, 'bot');
    }, 1500);
}
