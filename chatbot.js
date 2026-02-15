const chatStyles = `
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&display=swap');

        .chat-widget-launcher {
            position: fixed;
            bottom: 32px;
            right: 32px;
            width: 64px;
            height: 64px;
            background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
            border-radius: 50%;
            box-shadow: 0 10px 25px rgba(0, 198, 255, 0.5);
            cursor: pointer;
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            border: 1px solid rgba(255,255,255,0.2);
        }

        .chat-widget-launcher:hover {
            transform: scale(1.1) translateY(-5px);
            box-shadow: 0 15px 35px rgba(0, 198, 255, 0.6);
        }

        .chat-widget-launcher svg {
            width: 32px;
            height: 32px;
            fill: white;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));
        }

        .chat-window {
            position: fixed;
            bottom: 110px;
            right: 32px;
            width: 360px;
            height: 520px;
            background: rgba(30, 41, 59, 0.85);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            box-shadow: 0 20px 50px rgba(0,0,0,0.5);
            z-index: 9999;
            display: flex; /* Changed to flex to support animation state */
            flex-direction: column;
            overflow: hidden;
            font-family: 'Outfit', sans-serif;
            opacity: 0;
            transform: translateY(20px) scale(0.95);
            pointer-events: none; /* Prevent clicks when closed */
            transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            transform-origin: bottom right;
        }

        .chat-window.open {
            opacity: 1;
            transform: translateY(0) scale(1);
            pointer-events: all;
        }

        .chat-header {
            background: linear-gradient(135deg, rgba(0, 198, 255, 0.15), rgba(0, 114, 255, 0.15));
            padding: 24px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .chat-header h3 {
            margin: 0;
            background: linear-gradient(135deg, #fff 0%, #00c6ff 100%);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 1.15rem;
            font-weight: 700;
        }

        .chat-header span {
            font-size: 0.8rem;
            color: #34d399;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .chat-header span::before {
            content: '';
            display: block;
            width: 8px;
            height: 8px;
            background: #34d399;
            border-radius: 50%;
            box-shadow: 0 0 10px #34d399;
        }

        .chat-close {
            color: rgba(255,255,255,0.5);
            cursor: pointer;
            font-size: 1.5rem;
            line-height: 1;
            transition: color 0.2s;
        }
        
        .chat-close:hover {
            color: white;
        }

        .chat-messages {
            flex: 1;
            padding: 24px;
            overflow-y: auto;
            background: transparent;
            display: flex;
            flex-direction: column;
            gap: 20px;
            scrollbar-width: thin;
            scrollbar-color: rgba(255,255,255,0.1) transparent;
        }

        .message {
            max-width: 85%;
            padding: 14px 18px;
            border-radius: 18px;
            font-size: 0.95rem;
            line-height: 1.5;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .message.bot {
            background: rgba(255, 255, 255, 0.05);
            color: #e2e8f0;
            border-bottom-left-radius: 4px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            align-self: flex-start;
        }

        .message.user {
            background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
            color: white;
            border-bottom-right-radius: 4px;
            align-self: flex-end;
            box-shadow: 0 4px 15px rgba(0, 198, 255, 0.25);
        }

        .typing-indicator {
            display: none;
            padding: 12px 16px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 18px;
            width: fit-content;
            margin-bottom: 20px;
            margin-left: 24px;
        }

        .typing-dot {
            width: 6px;
            height: 6px;
            background: rgba(255,255,255,0.5);
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
            padding: 20px;
            background: rgba(0, 0, 0, 0.2);
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            display: flex;
            gap: 12px;
            align-items: center;
        }

        .chat-input {
            flex: 1;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 12px 18px;
            border-radius: 100px;
            color: white;
            outline: none;
            font-family: inherit;
            transition: all 0.2s;
        }
        
        .chat-input:focus {
            border-color: #00c6ff;
            background: rgba(0, 0, 0, 0.5);
            box-shadow: 0 0 0 2px rgba(0, 198, 255, 0.1);
        }

        .chat-send {
            background: transparent;
            border: none;
            color: #00c6ff;
            cursor: pointer;
            font-size: 1.2em;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            transition: all 0.2s;
        }
        
        .chat-send:hover {
            background: rgba(0, 198, 255, 0.1);
        }

        .quick-replies {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 8px;
            justify-content: flex-end;
        }

        .quick-reply-btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-muted);
            padding: 6px 14px;
            border-radius: 100px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s;
            color: #94a3b8;
        }

        .quick-reply-btn:hover {
            background: rgba(0, 198, 255, 0.1);
            border-color: #00c6ff;
            color: #00c6ff;
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
            response = "You can try the live demo! Just entered your number and I'll give you a call.";
            // Ideally trigger the call API here if you had the phone number
        } else {
            response = "That's a great question! I'm still learning, but our team can answer that in more detail. Would you like to connect with a support agent?";
        }

        addMessage(response, 'bot');
    }, 1500);
}
