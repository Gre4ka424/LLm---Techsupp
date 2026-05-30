import os
import uvicorn
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
from contextlib import asynccontextmanager

# Define model path
model_path = "./llama3_final_weights/checkpoint-4000"
model = None
tokenizer = None
sessions = {}

# Context buffer for conversation history
class ContextBuffer:
    def __init__(self, max_history=3):
        self.history = []
        self.max_history = max_history

    def add_interaction(self, user_msg, bot_msg):
        self.history.append({"user": user_msg, "bot": bot_msg})
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def build_prompt(self, new_user_msg):
        prompt = ""
        for turn in self.history:
            prompt += f"### Customer:\n{turn['user']}\n\n### Agent:\n{turn['bot']}\n\n"
        prompt += f"### Customer:\n{new_user_msg}\n\n### Agent:\n"
        return prompt

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str

# HTML, CSS and JS for the Web UI
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Customer Support</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f0f2f5; margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; height: 100vh; }
        .chat-container { width: 400px; height: 600px; background: white; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); display: flex; flex-direction: column; overflow: hidden; }
        .chat-header { background: #0078ff; color: white; padding: 20px; text-align: center; font-size: 1.2em; font-weight: bold; }
        .chat-box { flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 15px; background: #e5ddd5; }
        .message { max-width: 80%; padding: 10px 15px; border-radius: 20px; font-size: 0.95em; line-height: 1.4; word-wrap: break-word; }
        .user-message { background: #dcf8c6; align-self: flex-end; border-bottom-right-radius: 5px; }
        .bot-message { background: white; align-self: flex-start; border-bottom-left-radius: 5px; box-shadow: 0 1px 2px rgba(0,0,0,0.1); }
        .input-container { display: flex; padding: 15px; background: #f0f0f0; border-top: 1px solid #ccc; }
        .input-container input { flex: 1; padding: 12px; border: none; border-radius: 25px; outline: none; font-size: 1em; padding-left: 15px; }
        .input-container button { margin-left: 10px; padding: 10px 20px; background: #0078ff; color: white; border: none; border-radius: 25px; cursor: pointer; font-weight: bold; transition: background 0.3s; }
        .input-container button:hover { background: #005bb5; }
        .loading { font-style: italic; color: gray; font-size: 0.85em; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">AI Support Agent</div>
        <div class="chat-box" id="chat-box">
            <div class="message bot-message">Hello! I am the AI Support Agent. How can I help you today?</div>
        </div>
        <div class="input-container">
            <input type="text" id="user-input" placeholder="Type your message..." onkeypress="handleKeyPress(event)">
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        const sessionId = "user_" + Math.floor(Math.random() * 10000);
        const chatBox = document.getElementById('chat-box');
        const userInput = document.getElementById('user-input');

        function appendMessage(text, sender) {
            const msgDiv = document.createElement('div');
            msgDiv.classList.add('message');
            msgDiv.classList.add(sender === 'user' ? 'user-message' : 'bot-message');
            msgDiv.innerText = text;
            chatBox.appendChild(msgDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        async function sendMessage() {
            const text = userInput.value.trim();
            if (!text) return;

            appendMessage(text, 'user');
            userInput.value = '';

            const loadingDiv = document.createElement('div');
            loadingDiv.classList.add('message', 'bot-message', 'loading');
            loadingDiv.innerText = "Agent is typing...";
            chatBox.appendChild(loadingDiv);
            chatBox.scrollTop = chatBox.scrollHeight;

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sessionId, message: text })
                });
                const data = await response.json();
                
                chatBox.removeChild(loadingDiv);
                appendMessage(data.response, 'bot');
            } catch (error) {
                chatBox.removeChild(loadingDiv);
                appendMessage("Error connecting to the server.", 'bot');
            }
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
    </script>
</body>
</html>
"""

# Lifespan manager to handle startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    print("[INFO] Loading model and tokenizer for API...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Configure 4-bit quantization to fit into 12GB VRAM
    quant_config = BitsAndBytesConfig(load_in_4bit=True)
    
    # Use AutoPeftModel to automatically merge base LLaMA-3 with your LoRA adapter
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        quantization_config=quant_config,
        device_map={"": 0}
    )
    print("[INFO] ===============================================")
    print("[INFO] FULL UI IS READY! Go to: http://localhost:8080/")
    print("[INFO] ===============================================")
    yield
    print("[INFO] Shutting down API...")

# Initialize FastAPI application
app = FastAPI(title="Customer Support LLM API", lifespan=lifespan)

# Endpoint to serve the HTML UI
@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return HTMLResponse(content=html_content)

# Endpoint to process chat messages
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    session_id = request.session_id
    user_message = request.message

    if session_id not in sessions:
        sessions[session_id] = ContextBuffer(max_history=3)
    
    memory = sessions[session_id]
    prompt = memory.build_prompt(user_message)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            max_length=None,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    bot_response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    memory.add_interaction(user_message, bot_response)
    
    return ChatResponse(response=bot_response)

if __name__ == "__main__":
    # Start the server on port 8080
    uvicorn.run("main:app", host="0.0.0.0", port=8080)