from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import uvicorn
import torch

app = FastAPI()
pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
pipe2 = pipeline("summarization", model="Falconsai/text_summarization")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Load summarization model and tokenizer (T5 model)
sum_tokenizer = AutoTokenizer.from_pretrained("Falconsai/text_summarization")
sum_model = AutoModelForSeq2SeqLM.from_pretrained("Falconsai/text_summarization")

class GenRequest(BaseModel):
    text: str
    max_new_tokens: int = 150
    do_sample: bool = False  # set True if you want to use temperature/top_p, etc.

@app.post("/generate")
def generate(req: GenRequest):
    out = pipe(
        req.text,
        max_new_tokens=req.max_new_tokens,
        do_sample=req.do_sample,
        truncation=True,
        return_full_text=False,
    )
    return {"generated_text": out[0]["generated_text"]}

@app.post("/summarize")
def summarize(req: GenRequest):
    inputs = sum_tokenizer(req.text, return_tensors="pt", truncation=True, max_length=512)
    
    # Generate summary tokens (T5 is encoder-decoder, so only generated tokens are returned)
    with torch.no_grad():
        generated_ids = sum_model.generate(
            inputs["input_ids"],
            max_new_tokens=req.max_new_tokens,
            do_sample=req.do_sample,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=sum_tokenizer.pad_token_id
        )
    
    # Get generated token ids
    all_token_ids = generated_ids.sequences[0]
    
    # Get logits for generated tokens
    output = []
    
    # Process generated tokens (T5 doesn't output input tokens in generation)
    if hasattr(generated_ids, 'scores') and generated_ids.scores:
        for i, scores in enumerate(generated_ids.scores):
            token_id = all_token_ids[i]
            logit_scores = scores[0]
            _, top_indices = torch.topk(logit_scores, k=5, dim=-1)
            logit_list = sum_tokenizer.decode(top_indices)
            decoded = sum_tokenizer.decode([token_id])
            output.append((decoded, logit_list))

    return {"output": output}

@app.post("/generate_tokens")
def gen_tokens(req: GenRequest):
    inputs = tokenizer(req.text, return_tensors="pt")
    input_length = inputs["input_ids"].shape[1]
    
    # Generate new tokens
    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=req.max_new_tokens,
            do_sample=req.do_sample,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Get all token ids (input + generated)
    all_token_ids = generated_ids.sequences[0]
    
    # Get logits for generated tokens
    output = []
    
    # Process input tokens
    with torch.no_grad():
        input_output = model(**inputs)
        input_logits = input_output.logits
    
    for i, token_id in enumerate(inputs["input_ids"][0]):
        logit_scores = input_logits[0, i, :]
        _, top_indices = torch.topk(logit_scores, k=5, dim=-1)
        logit_list = tokenizer.decode(top_indices)
        decoded = tokenizer.decode([token_id])
        output.append((decoded, logit_list))
    
    # Process generated tokens
    if hasattr(generated_ids, 'scores') and generated_ids.scores:
        for i, scores in enumerate(generated_ids.scores):
            token_id = all_token_ids[input_length + i]
            logit_scores = scores[0]
            _, top_indices = torch.topk(logit_scores, k=5, dim=-1)
            logit_list = tokenizer.decode(top_indices)
            decoded = tokenizer.decode([token_id])
            output.append((decoded, logit_list))

    return {"output": output}


@app.get("/", response_class=HTMLResponse)
def index():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Text Generation & Summarization</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .toggle-container {
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 20px 0;
                gap: 10px;
            }
            .toggle {
                position: relative;
                display: inline-block;
                width: 60px;
                height: 34px;
            }
            .toggle input {
                opacity: 0;
                width: 0;
                height: 0;
            }
            .slider {
                position: absolute;
                cursor: pointer;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: #2196F3;
                transition: .4s;
                border-radius: 34px;
            }
            .slider:before {
                position: absolute;
                content: "";
                height: 26px;
                width: 26px;
                left: 4px;
                bottom: 4px;
                background-color: white;
                transition: .4s;
                border-radius: 50%;
            }
            input:checked + .slider {
                background-color: #4CAF50;
            }
            input:checked + .slider:before {
                transform: translateX(26px);
            }
            textarea {
                width: 100%;
                min-height: 150px;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
                box-sizing: border-box;
                resize: vertical;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
                color: #555;
            }
            button {
                width: 100%;
                padding: 12px;
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 16px;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            button:hover {
                background-color: #0b7dda;
            }
            button:disabled {
                background-color: #ccc;
                cursor: not-allowed;
            }
            .result {
                margin-top: 20px;
                padding: 15px;
                background-color: #f9f9f9;
                border-left: 4px solid #2196F3;
                border-radius: 4px;
                display: none;
            }
            .result h3 {
                margin-top: 0;
                color: #333;
            }
            .result-text {
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            .token-container {
                display: inline-block;
                line-height: 2;
            }
            .token {
                display: inline-block;
                padding: 2px 4px;
                margin: 2px;
                border-radius: 3px;
                cursor: pointer;
                transition: background-color 0.2s;
            }
            .token:hover {
                background-color: #ffeb3b;
            }
            .logit-popup {
                position: fixed;
                background-color: white;
                border: 2px solid #2196F3;
                border-radius: 4px;
                padding: 10px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                z-index: 1000;
                display: none;
                max-width: 300px;
            }
            .logit-popup.show {
                display: block;
            }
            .logit-popup h4 {
                margin: 0 0 8px 0;
                color: #2196F3;
                font-size: 14px;
            }
            .logit-popup ul {
                margin: 0;
                padding-left: 20px;
                font-size: 13px;
            }
            .logit-popup li {
                margin: 4px 0;
            }
            .loading {
                text-align: center;
                color: #666;
            }
            .endpoint-label {
                font-weight: bold;
                font-size: 16px;
            }
            .generate-label {
                color: #2196F3;
            }
            .summarize-label {
                color: #4CAF50;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Text Generation & Summarization</h1>
            
            <div class="toggle-container">
                <span class="endpoint-label generate-label" id="generateLabel">Generate</span>
                <label class="toggle">
                    <input type="checkbox" id="endpointToggle">
                    <span class="slider"></span>
                </label>
                <span class="endpoint-label summarize-label" id="summarizeLabel">Summarize</span>
            </div>
            
            <form id="textForm">
                <div class="form-group">
                    <label for="textInput">Enter your text:</label>
                    <textarea id="textInput" placeholder="Type or paste your text here..." required></textarea>
                </div>
                
                <button type="submit" id="submitBtn">Submit</button>
            </form>
            
            <div class="result" id="result">
                <h3>Result:</h3>
                <div class="result-text" id="resultText"></div>
            </div>
            
            <div class="logit-popup" id="logitPopup">
                <h4>Next Predicted Tokens:</h4>
                <ul id="logitList"></ul>
            </div>
        </div>
        
        <script>
            const form = document.getElementById('textForm');
            const toggle = document.getElementById('endpointToggle');
            const submitBtn = document.getElementById('submitBtn');
            const resultDiv = document.getElementById('result');
            const resultText = document.getElementById('resultText');
            const generateLabel = document.getElementById('generateLabel');
            const summarizeLabel = document.getElementById('summarizeLabel');
            
            // Update label styles based on toggle
            function updateLabels() {
                if (toggle.checked) {
                    generateLabel.style.opacity = '0.4';
                    summarizeLabel.style.opacity = '1';
                } else {
                    generateLabel.style.opacity = '1';
                    summarizeLabel.style.opacity = '0.4';
                }
            }
            
            toggle.addEventListener('change', updateLabels);
            updateLabels();
            
            const logitPopup = document.getElementById('logitPopup');
            const logitList = document.getElementById('logitList');
            
            // Handle token hover events
            document.addEventListener('mouseover', (e) => {
                if (e.target.classList.contains('token')) {
                    const logits = e.target.dataset.logits;
                    
                    // Parse and display logits
                    logitList.innerHTML = '';
                    const tokens = logits.split(' ').filter(t => t.length > 0);
                    tokens.forEach(token => {
                        const li = document.createElement('li');
                        li.textContent = token;
                        logitList.appendChild(li);
                    });
                    
                    // Position popup near mouse
                    logitPopup.style.left = (e.pageX + 15) + 'px';
                    logitPopup.style.top = (e.pageY + 15) + 'px';
                    logitPopup.classList.add('show');
                }
            });
            
            document.addEventListener('mouseout', (e) => {
                if (e.target.classList.contains('token')) {
                    logitPopup.classList.remove('show');
                }
            });
            
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const text = document.getElementById('textInput').value;
                const endpoint = toggle.checked ? '/summarize' : '/generate_tokens';
                
                // Show loading state
                submitBtn.disabled = true;
                submitBtn.textContent = 'Processing...';
                resultDiv.style.display = 'none';
                
                try {
                    const response = await fetch(endpoint, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            text: text,
                            max_new_tokens: 150,
                            do_sample: false
                        })
                    });
                    
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    
                    const data = await response.json();
                    
                    if (data.output) {
                        // Display tokens with hover functionality
                        resultText.innerHTML = '';
                        const tokenContainer = document.createElement('div');
                        tokenContainer.className = 'token-container';
                        
                        data.output.forEach((item, index) => {
                            const tokenSpan = document.createElement('span');
                            tokenSpan.className = 'token';
                            tokenSpan.textContent = item[0];
                            tokenSpan.dataset.logits = item[1];
                            tokenSpan.dataset.index = index;
                            tokenContainer.appendChild(tokenSpan);
                        });
                        
                        resultText.appendChild(tokenContainer);
                    } else {
                        resultText.textContent = data.generated_text;
                    }
                    
                    resultDiv.style.display = 'block';
                } catch (error) {
                    resultText.textContent = 'Error: ' + error.message;
                    resultDiv.style.display = 'block';
                } finally {
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Submit';
                }
            });
        </script>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)