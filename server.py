from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

app = FastAPI()
pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
pipe2 = pipeline("summarization", model="Falconsai/text_summarization")

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
    out = pipe2(
        req.text
    )
    # print(out)
    return {"generated_text": out[0]["summary_text"]}

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
            
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const text = document.getElementById('textInput').value;
                const endpoint = toggle.checked ? '/summarize' : '/generate';
                
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
                    resultText.textContent = data.generated_text;
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
