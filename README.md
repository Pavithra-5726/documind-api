# 📄 AI-Powered Document Analysis & Extraction System

> An intelligent document analysis platform that extracts key information, sentiment, entities, and structured data from PDFs, DOCX, and image files using AI.

![Status](https://img.shields.io/badge/Status-Live-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.5-009688)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🌐 Live URL

🔗 **Frontend:** `https://your-frontend-url.vercel.app`  
🔗 **Backend API:** `https://your-backend-url.onrender.com`  
🔗 **API Docs:** `https://your-backend-url.onrender.com/docs`

---

## 🎥 Video Demo

📹 **Demo Link:** `https://your-youtube-or-drive-link`

---

## 📌 Project Overview

The AI-Powered Document Analysis & Extraction System is a full-stack web application that allows users to upload documents (PDF, DOCX, images) and instantly receive:

- A structured **summary** of the document
- **Key information** extraction (title, date, author, recipient, subject)
- **Contact details** (emails, phones, addresses)
- **Sentiment analysis** (positive / negative / neutral / mixed)
- **Named entity recognition**
- **Monetary amounts** and **date extraction**
- **Confidence score** for the analysis

---

## ✨ Features

- 📁 Upload PDF, DOCX, JPG, PNG, WEBP files
- 🤖 AI-powered analysis using LLaMA 3.3 70B via Groq API
- 📊 Structured data extraction with confidence scoring
- 💬 Sentiment detection
- 🌐 REST API with full Swagger documentation
- 🎨 Dark-themed, responsive frontend UI
- ⚡ Fast response times with async processing

---

## 🏗️ Architecture Overview

```
┌─────────────────┐         ┌─────────────────────┐        ┌──────────────┐
│                 │  HTTP   │                     │  API   │              │
│    Frontend     │────────▶│   FastAPI Backend   │───────▶│  Groq AI     │
│  (HTML/CSS/JS)  │◀────────│   (Python)          │◀───────│  (LLaMA 3.3) │
│                 │  JSON   │                     │  JSON  │              │
└─────────────────┘         └─────────────────────┘        └──────────────┘
       │                             │
       │                             │
  Vercel/Netlify                Render/Railway
  (Frontend Host)               (Backend Host)
```

**Flow:**
1. User uploads a document via the frontend
2. Frontend sends the file to the FastAPI backend via `/analyze` endpoint
3. Backend extracts text from the document (PDF → pdfplumber, DOCX → python-docx)
4. Extracted text is sent to Groq AI (LLaMA 3.3 70B) with a structured prompt
5. AI returns structured JSON with all extracted information
6. Backend sends the result back to the frontend
7. Frontend displays the results in a clean, readable UI

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Backend | Python, FastAPI |
| AI Model | LLaMA 3.3 70B Versatile (via Groq) |
| PDF Extraction | pdfplumber |
| DOCX Extraction | python-docx |
| HTTP Client | httpx (async) |
| Server | Uvicorn (ASGI) |
| Frontend Hosting | Vercel / Netlify |
| Backend Hosting | Render / Railway |

---

## 🤖 AI Tools Used

> **Disclosure as required by Guvi Hackathon AI Tool Policy:**

| Tool | Purpose |
|---|---|
| **Groq API (LLaMA 3.3 70B Versatile)** | Core document analysis, entity extraction, sentiment analysis, summarization |
| **Claude (Anthropic)** | Development assistance, code generation, debugging, README writing |

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.10+
- A Groq API key (free at [console.groq.com](https://console.groq.com))

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Backend Setup
```bash
cd backend
pip install -r requirements.txt
```

Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
```

Run the backend:
```bash
uvicorn main:app --reload --port 8000
```

API will be available at: `http://localhost:8000`  
Swagger docs at: `http://localhost:8000/docs`

### 3. Frontend Setup
```bash
cd frontend
```

Update the API URL in your frontend JS file:
```javascript
const API_URL = "http://localhost:8000"; // for local dev
```

Open `index.html` in your browser or use Live Server.

### 4. Environment Variables

| Variable | Description | Required |
|---|---|---|
| `GROQ_API_KEY` | Your Groq API key from console.groq.com | ✅ Yes |

---

## 📁 Project Structure

```
├── backend/
│   ├── main.py              # FastAPI application & all endpoints
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── index.html           # Main UI
│   ├── style.css            # Styling
│   └── app.js               # Frontend logic
└── README.md
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| POST | `/analyze` | Analyze uploaded document |

### Example Response (`/analyze`)
```json
{
  "success": true,
  "fileName": "report.pdf",
  "document_type": "Report",
  "summary": "This document discusses...",
  "sentiment": "neutral",
  "confidence_score": 0.95,
  "key_information": {
    "title": "AgriSense Report",
    "date": "2024-01-01",
    "author_or_sender": "John Doe"
  },
  "extracted_data": {
    "entities": ["AgriSense", "AI"],
    "dates": ["2024-01-01"],
    "amounts": [],
    "contact_info": {
      "emails": [],
      "phones": [],
      "addresses": []
    },
    "key_points": ["AI-driven agriculture", "Real-time data"]
  }
}
```

---

## ⚠️ Known Limitations

- Image files (JPG, PNG) are processed by filename only — OCR not yet implemented
- Very large PDFs (50+ pages) may be truncated to first 8000 characters
- Analysis quality depends on document text clarity
- Groq free tier has rate limits (limited requests per minute)
- No user authentication or file history storage

---

## 🚀 Deployment

### Backend (Render)
1. Push code to GitHub
2. Connect repo to [render.com](https://render.com)
3. Set environment variable: `GROQ_API_KEY`
4. Deploy as a Web Service with command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Frontend (Vercel/Netlify)
1. Push frontend folder to GitHub
2. Connect to Vercel or Netlify
3. Deploy — no environment variables needed

---

## 👨‍💻 Author

**Your Name**  
Guvi Hackathon 2025  
GitHub: [PAVITHRA S](https://github.com/pavithra-5726)

---

## 📜 License

MIT License — feel free to use and modify.
