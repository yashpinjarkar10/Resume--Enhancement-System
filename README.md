# 🚀 Intelligent Resume Enhancement System

An AI-powered web application that analyzes, scores, and enhances resumes for optimal ATS (Applicant Tracking System) compatibility using Google Gemini AI.

## ✨ Features

- **🎯 ATS Scoring**: Comprehensive 100-point analysis system
- **🤖 AI Enhancement**: Google Gemini AI-powered resume optimization  
- **📄 Multi-format Support**: PDF files and images (OCR enabled)
- **🎨 Professional PDF Output**: Beautifully formatted enhanced resumes
- **⚡ Real-time Analysis**: Instant feedback and recommendations
- **🖱️ Modern Interface**: Drag-and-drop upload with progress tracking

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python 3.8+
- **AI**: LangChain + Google Gemini 2.5 Flash
- **PDF Processing**: PyPDF2, ReportLab  
- **OCR**: Unstructured (image text extraction)
- **Frontend**: Modern HTML/CSS/JavaScript

## 🚀 Quick Start

1. **Clone & Setup**
   ```bash
   git clone <repository-url>
   cd Whats_app
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   # Create .env file
   GOOGLE_API_KEY=your_google_gemini_api_key
   ```

3. **Run Application**
   ```bash
   uvicorn main:app --reload
   # Open: http://localhost:8000
   ```

## 📊 ATS Scoring System

| Category | Points | Criteria |
|----------|--------|----------|
| **Content** | 25 | Quantified achievements, action verbs, grammar |
| **Sections** | 25 | Essential sections, contact information |
| **ATS Essentials** | 25 | Email, phone, professional links |
| **Tailoring** | 25 | Technical skills, keywords, optimization |

## 🔗 API Endpoints

- `GET /` → Web interface
- `POST /analyze-resume` → ATS score analysis
- `POST /enhance-resume` → AI enhancement + PDF
- `GET /download/{filename}` → Download enhanced resume

## 📁 Project Structure

```
├── main.py              # FastAPI application & core logic
├── index.html           # Professional web interface
├── requirements.txt     # Python dependencies
├── .env                # Environment variables
└── README.md           # Project documentation
```

## 🌐 Deployment Options

**Hugging Face Spaces**
Live Link: https://yashpinjarkar10-resume-enhancement-system.hf.space

## 🧪 Testing

```bash
# Test API endpoints
python test_api.py

# Health check
curl http://localhost:8000/health

# API documentation
http://localhost:8000/docs
```

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Make changes
4. Submit pull request

## 📄 License

MIT License - Open source and free to use

---


**✨ Boost your career with AI-enhanced resumes!**
