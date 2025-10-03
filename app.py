import os
import io
import tempfile
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import re
from PIL import Image
import easyocr
from PIL import Image
import io
import numpy as np
import base64
import json
from jinja2 import Template
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from datetime import datetime

# Load environment variables
load_dotenv()

app = FastAPI(title="Intelligent Resume Enhancement System")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is required")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key,
    temperature=0
)

def extract_text_from_pdf(pdf_file: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting text from PDF: {str(e)}")


def extract_text_from_image(image_file: bytes) -> str:
    """Extract text from image file using Google Gemini Vision"""
    try:
         
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_file))
        
        # Convert to numpy array (EasyOCR expects this format)
        image_array = np.array(image)
        
        # Initialize EasyOCR reader (downloads model if needed)
        reader = easyocr.Reader(['en'])  # Add other languages as needed, e.g., ['en', 'fr']
        
        # Extract text
        results = reader.readtext(image_array)
        
        # Join extracted texts
        extracted_text = ' '.join([result[1] for result in results])
        
        return extracted_text.strip()

    except Exception as e:
        return f"""
        IMAGE PROCESSING NOTE: Unable to extract text from image automatically.
        
        To analyze your resume from an image, please:
        1. Convert the image to a PDF file, or
        2. Type/copy the resume content manually
        
        Error details: {str(e)}
        
        Alternatively, you can still proceed with a basic analysis by providing key information:
        - Name and contact information
        - Work experience with dates and achievements
        - Education background
        - Skills and certifications
        """

def get_ats_score(resume_text: str) -> dict:
    """Calculate ATS score based on various parameters"""
    
    # Content scoring (25 points)
    content_score = 0
    
    # Check for quantified impact (numbers, percentages)
    numbers = re.findall(r'\d+%|\d+\+|\d+k|\d+,\d+|\$\d+', resume_text.lower())
    if len(numbers) >= 3:
        content_score += 8
    elif len(numbers) >= 1:
        content_score += 5
    
    # Check for repetition (negative scoring)
    words = resume_text.lower().split()
    word_freq = {}
    for word in words:
        if len(word) > 4:  # Only check longer words
            word_freq[word] = word_freq.get(word, 0) + 1
    
    repetitive_words = sum(1 for count in word_freq.values() if count > 3)
    if repetitive_words < 3:
        content_score += 5
    
    # Basic spelling and grammar check (simple heuristics)
    grammar_indicators = ['achieved', 'managed', 'led', 'developed', 'implemented', 'created']
    grammar_score = min(5, len([word for word in grammar_indicators if word in resume_text.lower()]))
    content_score += grammar_score
    
    # Ensure action verbs are present
    action_verbs = ['managed', 'led', 'developed', 'created', 'implemented', 'achieved', 'improved', 'increased']
    action_verb_count = len([verb for verb in action_verbs if verb in resume_text.lower()])
    if action_verb_count >= 5:
        content_score += 7
    elif action_verb_count >= 3:
        content_score += 4
    
    # Section scoring (25 points)
    section_score = 0
    
    # Essential sections
    essential_sections = ['experience', 'education', 'skills', 'summary', 'objective']
    found_sections = [section for section in essential_sections if section in resume_text.lower()]
    section_score += min(15, len(found_sections) * 3)
    
    # Contact information
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    
    if re.search(email_pattern, resume_text):
        section_score += 5
    if re.search(phone_pattern, resume_text):
        section_score += 5
    
    # ATS Essentials (25 points)
    ats_score = 0
    
    # Email address presence
    if re.search(email_pattern, resume_text):
        ats_score += 10
    
    # Enhanced hyperlink detection - multiple patterns for better recognition
    # Full URLs with http/https
    full_url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    
    # Domain patterns without protocol (common in resumes)
    domain_patterns = [
        r'(?:www\.)?linkedin\.com/in/[\w\-]+',  # LinkedIn profiles
        r'(?:www\.)?github\.com/[\w\-]+',       # GitHub profiles
        r'(?:www\.)?gitlab\.com/[\w\-]+',       # GitLab profiles
        r'(?:www\.)?bitbucket\.org/[\w\-]+',    # Bitbucket profiles
        r'(?:www\.)?stackoverflow\.com/users/[\w\-/]+',  # Stack Overflow
        r'(?:www\.)?medium\.com/@?[\w\-.]+'    # Medium profiles
        r'(?:www\.)?twitter\.com/[\w\-]+',      # Twitter profiles
        r'(?:www\.)?facebook\.com/[\w\-.]+'    # Facebook profiles
        r'(?:www\.)?instagram\.com/[\w\-.]+'   # Instagram profiles
        r'(?:www\.)?behance\.net/[\w\-]+',      # Behance portfolios
        r'(?:www\.)?dribbble\.com/[\w\-]+',     # Dribbble portfolios
        r'[\w\-]+\.(?:com|org|net|io|dev|me|co)/[\w\-/]*'  # General domain patterns
    ]
    
    # Portfolio/website patterns
    portfolio_patterns = [
        r'[\w\-]+\.(?:portfolio|website|blog|site)\.com',
        r'[\w\-]+\.github\.io',
        r'[\w\-]+\.netlify\.app',
        r'[\w\-]+\.vercel\.app',
        r'[\w\-]+\.herokuapp\.com'
    ]
    
    # Check for any hyperlink patterns
    has_links = bool(re.search(full_url_pattern, resume_text, re.IGNORECASE))
    
    if not has_links:
        # Check domain patterns
        for pattern in domain_patterns + portfolio_patterns:
            if re.search(pattern, resume_text, re.IGNORECASE):
                has_links = True
                break
    
    # Also check for common link text indicators
    link_indicators = [
        r'portfolio:?\s*[\w\-\.]+\.[\w\-/]+',
        r'website:?\s*[\w\-\.]+\.[\w\-/]+',
        r'github:?\s*[\w\-\.]+\.[\w\-/]+',
        r'linkedin:?\s*[\w\-\.]+\.[\w\-/]+'
    ]
    
    if not has_links:
        for indicator in link_indicators:
            if re.search(indicator, resume_text, re.IGNORECASE):
                has_links = True
                break
    
    if has_links:
        ats_score += 15
    
    # Tailoring (25 points)
    tailoring_score = 0
    
    # Hard skills
    hard_skills = ['python', 'java', 'javascript', 'sql', 'aws', 'docker', 'kubernetes', 'react', 'angular']
    found_hard_skills = [skill for skill in hard_skills if skill in resume_text.lower()]
    tailoring_score += min(8, len(found_hard_skills) * 2)
    
    # Soft skills
    soft_skills = ['leadership', 'communication', 'teamwork', 'problem-solving', 'analytical']
    found_soft_skills = [skill for skill in soft_skills if skill in resume_text.lower()]
    tailoring_score += min(7, len(found_soft_skills) * 2)
    
    # Action verbs (already counted above, add remaining points)
    tailoring_score += min(10, action_verb_count * 1)
    
    total_score = content_score + section_score + ats_score + tailoring_score
    
    return {
        "total_score": min(100, total_score),
        "content_score": content_score,
        "section_score": section_score,
        "ats_essentials_score": ats_score,
        "tailoring_score": tailoring_score,
        "feedback": {
            "quantified_impact": len(numbers),
            "action_verbs_found": action_verb_count,
            "essential_sections_found": found_sections,
            "hard_skills_found": found_hard_skills,
            "soft_skills_found": found_soft_skills,
            "has_email": bool(re.search(email_pattern, resume_text)),
            "has_phone": bool(re.search(phone_pattern, resume_text)),
            "has_links": has_links
        }
    }

def enhance_resume_with_ai(resume_text: str, ats_analysis: dict) -> str:
    """Enhance resume using AI based on ATS analysis"""
    
    feedback = ats_analysis["feedback"]
    
    enhancement_prompt = f"""
    You are an expert resume writer and ATS optimization specialist. Please enhance the following resume based on the ATS analysis provided.

    Current Resume:
    {resume_text}

    ATS Analysis Results:
    - Total ATS Score: {ats_analysis['total_score']}/100
    - Quantified Impact Items Found: {feedback['quantified_impact']}
    - Action Verbs Found: {feedback['action_verbs_found']}
    - Essential Sections: {', '.join(feedback['essential_sections_found'])}
    - Hard Skills: {', '.join(feedback['hard_skills_found'])}
    - Soft Skills: {', '.join(feedback['soft_skills_found'])}
    - Has Email: {feedback['has_email']}
    - Has Phone: {feedback['has_phone']}
    - Has Links: {feedback['has_links']}

    Please enhance this resume by:
    1. Adding more quantified achievements with specific numbers, percentages, or dollar amounts
    2. Incorporating stronger action verbs
    3. Improving keyword density for ATS optimization
    4. Enhancing the structure and formatting for better readability
    5. Adding relevant technical skills if missing
    6. Improving contact information section if needed
    7. {'Adding professional links (LinkedIn, GitHub, portfolio) if missing' if not feedback['has_links'] else 'Optimizing existing professional links'}
    8. Making the content more compelling and professional

    {'IMPORTANT: This resume is missing professional links/URLs. Please add placeholders like "LinkedIn: linkedin.com/in/[your-profile]" or "Portfolio: [your-website.com]" in the contact section.' if not feedback['has_links'] else ''}

    Please provide the enhanced resume in the following structured format for optimal PDF template rendering:

    [Full Name]
    [Professional Title/Role]
    [Email] | [Phone] | [City, State]
    LinkedIn: [LinkedIn URL]
    GitHub: [GitHub URL]
    Website: [Portfolio URL]

    SUMMARY
    [2-3 sentences professional summary highlighting key achievements, skills, and career objectives]

    EXPERIENCE
    [Job Title]
    [Company Name] | [Location] | [Start Date - End Date]
    - [Quantified achievement with specific metrics and impact]
    - [Another achievement demonstrating skills and results]
    - [Third achievement showing growth and responsibility]

    [Previous Job Title]
    [Previous Company] | [Location] | [Start Date - End Date]
    - [Quantified achievement with numbers/percentages]
    - [Achievement showing technical or leadership skills]

    PROJECTS
    [Project Name] [GitHub/Demo Link]
    - [Brief description of project and technologies used]
    - [Quantified impact or key technical achievements]
    - [Notable features or learning outcomes]

    EDUCATION
    [Degree Name]
    [University Name] | [Location] | [Graduation Year]
    - [GPA if above 3.5, relevant coursework, honors, or achievements]

    SKILLS
    Technical Skills: [List comma-separated technical skills]
    Tools & Technologies: [List frameworks, tools, software]
    Soft Skills: [List relevant soft skills]

    ACHIEVEMENTS
    - [Notable achievement with quantified impact]
    - [Certification, award, or recognition]
    - [Competition result or significant accomplishment]

    IMPORTANT: Provide ONLY the resume content in the exact format above. Do not include any introductory phrases or explanatory text. Start directly with the candidate's name.

    Enhanced Resume:
    """
    
    try:
        response = llm.invoke([HumanMessage(content=enhancement_prompt)])
        enhanced_text = response.content
        
        # Clean up common AI introductory phrases
        cleanup_patterns = [
            r'^Here is the enhanced resume[^:]*:?\s*',
            r'^Enhanced Resume:?\s*',
            r'^Here\'s the enhanced resume[^:]*:?\s*',
            r'^I\'ve enhanced your resume[^:]*:?\s*',
            r'^Below is the enhanced resume[^:]*:?\s*',
            r'^Here\'s your improved resume[^:]*:?\s*',
            r'^Your enhanced resume[^:]*:?\s*',
            r'^The enhanced resume[^:]*:?\s*',
            r'optimized\s+for\s+ATS\s+and\s+readability[^:]*:?\s*',
            r'incorporating\s+all\s+your\s+requirements[^:]*:?\s*',
            r'^[Hh]ere\s+is\s+the\s+enhanced\s+resume[^:]*:?\s*',
            r'^[Hh]ere\'s\s+the\s+enhanced\s+resume[^:]*:?\s*'
        ]
        
        # Apply cleanup patterns
        for pattern in cleanup_patterns:
            enhanced_text = re.sub(pattern, '', enhanced_text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove any leading/trailing whitespace and empty lines
        enhanced_text = enhanced_text.strip()
        
        # Remove multiple consecutive newlines at the beginning
        enhanced_text = re.sub(r'^\n+', '', enhanced_text)
        
        return enhanced_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error enhancing resume: {str(e)}")

def parse_resume_data(resume_text: str) -> dict:
    """Parse resume text into structured data for template rendering"""
    lines = [line.strip() for line in resume_text.split('\n') if line.strip()]
    
    data = {
        'name': '',
        'title': '',
        'contact': {
            'email': '',
            'phone': '',
            'location': '',
            'website': '',
            'linkedin': '',
            'github': ''
        },
        'summary': '',
        'experience': [],
        'education': [],
        'skills': {
            'technical': [],
            'soft': [],
            'tools': []
        },
        'projects': [],
        'achievements': []
    }
    
    current_section = None
    current_item = {}
    
    # Email and phone patterns
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        
        # Extract name (usually first line)
        if i == 0 and not data['name']:
            data['name'] = line
            continue
            
        # Extract professional title (usually second line if not contact info)
        if i == 1 and not data['title'] and not re.search(email_pattern, line) and not re.search(phone_pattern, line):
            data['title'] = line
            continue
            
        # Extract contact information
        if re.search(email_pattern, line):
            data['contact']['email'] = re.search(email_pattern, line).group()
        if re.search(phone_pattern, line):
            data['contact']['phone'] = re.search(phone_pattern, line).group()
        if 'linkedin' in line_lower:
            data['contact']['linkedin'] = line.replace('LinkedIn: ', '')
        if 'github' in line_lower:
            data['contact']['github'] = line.replace('GitHub: ', '')
        if 'website:' in line_lower or ('http' in line and 'github' not in line_lower and 'linkedin' not in line_lower):
            data['contact']['website'] = line.replace('Website: ', '')
        
        # Parse location from contact lines
        if '|' in line and (re.search(email_pattern, line) or re.search(phone_pattern, line)):
            parts = [p.strip() for p in line.split('|')]
            for part in parts:
                if re.search(email_pattern, part):
                    data['contact']['email'] = re.search(email_pattern, part).group()
                elif re.search(phone_pattern, part):
                    data['contact']['phone'] = re.search(phone_pattern, part).group()
                elif not re.search(email_pattern, part) and not re.search(phone_pattern, part) and len(part.split()) <= 3:
                    if not data['contact']['location']:
                        data['contact']['location'] = part
        
        # Detect sections
        section_keywords = {
            'experience': ['experience', 'work', 'employment'],
            'education': ['education', 'qualification'],
            'skills': ['skills', 'technologies', 'technical'],
            'projects': ['projects', 'portfolio'],
            'summary': ['summary', 'objective', 'profile'],
            'achievements': ['achievements', 'awards', 'certifications']
        }
        
        for section, keywords in section_keywords.items():
            if any(keyword in line_lower for keyword in keywords) and len(line.split()) <= 3:
                current_section = section
                current_item = {}
                break
        else:
            # Process content within sections
            if current_section == 'experience':
                if '|' in line or re.search(r'\d{4}', line):
                    # Company and dates line
                    parts = line.split('|') if '|' in line else [line]
                    if current_item:
                        data['experience'].append(current_item)
                    current_item = {
                        'title': '',
                        'company': parts[0].strip(),
                        'duration': parts[1].strip() if len(parts) > 1 else '',
                        'description': []
                    }
                elif line.startswith('-') or line.startswith('•'):
                    if current_item:
                        current_item['description'].append(line.lstrip('- •').strip())
                elif not line.startswith('-') and not line.startswith('•') and current_item:
                    current_item['title'] = line
                    
            elif current_section == 'education':
                if re.search(r'\d{4}', line) or '|' in line:
                    parts = line.split('|') if '|' in line else [line]
                    data['education'].append({
                        'degree': current_item.get('degree', ''),
                        'institution': parts[0].strip(),
                        'duration': parts[1].strip() if len(parts) > 1 else '',
                        'details': []
                    })
                elif not line.startswith('-') and current_section == 'education':
                    current_item = {'degree': line}
                    
            elif current_section == 'skills':
                if ':' in line:
                    skill_type, skills = line.split(':', 1)
                    # Remove markdown formatting
                    skills = re.sub(r'\*\*(.*?)\*\*', r'\1', skills)
                    skill_list = [s.strip() for s in skills.split(',') if s.strip()]
                    if 'technical' in skill_type.lower() or 'programming' in skill_type.lower():
                        data['skills']['technical'].extend(skill_list)
                    elif 'soft' in skill_type.lower():
                        data['skills']['soft'].extend(skill_list)
                    elif 'tools' in skill_type.lower() or 'technologies' in skill_type.lower():
                        data['skills']['tools'].extend(skill_list)
                    else:
                        data['skills']['tools'].extend(skill_list)
                elif line.startswith('-') or line.startswith('•'):
                    skill_text = line.lstrip('- •').strip()
                    data['skills']['tools'].append(skill_text)
                        
            elif current_section == 'projects':
                if line.startswith('-') or line.startswith('•'):
                    if current_item:
                        current_item['description'].append(line.lstrip('- •').strip())
                elif '[' in line and ']' in line:
                    # Project with link
                    if current_item:
                        data['projects'].append(current_item)
                    title_part = line.split('[')[0].strip()
                    current_item = {
                        'title': title_part,
                        'link': line,
                        'description': []
                    }
                elif current_item:
                    current_item['description'].append(line)
                    
            elif current_section == 'summary':
                if data['summary']:
                    data['summary'] += ' ' + line
                else:
                    data['summary'] = line
                    
            elif current_section == 'achievements':
                if line.startswith('-') or line.startswith('•'):
                    data['achievements'].append(line.lstrip('- •').strip())
                else:
                    data['achievements'].append(line)
    
    # Add any remaining items
    if current_section == 'experience' and current_item:
        data['experience'].append(current_item)
    elif current_section == 'projects' and current_item:
        data['projects'].append(current_item)
    
    return data

def create_pdf_from_template(resume_data: dict, filename: str) -> str:
    """Create professional PDF from resume data using ReportLab"""
    try:
        temp_dir = tempfile.gettempdir()
        pdf_path = os.path.join(temp_dir, filename)
        
        # Set up document with margins
        doc = SimpleDocTemplate(
            pdf_path, 
            pagesize=letter,
            leftMargin=0.75*inch,
            rightMargin=0.75*inch,
            topMargin=0.5*inch,
            bottomMargin=0.75*inch
        )
        
        # Define professional styles
        styles = getSampleStyleSheet()
        
        # Custom styles for professional resume
        name_style = ParagraphStyle(
            'NameStyle',
            parent=styles['Title'],
            fontSize=26,
            fontName='Helvetica-Bold',
            textColor=colors.HexColor('#2C3E50'),
            alignment=TA_CENTER,
            spaceAfter=5,
            spaceBefore=10
        )
        
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Normal'],
            fontSize=14,
            fontName='Helvetica-Oblique',
            textColor=colors.HexColor('#7F8C8D'),
            alignment=TA_CENTER,
            spaceAfter=10
        )
        
        contact_style = ParagraphStyle(
            'ContactStyle',
            parent=styles['Normal'],
            fontSize=10,
            fontName='Helvetica',
            textColor=colors.HexColor('#34495E'),
            alignment=TA_CENTER,
            spaceAfter=15
        )
        
        section_header_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading2'],
            fontSize=14,
            fontName='Helvetica-Bold',
            textColor=colors.HexColor('#2C3E50'),
            spaceBefore=20,
            spaceAfter=12,
            borderWidth=1,
            borderColor=colors.HexColor('#3498DB'),
            borderPadding=3,
            leftIndent=0,
            alignment=TA_LEFT
        )
        
        job_title_style = ParagraphStyle(
            'JobTitleStyle',
            parent=styles['Normal'],
            fontSize=12,
            fontName='Helvetica-Bold',
            textColor=colors.HexColor('#2C3E50'),
            spaceBefore=8,
            spaceAfter=2
        )
        
        company_style = ParagraphStyle(
            'CompanyStyle',
            parent=styles['Normal'],
            fontSize=11,
            fontName='Helvetica-Oblique',
            textColor=colors.HexColor('#7F8C8D'),
            spaceAfter=6
        )
        
        description_style = ParagraphStyle(
            'DescriptionStyle',
            parent=styles['Normal'],
            fontSize=10,
            fontName='Helvetica',
            textColor=colors.black,
            leftIndent=15,
            spaceAfter=3,
            bulletIndent=5
        )
        
        summary_style = ParagraphStyle(
            'SummaryStyle',
            parent=styles['Normal'],
            fontSize=11,
            fontName='Helvetica',
            textColor=colors.HexColor('#34495E'),
            alignment=TA_JUSTIFY,
            spaceAfter=10,
            fontStyle='italic'
        )
        
        skill_category_style = ParagraphStyle(
            'SkillCategoryStyle',
            parent=styles['Normal'],
            fontSize=11,
            fontName='Helvetica-Bold',
            textColor=colors.HexColor('#2C3E50'),
            spaceAfter=3
        )
        
        skill_list_style = ParagraphStyle(
            'SkillListStyle',
            parent=styles['Normal'],
            fontSize=10,
            fontName='Helvetica',
            textColor=colors.HexColor('#34495E'),
            spaceAfter=8
        )
        
        def clean_text(text):
            """Remove markdown formatting and clean text"""
            if not text:
                return text
            # Remove markdown bold
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
            # Remove extra asterisks
            text = re.sub(r'\*+', '', text)
            return text.strip()
        
        content = []
        
        # Header Section
        if resume_data.get('name'):
            clean_name = clean_text(resume_data['name'])
            content.append(Paragraph(clean_name, name_style))
        
        if resume_data.get('title'):
            content.append(Paragraph(resume_data['title'], title_style))
        
        # Contact Information
        contact_parts = []
        contact = resume_data.get('contact', {})
        if contact.get('email'):
            contact_parts.append(contact['email'])
        if contact.get('phone'):
            contact_parts.append(contact['phone'])
        if contact.get('location'):
            contact_parts.append(contact['location'])
            
        if contact_parts:
            content.append(Paragraph(' | '.join(contact_parts), contact_style))
        
        # Social links on separate line
        social_parts = []
        if contact.get('linkedin'):
            linkedin_clean = contact['linkedin'].replace('LinkedIn: ', '').replace('linkedin.com/in/', '')
            social_parts.append('LinkedIn: linkedin.com/in/' + linkedin_clean)
        if contact.get('github'):
            github_clean = contact['github'].replace('GitHub: ', '').replace('github.com/', '')
            social_parts.append('GitHub: github.com/' + github_clean)
        if contact.get('website'):
            website_clean = contact['website'].replace('Website: ', '').replace('Portfolio: ', '')
            social_parts.append('Portfolio: ' + website_clean)
            
        if social_parts:
            content.append(Paragraph(' | '.join(social_parts), contact_style))
        
        # Add spacing after header
        content.append(Spacer(1, 20))
        
        # Summary Section
        if resume_data.get('summary'):
            content.append(Paragraph('PROFESSIONAL SUMMARY', section_header_style))
            content.append(Paragraph(resume_data['summary'], summary_style))
        
        # Experience Section
        if resume_data.get('experience'):
            content.append(Paragraph('PROFESSIONAL EXPERIENCE', section_header_style))
            for exp in resume_data['experience']:
                if exp.get('title'):
                    content.append(Paragraph(exp['title'], job_title_style))
                
                company_duration = []
                if exp.get('company'):
                    company_duration.append(exp['company'])
                if exp.get('duration'):
                    company_duration.append(exp['duration'])
                
                if company_duration:
                    content.append(Paragraph(' | '.join(company_duration), company_style))
                
                if exp.get('description'):
                    for desc in exp['description']:
                        content.append(Paragraph(f"• {desc}", description_style))
        
        # Projects Section
        if resume_data.get('projects'):
            content.append(Spacer(1, 10))
            content.append(Paragraph('PROJECTS', section_header_style))
            for project in resume_data['projects']:
                if project.get('title'):
                    title_text = clean_text(project['title'])
                    # Clean up project links
                    if project.get('link'):
                        link_text = project['link']
                        if '[GitHub:' in link_text or '[' in link_text:
                            # Extract clean link from markdown format
                            link_match = re.search(r'\[.*?\]', link_text)
                            if link_match:
                                link_text = link_match.group().strip('[]')
                        title_text += f" - {link_text}"
                    content.append(Paragraph(title_text, job_title_style))
                
                if project.get('description'):
                    for desc in project['description']:
                        clean_desc = clean_text(desc)
                        content.append(Paragraph(f"• {clean_desc}", description_style))
                content.append(Spacer(1, 5))
        
        # Education Section
        if resume_data.get('education'):
            content.append(Paragraph('EDUCATION', section_header_style))
            for edu in resume_data['education']:
                if edu.get('degree'):
                    content.append(Paragraph(edu['degree'], job_title_style))
                
                institution_duration = []
                if edu.get('institution'):
                    institution_duration.append(edu['institution'])
                if edu.get('duration'):
                    institution_duration.append(edu['duration'])
                
                if institution_duration:
                    content.append(Paragraph(' | '.join(institution_duration), company_style))
                
                if edu.get('details'):
                    for detail in edu['details']:
                        content.append(Paragraph(f"• {detail}", description_style))
        
        # Skills Section
        skills = resume_data.get('skills', {})
        if any([skills.get('technical'), skills.get('tools'), skills.get('soft')]):
            content.append(Spacer(1, 10))
            content.append(Paragraph('SKILLS', section_header_style))
            
            if skills.get('technical'):
                content.append(Paragraph('<b>Technical Skills:</b>', skill_category_style))
                tech_skills = [clean_text(skill) for skill in skills['technical'] if skill.strip()]
                if tech_skills:
                    content.append(Paragraph(', '.join(tech_skills), skill_list_style))
            
            if skills.get('tools'):
                content.append(Paragraph('<b>Tools & Technologies:</b>', skill_category_style))
                tool_skills = [clean_text(skill) for skill in skills['tools'] if skill.strip()]
                if tool_skills:
                    content.append(Paragraph(', '.join(tool_skills), skill_list_style))
            
            if skills.get('soft'):
                content.append(Paragraph('<b>Soft Skills:</b>', skill_category_style))
                soft_skills = [clean_text(skill) for skill in skills['soft'] if skill.strip()]
                if soft_skills:
                    content.append(Paragraph(', '.join(soft_skills), skill_list_style))
        
        # Achievements Section
        if resume_data.get('achievements'):
            content.append(Spacer(1, 10))
            content.append(Paragraph('ACHIEVEMENTS', section_header_style))
            for achievement in resume_data['achievements']:
                clean_achievement = clean_text(achievement)
                content.append(Paragraph(f"★ {clean_achievement}", description_style))
            content.append(Spacer(1, 10))
        
        # Build the PDF
        doc.build(content)
        return pdf_path
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating professional PDF: {str(e)}")

def create_pdf_from_text(text: str, filename: str) -> str:
    """Create PDF from enhanced text using HTML template"""
    try:
        # Parse the resume text into structured data
        resume_data = parse_resume_data(text)
        
        # Use the template-based PDF creation
        return create_pdf_from_template(resume_data, filename)
        
    except Exception as e:
        # Fallback to simple text-based PDF if template fails
        return create_simple_pdf_fallback(text, filename)

def create_simple_pdf_fallback(text: str, filename: str) -> str:
    """Simple fallback PDF creation using reportlab"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        
        temp_dir = tempfile.gettempdir()
        pdf_path = os.path.join(temp_dir, filename)
        
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        content = []
        
        paragraphs = text.split('\n')
        for para in paragraphs:
            if para.strip():
                content.append(Paragraph(para.strip(), styles['Normal']))
        
        doc.build(content)
        return pdf_path
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating fallback PDF: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend HTML interface"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return {"message": "Intelligent Resume Enhancement System API - Frontend not found"}

@app.get("/api")
def read_root():
    return {"message": "Intelligent Resume Enhancement System API"}

@app.post("/analyze-resume")
async def analyze_resume(file: UploadFile = File(...)):
    """Analyze uploaded resume and return ATS score"""
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Read file content
    content = await file.read()
    
    # Determine file type and extract text
    if file.content_type == "application/pdf":
        resume_text = extract_text_from_pdf(content)
    elif file.content_type.startswith("image/"):
        resume_text = extract_text_from_image(content)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload PDF or image files.")
    
    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the file")
    
    # Calculate ATS score
    ats_analysis = get_ats_score(resume_text)
    
    return {
        "filename": file.filename,
        "extracted_text": resume_text[:500] + "..." if len(resume_text) > 500 else resume_text,
        "ats_analysis": ats_analysis
    }

@app.post("/enhance-resume")
async def enhance_resume(file: UploadFile = File(...)):
    """Enhance uploaded resume and return improved version"""
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Read file content
    content = await file.read()
    
    # Determine file type and extract text
    if file.content_type == "application/pdf":
        resume_text = extract_text_from_pdf(content)
    elif file.content_type.startswith("image/"):
        resume_text = extract_text_from_image(content)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload PDF or image files.")
    
    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the file")
    
    # Calculate original ATS score
    original_ats_analysis = get_ats_score(resume_text)
    
    # Enhance resume with AI
    enhanced_text = enhance_resume_with_ai(resume_text, original_ats_analysis)
    
    # Calculate enhanced ATS score
    enhanced_ats_analysis = get_ats_score(enhanced_text)
    
    # Create enhanced PDF
    enhanced_filename = f"enhanced_{file.filename.rsplit('.', 1)[0]}.pdf"
    pdf_path = create_pdf_from_text(enhanced_text, enhanced_filename)
    
    return {
        "original_filename": file.filename,
        "enhanced_filename": enhanced_filename,
        "original_ats_score": original_ats_analysis["total_score"],
        "enhanced_ats_score": enhanced_ats_analysis["total_score"],
        "improvement": enhanced_ats_analysis["total_score"] - original_ats_analysis["total_score"],
        "original_analysis": original_ats_analysis,
        "enhanced_analysis": enhanced_ats_analysis,
        "download_url": f"/download/{enhanced_filename}"
    }

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download enhanced resume file"""
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/pdf"
    )

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "Resume Enhancement System is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
