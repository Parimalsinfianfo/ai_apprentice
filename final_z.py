import streamlit as st
import pandas as pd
import numpy as np
import json
import hashlib
import uuid
import re
import time
import random
import os
import tempfile
import subprocess
import base64
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
from io import BytesIO

# Configure Streamlit page
st.set_page_config(
    page_title="AI Apprentice Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #f0fff0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #32cd32;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #fff0f0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff4444;
        margin: 1rem 0;
    }
    .video-container {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    .record-button {
        background-color: #ff4444;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
    }
    .apprent-logo {
        width: 100%;
        max-width: 300px;
        margin: 0 auto 20px;
        display: block;
    }
    .accreditation-img {
        max-width: 150px;
        margin: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 5px;
    }
    .cohort-table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 20px;
    }
    .cohort-table th {
        background-color: #f0f8ff;
        text-align: left;
        padding: 10px;
        border: 1px solid #ddd;
    }
    .cohort-table td {
        padding: 10px;
        border: 1px solid #ddd;
    }
    .assessment-question {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 4px solid #4e73df;
    }
    .mcq-option {
        padding: 8px;
        margin: 5px 0;
        border-radius: 4px;
        cursor: pointer;
    }
    .mcq-option:hover {
        background-color: #f0f8ff;
    }
    .mcq-selected {
        background-color: #d1e7ff;
        border: 1px solid #1f77b4;
    }
    .assessment-container {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: #f9f9f9;
    }
    .personality-insights {
        background-color: #000000;
        color: #ffffff
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .personality-insights h1,
    .personality-insights h2,
    .personality-insights h3 {
        color: #ffffff    
    }
    
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    session_defaults = {
        'users': {},
        'apprentices': {},
        'companies': {},
        'training_providers': {},
        'current_user': None,
        'user_type': None,
        'video_questions': [
            "Talk 20 seconds about your Name/sex/ethnicity or skip to next",
            "Talk 20 seconds Country/city/College or skip to next",
            "Talk 30 seconds about yourself or skip to next",
            "Talk 20 seconds of education or skip to next",
            "Talk 20 about languages you can speak",
            "Talk 20 seconds your specialisation or skip to next",
            "Talk 20 seconds about what you are looking for or skip to next",
            "Talk 20 second about your current location and flexibility to travel or skip to next",
            "Talk 20 second sports and hobbies or skip to next",
            "Talk 20 your special needs or skip to next"
        ],
        'psychometric_questions': [
            {"question": "I see myself as someone who is talkative", "trait": "Extraversion"},
            {"question": "I see myself as someone who tends to find fault with others", "trait": "Agreeableness"},
            {"question": "I see myself as someone who does a thorough job", "trait": "Conscientiousness"},
            {"question": "I see myself as someone who is depressed, blue", "trait": "Neuroticism"},
            {"question": "I see myself as someone who is original, comes up with new ideas", "trait": "Openness"},
            {"question": "I see myself as someone who is reserved", "trait": "Extraversion"},
            {"question": "I see myself as someone who is helpful and unselfish with others", "trait": "Agreeableness"},
            {"question": "I see myself as someone who can be somewhat careless", "trait": "Conscientiousness"},
            {"question": "I see myself as someone who is relaxed, handles stress well", "trait": "Neuroticism"},
            {"question": "I see myself as someone who is curious about many different things", "trait": "Openness"}
        ],
        'subscription_plans': {
            'BASIC': {'candidates': 50, 'price': 99, 'features': ['Location filtering', 'Basic search']},
            'STANDARD': {'candidates': 100, 'price': 199, 'features': ['Location filtering', 'Advanced search', 'Language preferences']},
            'PROFESSIONAL': {'candidates': 300, 'price': 299, 'features': ['Unlimited access', 'AI matching', 'Dedicated support']}
        },
        'profiler': None,
        'cohorts': {},
        'assessments': {
            'FA0005': {
                'name': 'Software and data foundation apprenticeship',
                'reference': 'FA0005',
                'level': 'L2',
                'questions': [
                    {
                        'id': 'K1',
                        'question': 'Data encryption ensures:',
                        'options': [
                            'A) Data integrity',
                            'B) Data visibility',
                            'C) Protection against unauthorized access',
                            'D) Faster data transmission'
                        ],
                        'correct': 2,
                        'explanation': 'Data encryption ensures protection against unauthorized access by converting data into a code to prevent unauthorized access.'
                    },
                    {
                        'id': 'K2',
                        'question': 'Strong password policies prevent:',
                        'options': [
                            'A) Efficient workflow',
                            'B) Unauthorized access',
                            'C) Software updates',
                            'D) Hardware issues'
                        ],
                        'correct': 1,
                        'explanation': 'Strong password policies help prevent unauthorized access to systems and data.'
                    },
                    {
                        'id': 'K6',
                        'question': 'Phishing primarily targets:',
                        'options': [
                            'A) System performance',
                            'B) User credentials',
                            'C) Data storage',
                            'D) Software licensing'
                        ],
                        'correct': 1,
                        'explanation': 'Phishing attacks primarily target user credentials by tricking users into revealing sensitive information.'
                    },
                    {
                        'id': 'K12',
                        'question': 'Secure data retrieval involves:',
                        'options': [
                            'A) Public networks',
                            'B) Authenticated access',
                            'C) Automatic deletion',
                            'D) Unrestricted access'
                        ],
                        'correct': 1,
                        'explanation': 'Secure data retrieval requires authenticated access to ensure only authorized users can access data.'
                    },
                    {
                        'id': 'K17',
                        'question': 'Security vulnerabilities lead to:',
                        'options': [
                            'A) Improved security',
                            'B) Potential breaches',
                            'C) Reduced risks',
                            'D) Increased efficiency'
                        ],
                        'correct': 1,
                        'explanation': 'Security vulnerabilities can lead to potential breaches if not addressed properly.'
                    },
                    {
                        'id': 'K3',
                        'question': 'Who is typically responsible for defining the software requirements?',
                        'options': [
                            'A) Software Developer',
                            'B) Project Manager',
                            'C) Network Engineer',
                            'D) End-user'
                        ],
                        'correct': 1,
                        'explanation': 'Project managers are typically responsible for defining software requirements.'
                    },
                    {
                        'id': 'K4',
                        'question': 'Supporting organizational goals involves:',
                        'options': [
                            'A) Reducing productivity',
                            'B) Enhancing data accuracy',
                            'C) Restricting collaboration',
                            'D) Ignoring teamwork'
                        ],
                        'correct': 1,
                        'explanation': 'Supporting organizational goals involves enhancing data accuracy to make informed decisions.'
                    },
                    {
                        'id': 'K5',
                        'question': 'Proper documentation ensures:',
                        'options': [
                            'A) Increased confusion',
                            'B) Accuracy and clarity',
                            'C) Less productivity',
                            'D) Higher costs'
                        ],
                        'correct': 1,
                        'explanation': 'Proper documentation ensures accuracy and clarity in processes and communications.'
                    },
                    {
                        'id': 'K7',
                        'question': 'Which stage focuses on checking the functionality and fixing defects?',
                        'options': [
                            'A) Deployment',
                            'B) Testing',
                            'C) Design',
                            'D) Analysis'
                        ],
                        'correct': 1,
                        'explanation': 'The testing stage focuses on checking functionality and fixing defects.'
                    },
                    {
                        'id': 'K8',
                        'question': 'User requirements are best prioritized based on:',
                        'options': [
                            'A) Developer preferences',
                            'B) User and business needs',
                            'C) Technical feasibility alone',
                            'D) Ease of implementation'
                        ],
                        'correct': 1,
                        'explanation': 'User requirements should be prioritized based on user and business needs.'
                    },
                    {
                        'id': 'K9',
                        'question': 'Effective solution architecture must primarily ensure:',
                        'options': [
                            'A) Only coding efficiency',
                            'B) Alignment with user needs and technical feasibility',
                            'C) Rapid completion',
                            'D) Extensive documentation'
                        ],
                        'correct': 1,
                        'explanation': 'Solution architecture must align with user needs and technical feasibility.'
                    },
                    {
                        'id': 'K10',
                        'question': 'Automation primarily:',
                        'options': [
                            'A) Reduces accuracy',
                            'B) Improves efficiency',
                            'C) Limits productivity',
                            'D) Decreases safety'
                        ],
                        'correct': 1,
                        'explanation': 'Automation primarily improves efficiency by reducing manual effort.'
                    },
                    {
                        'id': 'K11',
                        'question': 'Structured data includes:',
                        'options': [
                            'A) Audio files',
                            'B) Databases',
                            'C) Videos',
                            'D) Emails'
                        ],
                        'correct': 1,
                        'explanation': 'Structured data includes databases with organized information.'
                    },
                    {
                        'id': 'K13',
                        'question': 'Data validation ensures:',
                        'options': [
                            'A) Data duplication',
                            'B) Data accuracy',
                            'C) Public access',
                            'D) Increased vulnerability'
                        ],
                        'correct': 1,
                        'explanation': 'Data validation ensures data accuracy and reliability.'
                    },
                    {
                        'id': 'K14',
                        'question': 'Effective data presentation should:',
                        'options': [
                            'A) Confuse audience',
                            'B) Clearly communicate insights',
                            'C) Include excessive details',
                            'D) Exclude visuals'
                        ],
                        'correct': 1,
                        'explanation': 'Effective data presentation should clearly communicate insights to the audience.'
                    },
                    {
                        'id': 'K15',
                        'question': 'UX in software development primarily focuses on:',
                        'options': [
                            'A) Backend coding',
                            'B) User satisfaction and ease of use',
                            'C) Hardware setup',
                            'D) Software licenses'
                        ],
                        'correct': 1,
                        'explanation': 'UX (User Experience) focuses on user satisfaction and ease of use.'
                    },
                    {
                        'id': 'K16',
                        'question': 'Unit testing frameworks focus on:',
                        'options': [
                            'A) Testing small components of software individually',
                            'B) Testing the entire system at once',
                            'C) User interface testing',
                            'D) Performance testing'
                        ],
                        'correct': 0,
                        'explanation': 'Unit testing frameworks test small components of software individually.'
                    }
                ]
            }
        }
    }
    
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Utility functions
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def generate_id() -> str:
    return str(uuid.uuid4())[:8]

# Authentication functions
def register_user(username: str, password: str, user_type: str, profile_data: Dict) -> bool:
    if username in st.session_state.users:
        return False
    
    user_id = generate_id()
    st.session_state.users[username] = {
        'id': user_id,
        'password': hash_password(password),
        'user_type': user_type,
        'created_at': datetime.now(),
        'profile_data': profile_data
    }
    
    # Store in appropriate collection
    if user_type == 'apprentice':
        st.session_state.apprentices[user_id] = profile_data
    elif user_type == 'company':
        st.session_state.companies[user_id] = profile_data
    elif user_type == 'training_provider':
        st.session_state.training_providers[user_id] = profile_data
    
    return True

def login_user(username: str, password: str) -> bool:
    if username in st.session_state.users:
        user = st.session_state.users[username]
        if user['password'] == hash_password(password):
            st.session_state.current_user = username
            st.session_state.user_type = user['user_type']
            return True
    return False

def logout_user():
    st.session_state.current_user = None
    st.session_state.user_type = None

# Profiler functionality - AI libraries loaded only when needed
class GroqLLM:
    def __init__(self, groq_api_key: str, model_name: str = "llama3-8b-8192"):
        try:
            from groq import Groq
            self.client = Groq(api_key=groq_api_key)
            self.model_name = model_name
        except ImportError:
            raise ImportError("Groq library not available")
    
    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

class ApprenticeProfiler:
    def __init__(self):
        self.models = {}
        
    def setup_models(self):
        self.models = {}
        
        try:
            import torch
            from transformers import pipeline
        except ImportError:
            st.warning("Transformers library not available. Some AI features will be limited.")
            return
            
        with st.spinner("Loading AI models..."):
            try:
                self.models['whisper'] = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-tiny.en",
                    device=-1
                )
            except Exception as e:
                st.warning(f"Couldn't load Whisper model: {str(e)}")
            
    def extract_audio_from_video(self, video_path: str) -> Optional[str]:
        try:
            audio_path = video_path.replace('.mp4', '.wav').replace('.mov', '.wav').replace('.webm', '.wav')
            
            commands = [
                ['ffmpeg', '-i', video_path, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', audio_path],
                ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'wav', '-y', audio_path],
                ['ffmpeg', '-i', video_path, audio_path, '-y']
            ]
            
            for cmd in commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    if result.returncode == 0 and os.path.exists(audio_path):
                        return audio_path
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
            return None
        except Exception:
            return None
    
    def transcribe_audio(self, audio_path: str) -> str:
        try:
            if 'whisper' in self.models:
                with st.spinner("Transcribing audio with AI model..."):
                    result = self.models['whisper'](audio_path)
                    return result['text']
            else:
                return "Transcription not available - please enter text manually or ensure proper model setup."
        except Exception:
            return "Transcription failed"
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        try:
            try:
                import pymupdf
            except ImportError:
                return "PDF processing not available"
            
            doc = pymupdf.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception:
            return ""
    
    def extract_candidate_info(self, text: str) -> Dict[str, Any]:
        try:
            info = {
                'name': '',
                'email': '',
                'phone': '',
                'skills': [],
                'education': '',
                'experience': '',
                'goals': '',
                'organizations': []
            }
            
            # Enhanced regex extraction
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, text)
            if emails:
                info['email'] = emails[0]
            
            phone_patterns = [
                r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
                r'(\+\d{1,3}[-.\s]?)?\d{10}',
                r'(\+\d{1,3}[-.\s]?)?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'
            ]
            
            for pattern in phone_patterns:
                phones = re.findall(pattern, text)
                if phones:
                    info['phone'] = phones[0] if isinstance(phones[0], str) else ''.join(phones[0])
                    break
            
            lines = text.split('\n')
            for line in lines[:10]:
                line = line.strip()
                if line and not any(char.isdigit() for char in line) and not '@' in line:
                    words = re.findall(r'\b[A-Z][a-z]+\b', line)
                    if len(words) >= 2:
                        info['name'] = ' '.join(words[:3])
                        break
            
            skill_keywords = [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
                'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring',
                'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'gitlab', 'github',
                'machine learning', 'deep learning', 'ai', 'tensorflow', 'pytorch', 'scikit-learn',
                'leadership', 'communication', 'teamwork', 'problem solving', 'project management',
                'agile', 'scrum', 'analytical thinking', 'creativity'
            ]
            
            text_lower = text.lower()
            found_skills = []
            for skill in skill_keywords:
                if skill in text_lower:
                    found_skills.append(skill.title())
            
            info['skills'] = list(set(found_skills))
            
            education_keywords = ['degree', 'bachelor', 'master', 'phd', 'university', 'college', 'diploma', 'certification']
            education_sentences = []
            
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in education_keywords):
                    education_sentences.append(sentence.strip())
            
            if education_sentences:
                info['education'] = '. '.join(education_sentences[:2])
            
            experience_keywords = ['experience', 'worked', 'position', 'role', 'job', 'company', 'years']
            experience_sentences = []
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in experience_keywords):
                    experience_sentences.append(sentence.strip())
            
            if experience_sentences:
                info['experience'] = '. '.join(experience_sentences[:3])
            
            return info
        except Exception:
            return {
                'name': '',
                'email': '',
                'phone': '',
                'skills': [],
                'education': '',
                'experience': '',
                'goals': '',
                'organizations': []
            }
    
    def analyze_personality(self, text: str) -> Dict[str, float]:
        try:
            personality = {
                'openness': 0.5,
                'conscientiousness': 0.5,
                'extraversion': 0.5,
                'agreeableness': 0.5,
                'neuroticism': 0.5
            }
            
            trait_keywords = {
                'openness': ['creative', 'innovative', 'curious', 'imaginative', 'artistic', 'original', 'inventive', 'experimental'],
                'conscientiousness': ['organized', 'responsible', 'detail', 'planning', 'systematic', 'thorough', 'careful', 'disciplined'],
                'extraversion': ['outgoing', 'social', 'energetic', 'talkative', 'assertive', 'enthusiastic', 'active', 'gregarious'],
                'agreeableness': ['cooperative', 'helpful', 'kind', 'supportive', 'friendly', 'empathetic', 'considerate', 'compassionate'],
                'neuroticism': ['anxious', 'worried', 'stressed', 'emotional', 'sensitive', 'nervous', 'tense', 'moody']
            }
            
            text_lower = text.lower()
            
            for trait, words in trait_keywords.items():
                word_count = sum(1 for word in words if word in text_lower)
                word_frequency = word_count / len(words)
                adjustment = (word_frequency - 0.1) * 0.4
                personality[trait] = max(0.1, min(0.9, 0.5 + adjustment))
            
            return personality
        except Exception:
            return {
                'openness': 0.5,
                'conscientiousness': 0.5,
                'extraversion': 0.5,
                'agreeableness': 0.5,
                'neuroticism': 0.5
            }
    
    def generate_insights(self, candidate_data: Dict, transcript: str, 
                         personality: Dict) -> str:
        try:
            # Get Groq API key from Streamlit secrets
            groq_api_key = st.secrets.get("GROQ_API_KEY", "")
            
            if groq_api_key:
                try:
                    llm = GroqLLM(groq_api_key=groq_api_key)
                    
                    prompt = f"""
Analyze the following candidate information and provide comprehensive insights:

CANDIDATE INFORMATION:
Name: {candidate_data.get('name', 'Not provided')}
Email: {candidate_data.get('email', 'Not provided')}
Phone: {candidate_data.get('phone', 'Not provided')}
Skills: {', '.join(candidate_data.get('skills', []))}
Education: {candidate_data.get('education', 'Not provided')}
Experience: {candidate_data.get('experience', 'Not provided')}

PERSONALITY TRAITS (0-1 scale):
Openness: {personality.get('openness', 0.5):.2f}
Conscientiousness: {personality.get('conscientiousness', 0.5):.2f}
Extraversion: {personality.get('extraversion', 0.5):.2f}
Agreeableness: {personality.get('agreeableness', 0.5):.2f}
Neuroticism: {personality.get('neuroticism', 0.5):.2f}

TRANSCRIPT SAMPLE: {transcript[:500]}...

Please provide a comprehensive analysis with the following sections:
1. Professional Background Summary
2. Key Strengths and Skills Assessment
3. Personality Profile Analysis
4. Career Path Recommendations
5. Development Areas and Suggestions

Format your response clearly with headers and bullet points where appropriate.
"""
                    return llm.generate(prompt)
                except ImportError:
                    return self.generate_rule_based_insights(candidate_data, personality)
            else:
                return self.generate_rule_based_insights(candidate_data, personality)
        except Exception:
            return self.generate_rule_based_insights(candidate_data, personality)

    def generate_rule_based_insights(self, candidate_data: Dict, personality: Dict) -> str:
        insights = []
        name = candidate_data.get('name', 'The candidate')
        
        insights.append("## 📌 Professional Background Summary")
        insights.append(f"{name} presents a profile with the following characteristics:")
        
        if candidate_data.get('skills'):
            skills_count = len(candidate_data['skills'])
            insights.append(f"- Demonstrates {skills_count} identified technical and soft skills")
        
        if candidate_data.get('education'):
            insights.append(f"- Educational background: {candidate_data['education'][:100]}...")
            
        if candidate_data.get('experience'):
            insights.append(f"- Professional experience: {candidate_data['experience'][:100]}...")
        
        insights.append("")
        
        insights.append("## 💡 Key Strengths and Skills Assessment")
        skills = candidate_data.get('skills', [])
        
        if skills:
            technical_skills = [s for s in skills if any(tech in s.lower() for tech in ['python', 'java', 'sql', 'react', 'aws', 'docker'])]
            soft_skills = [s for s in skills if any(soft in s.lower() for soft in ['leadership', 'communication', 'teamwork', 'problem'])]
            
            if technical_skills:
                insights.append(f"- **Technical Skills**: {', '.join(technical_skills[:5])}")
            if soft_skills:
                insights.append(f"- **Soft Skills**: {', '.join(soft_skills[:5])}")
            if len(skills) > 10:
                insights.append(f"- **Skill Diversity**: Demonstrates broad skill set with {len(skills)} identified competencies")
        else:
            insights.append("- Skills assessment requires more detailed information from candidate")
        
        insights.append("")
        
        insights.append("## 🧠 Personality Profile Analysis (OCEAN Model)")
        
        sorted_traits = sorted(personality.items(), key=lambda x: x[1], reverse=True)
        top_traits = [trait for trait, score in sorted_traits[:2] if score > 0.6]
        low_traits = [trait for trait, score in sorted_traits if score < 0.4]
        
        trait_descriptions = {
            'openness': 'creative, curious, and open to new experiences',
            'conscientiousness': 'organized, responsible, and detail-oriented',
            'extraversion': 'outgoing, energetic, and socially confident',
            'agreeableness': 'cooperative, empathetic, and team-oriented',
            'neuroticism': 'emotionally sensitive and reactive to stress'
        }
        
        for trait, score in sorted_traits:
            level = "High" if score > 0.7 else "Moderate" if score > 0.4 else "Low"
            insights.append(f"- **{trait.title()}**: {level} ({score:.2f}) - {trait_descriptions.get(trait, 'characteristic trait')}")
        
        insights.append("")
        
        insights.append("## 🚀 Career Path Recommendations")
        
        if top_traits:
            primary_trait = sorted_traits[0][0]
            
            career_recommendations = {
                'openness': [
                    "Product Manager - Innovation focused roles",
                    "UX/UI Designer - Creative problem solving",
                    "Research & Development - Experimental projects",
                    "Strategy Consultant - Novel solution development"
                ],
                'conscientiousness': [
                    "Project Manager - Detail-oriented execution",
                    "Quality Assurance - Systematic testing approaches",
                    "Operations Manager - Process optimization",
                    "Financial Analyst - Thorough data analysis"
                ],
                'extraversion': [
                    "Sales Manager - Client relationship building",
                    "Team Lead - People management and motivation",
                    "Business Development - Networking and partnerships",
                    "Customer Success Manager - Client engagement"
                ],
                'agreeableness': [
                    "Human Resources - Employee relations and support",
                    "Social Work - Community and individual assistance",
                    "Team Coordinator - Collaborative project management",
                    "Customer Service Manager - Client satisfaction focus"
                ],
                'neuroticism': [
                    "Data Analyst - Structured analytical work",
                    "Technical Writer - Detailed documentation",
                    "Quality Control - Attention to detail and standards",
                    "Research Assistant - Methodical investigation"
                ]
            }
            
            recommendations = career_recommendations.get(primary_trait, ["General management roles", "Analytical positions"])
            
            for rec in recommendations:
                insights.append(f"- {rec}")
        else:
            insights.append("- Versatile profile suitable for various analytical and collaborative roles")
            insights.append("- Consider roles that balance technical skills with interpersonal interaction")
        
        insights.append("")
        
        insights.append("## 📈 Development Areas and Suggestions")
        
        if low_traits:
            for trait in low_traits:
                development_suggestions = {
                    'openness': "Consider exploring creative projects, attending innovation workshops, or engaging in brainstorming sessions",
                    'conscientiousness': "Focus on developing organizational systems, time management tools, and attention to detail practices",
                    'extraversion': "Practice public speaking, join networking events, or take on presentation opportunities",
                    'agreeableness': "Develop empathy through active listening training and collaborative project participation",
                    'neuroticism': "Consider stress management techniques, mindfulness practices, and emotional regulation strategies"
                }
                
                suggestion = development_suggestions.get(trait, f"Focus on developing {trait} skills")
                insights.append(f"- **{trait.title()} Development**: {suggestion}")
        
        insights.append("- **Continuous Learning**: Stay updated with industry trends and emerging technologies")
        insights.append("- **Network Building**: Engage with professional communities and industry groups")
        insights.append("- **Skill Certification**: Consider relevant certifications to validate expertise")
        
        return '\n'.join(insights)

def create_personality_radar_chart(personality_data: Dict[str, float]):
    traits = list(personality_data.keys())
    values = list(personality_data.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=[trait.title() for trait in traits],
        fill='toself',
        name='Personality Profile',
        fillcolor='rgba(31, 119, 180, 0.3)',
        line=dict(color='rgb(31, 119, 180)', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickmode='array',
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=['Low', 'Below Avg', 'Average', 'Above Avg', 'High']
            )),
        showlegend=True,
        title={
            'text': "OCEAN Personality Traits Profile",
            'x': 0.5,
            'xanchor': 'center'
        },
        font=dict(size=12),
        width=500,
        height=500
    )
    
    return fig

def create_skills_chart(skills_list):
    if not skills_list:
        return None
    
    categories = {
        'Programming': ['Python', 'Java', 'Javascript', 'C++', 'C#'],
        'Web Technologies': ['React', 'Angular', 'Vue', 'Node.js', 'Express'],
        'Database': ['SQL', 'MySQL', 'MongoDB', 'PostgreSQL'],
        'Cloud & DevOps': ['AWS', 'Azure', 'Docker', 'Kubernetes'],
        'AI/ML': ['Machine Learning', 'Deep Learning', 'AI', 'Tensorflow'],
        'Soft Skills': ['Leadership', 'Communication', 'Teamwork', 'Problem Solving']
    }
    
    skill_counts = {cat: 0 for cat in categories}
    
    for skill in skills_list:
        for category, keywords in categories.items():
            if any(keyword.lower() in skill.lower() for keyword in keywords):
                skill_counts[category] += 1
                break
    
    filtered_counts = {k: v for k, v in skill_counts.items() if v > 0}
    
    if not filtered_counts:
        return None
    
    fig = go.Figure(data=[
        go.Bar(x=list(filtered_counts.keys()), 
               y=list(filtered_counts.values()),
               marker_color='rgba(31, 119, 180, 0.7)')
    ])
    
    fig.update_layout(
        title="Skills Distribution by Category",
        xaxis_title="Skill Category",
        yaxis_title="Number of Skills",
        showlegend=False
    )
    
    return fig

# Platform UI Components
def render_login_page():
    st.markdown("<div class='main-header'>AI Apprentice Platform</div>", unsafe_allow_html=True)
    st.image("https://github.com/Parimalsinfianfo/zedpro/blob/main/apprent.ai.png?raw=true", 
             width=300, 
             use_container_width=False,
             output_format="PNG")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            if login_user(username, password):
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")
    
    with tab2:
        st.subheader("Register")
        reg_username = st.text_input("Username", key="reg_username")
        reg_password = st.text_input("Password", type="password", key="reg_password")
        user_type = st.selectbox("User Type", ["apprentice", "company", "training_provider", "admin"])
        
        # Basic profile information
        if user_type == "apprentice":
            name = st.text_input("Full Name")
            location = st.text_input("Location")
            profile_data = {"name": name, "location": location, "availability": True}
        elif user_type == "company":
            company_name = st.text_input("Company Name")
            industry = st.text_input("Industry")
            profile_data = {"company_name": company_name, "industry": industry, "subscription": "BASIC"}
        elif user_type == "training_provider":
            provider_name = st.text_input("Provider Name")
            specialization = st.text_input("Specialization")
            profile_data = {"provider_name": provider_name, "specialization": specialization, "subscription": "BASIC"}
        else:
            profile_data = {"admin_level": "standard"}
        
        if st.button("Register"):
            if reg_username and reg_password:
                if register_user(reg_username, reg_password, user_type, profile_data):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Username already exists")
            else:
                st.error("Please fill all fields")

def render_apprentice_dashboard():
    st.title("🎓 Apprentice Dashboard")
    
    user_id = st.session_state.users[st.session_state.current_user]['id']
    profile = st.session_state.apprentices.get(user_id, {})
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Profile", "Video Recording", "Psychometric Test", "Opportunities", "Profile Analyzer"])
    
    with tab1:
        render_apprentice_profile(user_id, profile)
    
    with tab2:
        render_video_recording()
    
    with tab3:
        render_psychometric_test(user_id)
    
    with tab4:
        render_opportunities()
    
    with tab5:
        render_profile_analyzer()

def render_apprentice_profile(user_id: str, profile: Dict):
    st.subheader("Your Profile")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("https://via.placeholder.com/150x150?text=Profile", caption="Profile Picture")
        availability = st.toggle("Available for Opportunities", value=profile.get('availability', True))
        
        if availability != profile.get('availability'):
            st.session_state.apprentices[user_id]['availability'] = availability
    
    with col2:
        st.write("**Personal Information**")
        name = st.text_input("Name", value=profile.get('name', ''))
        location = st.text_input("Location", value=profile.get('location', ''))
        
        st.write("**Academic Information**")
        education = st.text_area("Education", value=profile.get('education', ''))
        
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            skills = st.text_area("Skills", value=", ".join(profile.get('skills', [])))
        with col2_2:
            languages = st.text_area("Languages", value=", ".join(profile.get('languages', [])))
        
        if st.button("Update Profile"):
            st.session_state.apprentices[user_id].update({
                'name': name,
                'location': location,
                'education': education,
                'skills': skills.split(', ') if skills else [],
                'languages': languages.split(', ') if languages else []
            })
            st.success("Profile updated!")

def render_video_recording():
    st.subheader("Video Profile Recording")
    st.info("Record a video profile to help employers get to know you better!")
    
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    
    questions = st.session_state.video_questions
    current_q = st.session_state.current_question
    
    if current_q < len(questions):
        st.write(f"**Question {current_q + 1}/{len(questions)}:**")
        st.write(questions[current_q])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Record Answer"):
                st.info("Recording... (Simulated)")
                time.sleep(2)
                st.success("Answer recorded!")
        
        with col2:
            if st.button("Skip Question"):
                st.session_state.current_question += 1
                st.rerun()
        
        with col3:
            if st.button("Next Question"):
                st.session_state.current_question += 1
                st.rerun()
    
    else:
        st.success("All questions completed!")
        if st.button("Process Video Profile"):
            sample_text = "Hello, my name is John. I'm from London and studying computer science. I speak English and Spanish fluently."
            user_id = st.session_state.users[st.session_state.current_user]['id']
            st.session_state.apprentices[user_id]['video_completed'] = True
            st.success("Video profile processed successfully!")
        
        if st.button("Reset Recording"):
            st.session_state.current_question = 0
            st.rerun()

def generate_personality_insights(scores: Dict[str, float]) -> str:
    insights = []
    name = st.session_state.apprentices.get(
        st.session_state.users[st.session_state.current_user]['id'], {}
    ).get('name', 'You')
    
    insights.append(f"## 🧠 Detailed Personality Insights for {name}")
    
    # Openness insights
    insights.append("### 🌈 Openness to Experience")
    openness = scores.get('openness', 0.5)
    if openness >= 0.7:
        insights.append("- **Creative Visionary**: You have a vivid imagination and enjoy thinking about abstract concepts.")
        insights.append("- **Intellectually Curious**: You're constantly seeking new knowledge and experiences.")
        insights.append("- **Adventurous Spirit**: You're willing to try new things and explore unconventional ideas.")
        insights.append("- **Recommended**: Creative fields, R&D roles, entrepreneurial ventures")
    elif openness >= 0.5:
        insights.append("- **Balanced Approach**: You appreciate both tradition and innovation.")
        insights.append("- **Practical Creativity**: You apply creative thinking to practical problems.")
        insights.append("- **Selective Exploration**: You're open to new experiences but with reasonable boundaries.")
    else:
        insights.append("- **Practical Realist**: You prefer concrete facts over abstract theories.")
        insights.append("- **Traditional Approach**: You value established methods and proven solutions.")
        insights.append("- **Consistency Preference**: You thrive in structured, predictable environments.")
        insights.append("- **Recommended**: Operations, quality control, administrative roles")
    
    # Conscientiousness insights
    insights.append("### 📝 Conscientiousness")
    conscientiousness = scores.get('conscientiousness', 0.5)
    if conscientiousness >= 0.7:
        insights.append("- **Highly Organized**: You have exceptional planning and organizational skills.")
        insights.append("- **Goal-Oriented**: You set ambitious goals and systematically work toward them.")
        insights.append("- **Detail-Focused**: You notice and remember important details others might miss.")
        insights.append("- **Recommended**: Project management, accounting, engineering roles")
    elif conscientiousness >= 0.5:
        insights.append("- **Balanced Productivity**: You maintain good work habits without being perfectionistic.")
        insights.append("- **Adaptable Discipline**: You can be disciplined when needed but also flexible.")
        insights.append("- **Reliable Contributor**: You consistently meet expectations and deadlines.")
    else:
        insights.append("- **Flexible Approach**: You prefer to keep options open rather than plan extensively.")
        insights.append("- **Spontaneous Style**: You're comfortable improvising and adapting in the moment.")
        insights.append("- **Process-Oriented**: You focus more on experiences than strict outcomes.")
        insights.append("- **Recommended**: Creative fields, sales, roles requiring adaptability")
    
    # Extraversion insights
    insights.append("### 💬 Extraversion")
    extraversion = scores.get('extraversion', 0.5)
    if extraversion >= 0.7:
        insights.append("- **Outgoing Socializer**: You gain energy from social interactions and group activities.")
        insights.append("- **Assertive Communicator**: You express your thoughts and opinions confidently.")
        insights.append("- **Enthusiastic Energizer**: You bring energy and excitement to group settings.")
        insights.append("- **Recommended**: Sales, public relations, event planning, team leadership")
    elif extraversion >= 0.5:
        insights.append("- **Socially Balanced**: You enjoy social interaction but also value alone time.")
        insights.append("- **Situational Engagement**: You can be outgoing in familiar settings but reserved in new ones.")
        insights.append("- **Adaptive Energy**: You can draw energy from groups but also recharge alone.")
    else:
        insights.append("- **Reflective Thinker**: You prefer deep one-on-one conversations over large gatherings.")
        insights.append("- **Focused Listener**: You're skilled at attentive listening and thoughtful response.")
        insights.append("- **Independent Worker**: You thrive when working autonomously.")
        insights.append("- **Recommended**: Research, writing, technical roles, independent projects")
    
    # Agreeableness insights
    insights.append("### 🤝 Agreeableness")
    agreeableness = scores.get('agreeableness', 0.5)
    if agreeableness >= 0.7:
        insights.append("- **Empathetic Helper**: You naturally understand and respond to others' feelings.")
        insights.append("- **Cooperative Team Player**: You prioritize group harmony and consensus.")
        insights.append("- **Trusting Nature**: You tend to see the best in people and situations.")
        insights.append("- **Recommended**: Counseling, human resources, customer service, teaching")
    elif agreeableness >= 0.5:
        insights.append("- **Balanced Perspective**: You're generally cooperative but can be assertive when needed.")
        insights.append("- **Situational Empathy**: You're compassionate while maintaining healthy boundaries.")
        insights.append("- **Practical Cooperation**: You value harmony but not at the expense of important principles.")
    else:
        insights.append("- **Analytical Critic**: You focus more on objective analysis than emotional considerations.")
        insights.append("- **Direct Communicator**: You prioritize honesty and efficiency over social niceties.")
        insights.append("- **Competitive Spirit**: You thrive in environments where you can demonstrate your abilities.")
        insights.append("- **Recommended**: Law, investigative work, competitive sales, critical analysis roles")
    
    # Neuroticism insights
    insights.append("### 🧘‍♀️ Emotional Stability (Neuroticism)")
    neuroticism = scores.get('neuroticism', 0.5)
    if neuroticism >= 0.7:
        insights.append("- **Sensitive Perceiver**: You're highly attuned to emotional nuances in yourself and others.")
        insights.append("- **Detail-Oriented Worrier**: You notice potential problems others might overlook.")
        insights.append("- **Deep Feeler**: You experience emotions intensely and reflectively.")
        insights.append("- **Recommended**: Artistic fields, counseling, roles requiring emotional intelligence")
    elif neuroticism >= 0.5:
        insights.append("- **Balanced Reactivity**: You experience normal emotional responses to stress.")
        insights.append("- **Situational Sensitivity**: You respond appropriately to challenging situations without overreacting.")
        insights.append("- **Practical Concerns**: You address potential problems proactively without excessive worry.")
    else:
        insights.append("- **Emotionally Resilient**: You maintain composure even in stressful situations.")
        insights.append("- **Calm Problem-Solver**: You approach challenges with level-headed analysis.")
        insights.append("- **Steady Performer**: You provide stability during crises or high-pressure situations.")
        insights.append("- **Recommended**: Emergency services, crisis management, high-stress environments")
    
    # Career recommendations based on combination
    insights.append("## 🚀 Optimal Career Recommendations")
    insights.append("Based on your combined personality traits:")
    
    if openness >= 0.7 and conscientiousness >= 0.6:
        insights.append("- **Innovation Manager**: Lead creative teams while maintaining project structure")
        insights.append("- **Research Scientist**: Explore new discoveries with methodical approaches")
    elif extraversion >= 0.7 and agreeableness >= 0.6:
        insights.append("- **Client Success Manager**: Build relationships while solving client problems")
        insights.append("- **Team Facilitator**: Enhance collaboration and communication in organizations")
    elif conscientiousness >= 0.7 and neuroticism <= 0.4:
        insights.append("- **Crisis Management Specialist**: Maintain calm during emergencies")
        insights.append("- **Surgical Technician**: Perform precise work under pressure")
    else:
        insights.append("- **Technical Specialist**: Roles focusing on expertise rather than people management")
        insights.append("- **Content Creator**: Develop educational or entertainment material")
    
    # Add development opportunities section
    insights.append("## 📈 Development Opportunities")
    insights.append("Targeted growth areas based on your profile:")
    
    if openness < 0.4:
        insights.append("- **Expand Horizons**: Schedule time each week to explore new ideas or hobbies")
        insights.append("- **Creative Challenges**: Take on projects that require innovative thinking")
    if conscientiousness < 0.4:
        insights.append("- **Organization Systems**: Implement task management tools for better structure")
        insights.append("- **Goal Setting**: Break large objectives into manageable milestones")
    if extraversion < 0.4:
        insights.append("- **Networking Practice**: Attend industry events to build connections")
        insights.append("- **Presentation Skills**: Volunteer for opportunities to speak publicly")
    if agreeableness < 0.4:
        insights.append("- **Active Listening**: Practice summarizing others' perspectives before responding")
        insights.append("- **Collaboration Exercises**: Participate in team-building activities")
    if neuroticism > 0.6:
        insights.append("- **Stress Management**: Develop mindfulness or meditation practices")
        insights.append("- **Resilience Training**: Reframe challenges as growth opportunities")
    
    insights.append("")
    insights.append("### 💡 Continuous Improvement Plan")
    insights.append("1. **Monthly Review**: Reflect on progress in key development areas")
    insights.append("2. **Skill Building**: Identify 1-2 new skills to acquire each quarter")
    insights.append("3. **Feedback Seeking**: Regularly ask for constructive input from peers/managers")
    insights.append("4. **Mentorship**: Connect with experienced professionals in your field")
    
    return '\n'.join(insights)

def render_psychometric_test(user_id: str):
    st.subheader("Psychometric Assessment")
    st.info("Complete this assessment to improve your job matching accuracy!")
    
    if f'psychometric_responses_{user_id}' not in st.session_state:
        st.session_state[f'psychometric_responses_{user_id}'] = {}
    
    responses = st.session_state[f'psychometric_responses_{user_id}']
    questions = st.session_state.psychometric_questions
    
    with st.form("psychometric_test"):
        for i, q in enumerate(questions):
            response = st.slider(
                q['question'], 
                min_value=1, 
                max_value=7, 
                value=responses.get(i, 4),
                help="1 = Strongly Disagree, 7 = Strongly Agree"
            )
            responses[i] = response
        
        submitted = st.form_submit_button("Submit Assessment")
        if submitted:
            trait_scores = calculate_ocean_scores(responses, questions)
            st.session_state.apprentices[user_id]['psychometric_scores'] = trait_scores
            st.success("Assessment completed!")
            
            # Normalize scores to 0-1 range for visualization
            normalized_scores = {trait.lower(): score/7 for trait, score in trait_scores.items()}
            
            col1, col2 = st.columns([1, 2])
            with col1:
                fig = create_personality_radar_chart(normalized_scores)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Generate detailed insights
                insights = generate_personality_insights(normalized_scores)
                st.markdown('<div class="personality-insights">' + insights + '</div>', unsafe_allow_html=True)

def calculate_ocean_scores(responses: Dict, questions: List) -> Dict[str, float]:
    trait_sums = {}
    trait_counts = {}
    
    for i, q in enumerate(questions):
        trait = q['trait']
        if trait not in trait_sums:
            trait_sums[trait] = 0
            trait_counts[trait] = 0
        
        # Reverse score for negatively worded questions
        if "reserved" in q['question'].lower() or "careless" in q['question'].lower() or "fault" in q['question'].lower():
            score = 8 - responses.get(i, 4)  # Reverse scoring (1=7, 2=6, etc.)
        else:
            score = responses.get(i, 4)
            
        trait_sums[trait] += score
        trait_counts[trait] += 1
    
    return {trait: trait_sums[trait] / trait_counts[trait] for trait in trait_sums}

def render_opportunities():
    st.subheader("Available Opportunities")
    
    opportunities = [
        {"title": "Software Developer Apprenticeship", "company": "Tech Corp", "location": "London", "type": "Technology"},
        {"title": "Digital Marketing Apprentice", "company": "Marketing Plus", "location": "Manchester", "type": "Marketing"},
        {"title": "Data Analysis Trainee", "company": "Data Solutions", "location": "Birmingham", "type": "Data Science"},
        {"title": "Mechanical Engineering Apprentice", "company": "Engineering Ltd", "location": "Glasgow", "type": "Engineering"}
    ]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        location_filter = st.selectbox("Location", ["All"] + ["London", "Manchester", "Birmingham", "Glasgow"])
    with col2:
        type_filter = st.selectbox("Type", ["All"] + ["Technology", "Marketing", "Data Science", "Engineering"])
    with col3:
        st.write("")
    
    for opp in opportunities:
        if (location_filter == "All" or opp["location"] == location_filter) and \
           (type_filter == "All" or opp["type"] == type_filter):
            
            with st.expander(f"{opp['title']} - {opp['company']}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Location:** {opp['location']}")
                    st.write(f"**Type:** {opp['type']}")
                    st.write(f"**Company:** {opp['company']}")
                with col2:
                    if st.button(f"Apply", key=f"apply_{opp['title']}"):
                        st.success("Application submitted!")

def render_profile_analyzer():
    st.markdown('<div class="section-header">🧠 Profile Analyzer</div>', unsafe_allow_html=True)
    
    # Initialize profiler if needed
    if 'profiler' not in st.session_state or st.session_state.profiler is None:
        st.session_state.profiler = ApprenticeProfiler()
    
    profiler = st.session_state.profiler
    
    # Only load AI models when actually needed
    if 'models_loaded' not in st.session_state:
        with st.spinner("Initializing AI engine..."):
            profiler.setup_models()
            st.session_state.models_loaded = True
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="section-header">📥 Input Section</div>', unsafe_allow_html=True)
        
        input_method = st.radio(
            "Choose input method:",
            ["🎥 Record Video Interview", "📁 Upload File", "📝 Direct Text Input"],
            help="Select how you want to provide candidate information"
        )
        
        transcript = ""
        candidate_data = {}
        
        if input_method == "🎥 Record Video Interview":
            st.markdown("**Record video interview directly:**")
            st.markdown("""
            <div class="video-container">
                <h4>📹 Video Interview Recording</h4>
                <p>Click the button below to start recording the candidate interview.</p>
            </div>
            """, unsafe_allow_html=True)
            
            video_file = st.camera_input("🎬 Start Video Recording")
            
            if video_file is not None:
                st.video(video_file)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    process_video = st.button("🔄 Process Video", type="primary")
                
                if process_video:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(video_file.read())
                        video_path = tmp_file.name
                    
                    with st.spinner("Processing video interview..."):
                        audio_path = profiler.extract_audio_from_video(video_path)
                        
                        if audio_path and os.path.exists(audio_path):
                            transcript = profiler.transcribe_audio(audio_path)
                            os.unlink(audio_path)
                        os.unlink(video_path)
                        
                        if transcript and len(transcript.strip()) > 10:
                            st.success("✅ Video processed successfully!")
                        else:
                            st.error("❌ Could not process video")
        
        elif input_method == "📁 Upload File":
            st.markdown("**Upload candidate files:**")
            
            uploaded_file = st.file_uploader(
                "Choose file to upload",
                type=['mp3', 'wav', 'mp4', 'mov', 'webm', 'pdf', 'txt']
            )
            
            if uploaded_file is not None:
                file_extension = uploaded_file.name.split('.')[-1].lower()
                process_file = st.button("🔄 Process File", type="primary")
                
                if process_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        file_path = tmp_file.name
                    
                    with st.spinner(f"Processing {file_extension.upper()} file..."):
                        if file_extension == 'pdf':
                            transcript = profiler.extract_pdf_text(file_path)
                        elif file_extension in ['mp3', 'wav']:
                            transcript = profiler.transcribe_audio(file_path)
                        elif file_extension in ['mp4', 'mov', 'webm']:
                            audio_path = profiler.extract_audio_from_video(file_path)
                            if audio_path:
                                transcript = profiler.transcribe_audio(audio_path)
                                os.unlink(audio_path)
                        elif file_extension == 'txt':
                            with open(file_path, 'r', encoding='utf-8') as f:
                                transcript = f.read()
                    
                    os.unlink(file_path)
                    
                    if transcript and len(transcript.strip()) > 10:
                        st.success("✅ File processed successfully!")
                    else:
                        st.error("❌ Could not process file")
        
        else:
            st.markdown("**Enter candidate information manually:**")
            transcript = st.text_area(
                "Candidate Information",
                height=200,
                placeholder="Enter candidate details..."
            )
    
    with col2:
        st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)
        
        if transcript and len(transcript.strip()) > 20:
            with st.spinner("🧠 Analyzing candidate profile..."):
                candidate_data = profiler.extract_candidate_info(transcript)
                personality = profiler.analyze_personality(transcript)
                insights = profiler.generate_insights(
                    candidate_data, transcript, personality
                )
            
            st.success("✅ Analysis completed!")
            
            tab1, tab2, tab3, tab4 = st.tabs(["👤 Profile", "🧠 Personality", "🎯 Insights", "📊 Summary"])
            
            with tab1:
                st.subheader("👤 Candidate Profile")
                
                if any(candidate_data.get(key) for key in ['name', 'email', 'phone']):
                    st.markdown("### 📇 Contact Information")
                    col1, col2 = st.columns(2)
                    with col1:
                        if candidate_data.get('name'):
                            st.write(f"**👤 Name:** {candidate_data['name']}")
                        if candidate_data.get('email'):
                            st.write(f"📧 **Email:** {candidate_data['email']}")
                    with col2:
                        if candidate_data.get('phone'):
                            st.write(f"📱 **Phone:** {candidate_data['phone']}")
                
                if candidate_data.get('skills'):
                    st.markdown("### 💼 Skills")
                    skills_html = ""
                    for skill in candidate_data['skills'][:15]:
                        skills_html += f'<span style="background-color: #e1f5fe; color: #01579b; padding: 2px 8px; margin: 2px; border-radius: 12px; display: inline-block; font-size: 12px;">{skill}</span> '
                    st.markdown(skills_html, unsafe_allow_html=True)
                    
                    skills_chart = create_skills_chart(candidate_data['skills'])
                    if skills_chart:
                        st.plotly_chart(skills_chart, use_container_width=True)
                
                if candidate_data.get('education'):
                    st.markdown("### 🎓 Education")
                    st.write(candidate_data['education'])
                
                if candidate_data.get('experience'):
                    st.markdown("### 💼 Experience")
                    st.write(candidate_data['experience'])
            
            with tab2:
                st.subheader("🧠 Personality Analysis")
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    fig = create_personality_radar_chart(personality)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("### 📊 OCEAN Scores")
                    sorted_traits = sorted(personality.items(), key=lambda x: x[1], reverse=True)
                    
                    for trait, score in sorted_traits:
                        if score > 0.7:
                            level = "High"
                            color = "green"
                        elif score > 0.5:
                            level = "Above Average"
                            color = "blue"
                        elif score > 0.3:
                            level = "Average"
                            color = "orange"
                        else:
                            level = "Below Average"
                            color = "red"
                        
                        st.metric(
                            trait.title(), 
                            f"{score:.2f}",
                            delta=f"{level}"
                        )
            
            with tab3:
                st.subheader("🎯 AI-Generated Insights")
                st.markdown(insights)
            
            with tab4:
                st.subheader("📊 Executive Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Skills Identified", len(candidate_data.get('skills', [])))
                
                with col2:
                    dominant_trait = max(personality.items(), key=lambda x: x[1])
                    st.metric("Primary Trait", dominant_trait[0].title(), f"{dominant_trait[1]:.2f}")
                
                with col3:
                    contact_completeness = sum(1 for key in ['name', 'email', 'phone'] if candidate_data.get(key))
                    st.metric("Contact Info", f"{contact_completeness}/3")
                
                with col4:
                    st.metric("Content Words", len(transcript.split()))
                
                st.download_button(
                    label="💾 Download Report",
                    data=json.dumps({
                        'candidate_info': candidate_data,
                        'personality_analysis': personality,
                        'transcript': transcript,
                        'insights': insights
                    }, indent=2),
                    file_name="candidate_report.json",
                    mime="application/json"
                )
        else:
            st.info("👆 Please provide candidate information to begin analysis")

def render_company_dashboard():
    st.title("🏢 Company Dashboard")
    
    user_id = st.session_state.users[st.session_state.current_user]['id']
    profile = st.session_state.companies.get(user_id, {})
    
    tab1, tab2, tab3, tab4 = st.tabs(["Profile", "Search Candidates", "Subscription", "Analytics"])
    
    with tab1:
        render_company_profile(user_id, profile)
    
    with tab2:
        render_candidate_search(profile)
    
    with tab3:
        render_subscription_management(user_id, profile)
    
    with tab4:
        render_company_analytics()

def render_company_profile(user_id: str, profile: Dict):
    st.subheader("Company Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        company_name = st.text_input("Company Name", value=profile.get('company_name', ''))
        industry = st.text_input("Industry", value=profile.get('industry', ''))
        location = st.text_input("Location", value=profile.get('location', ''))
    
    with col2:
        description = st.text_area("About Us", value=profile.get('description', ''))
        services = st.text_area("Services", value=profile.get('services', ''))
        skill_sets = st.text_area("Required Skill Sets", value=profile.get('skill_sets', ''))
    
    if st.button("Update Profile"):
        st.session_state.companies[user_id].update({
            'company_name': company_name,
            'industry': industry,
            'location': location,
            'description': description,
            'services': services,
            'skill_sets': skill_sets
        })
        st.success("Profile updated!")

def render_candidate_search(profile: Dict):
    st.subheader("Search Candidates")
    
    subscription = profile.get('subscription', 'BASIC')
    plan_limits = st.session_state.subscription_plans[subscription]
    
    st.info(f"Current Plan: {subscription} - Access to {plan_limits['candidates']} candidates")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        location_filter = st.selectbox("Location", ["All", "London", "Manchester", "Birmingham", "Glasgow"])
    with col2:
        skills_filter = st.text_input("Skills (comma separated)")
    with col3:
        availability_filter = st.checkbox("Available only", value=True)
    
    if st.button("Search Candidates"):
        candidates = generate_mock_candidates(plan_limits['candidates'])
        
        st.write(f"Found {len(candidates)} candidates:")
        
        for candidate in candidates:
            with st.expander(f"{candidate['name']} - {candidate['location']}"):
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    st.image("https://via.placeholder.com/100x100?text=Photo", width=100)
                with col2:
                    st.write(f"**Skills:** {', '.join(candidate['skills'])}")
                    st.write(f"**Education:** {candidate['education']}")
                    st.write(f"**Available:** {'Yes' if candidate['availability'] else 'No'}")
                with col3:
                    if st.button("Shortlist", key=f"shortlist_{candidate['id']}"):
                        st.success("Candidate shortlisted!")
                    if st.button("Analyze Profile", key=f"analyze_{candidate['id']}"):
                        st.session_state['analyze_candidate'] = candidate
                        st.rerun()

def generate_mock_candidates(limit: int) -> List[Dict]:
    candidates = []
    names = ["Alice Johnson", "Bob Smith", "Carol Davis", "David Wilson", "Emma Brown", "Frank Miller"]
    locations = ["London", "Manchester", "Birmingham", "Glasgow", "Edinburgh", "Cardiff"]
    skills_list = [["Python", "JavaScript"], ["Data Analysis", "SQL"], ["Marketing", "SEO"], ["Engineering", "CAD"]]
    
    for i in range(min(limit, 20)):
        candidates.append({
            'id': generate_id(),
            'name': random.choice(names),
            'location': random.choice(locations),
            'skills': random.choice(skills_list),
            'education': random.choice(["University Degree", "College Diploma", "A-Levels"]),
            'availability': random.choice([True, False])
        })
    
    return candidates

def render_subscription_management(user_id: str, profile: Dict):
    st.subheader("Subscription Management")
    
    current_plan = profile.get('subscription', 'BASIC')
    st.write(f"**Current Plan:** {current_plan}")
    
    plans = st.session_state.subscription_plans
    
    for plan_name, plan_details in plans.items():
        with st.expander(f"{plan_name} Plan - £{plan_details['price']}/month"):
            st.write(f"**Access to:** {plan_details['candidates']} candidates")
            st.write("**Features:**")
            for feature in plan_details['features']:
                st.write(f"• {feature}")
            
            if plan_name != current_plan:
                if st.button(f"Upgrade to {plan_name}", key=f"upgrade_{plan_name}"):
                    st.session_state.companies[user_id]['subscription'] = plan_name
                    st.success(f"Successfully upgraded to {plan_name}!")
                    st.rerun()

def render_company_analytics():
    st.subheader("Analytics Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
        applications = np.random.randint(10, 50, len(dates))
        fig = px.line(x=dates, y=applications, title="Monthly Applications")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        skills = ['Python', 'JavaScript', 'Data Analysis', 'Marketing', 'Engineering']
        counts = [25, 20, 30, 15, 10]
        fig = px.pie(values=counts, names=skills, title="Candidate Skills Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Regional Statistics")
    regions = ['London', 'Manchester', 'Birmingham', 'Glasgow', 'Edinburgh']
    candidates = [120, 85, 95, 60, 40]
    fig = px.bar(x=regions, y=candidates, title="Candidates by Region")
    st.plotly_chart(fig, use_container_width=True)

def render_training_provider_dashboard():
    st.title("🏫 Training Provider Dashboard")
    
    # Add Apprent.ai logo to the top
    st.sidebar.image("https://github.com/Parimalsinfianfo/zedpro/blob/main/apprent.ai.png?raw=true", 
                     caption="Apprent.ai", 
                     width=200,
                     use_container_width=False)
    
    user_id = st.session_state.users[st.session_state.current_user]['id']
    profile = st.session_state.training_providers.get(user_id, {})
    
    # Updated navigation tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Profile", "Apprenticeship", "Cohorts", "Apprentice", "Employers", "Analytics"
    ])
    
    with tab1:
        render_tp_profile(user_id, profile)
    
    with tab2:
        render_apprenticeship_management()
    
    with tab3:
        render_cohort_management(user_id)
    
    with tab4:
        render_apprentice_management(user_id)
    
    with tab5:
        render_employers_collaboration()
    
    with tab6:
        render_tp_analytics()

def render_tp_profile(user_id: str, profile: Dict):
    st.subheader("Training Provider Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        provider_name = st.text_input("Provider Name", value=profile.get('provider_name', ''))
        specialization = st.text_input("Specialization", value=profile.get('specialization', ''))
        ofsted_rating = st.selectbox("Ofsted Rating", ["Outstanding", "Good", "Requires Improvement", "Inadequate"],
                                     index=1 if profile.get('ofsted_rating') == "Good" else 0)
        location = st.text_input("Location", value=profile.get('location', ''))
        city = st.text_input("City", value=profile.get('city', ''))
    
    with col2:
        about_us = st.text_area("About Us", value=profile.get('about_us', ''))
        courses = st.text_area("Courses or Programmes Offered", value=profile.get('courses', ''))
        accreditations = st.text_area("Accreditations", value=profile.get('accreditations', ''))
        
        st.markdown("**Accreditation Images**")
        uploaded_files = st.file_uploader("Upload Accreditation Images", 
                                         type=["png", "jpg", "jpeg"],
                                         accept_multiple_files=True)
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                st.image(uploaded_file, width=150)
    
    if st.button("Update Profile"):
        st.session_state.training_providers[user_id].update({
            'provider_name': provider_name,
            'specialization': specialization,
            'ofsted_rating': ofsted_rating,
            'location': location,
            'city': city,
            'about_us': about_us,
            'courses': courses,
            'accreditations': accreditations
        })
        st.success("Profile updated!")

def render_apprenticeship_management():
    st.subheader("Authorized Apprenticeships")
    st.info("List of apprenticeships this training provider is authorized to deliver")
    
    # Sample apprenticeships data
    apprenticeships = [
        {
            "type": "Foundation apprenticeship",
            "name": "Software and data foundation apprenticeship",
            "reference": "FA0005",
            "level": "L2"
        },
        {
            "type": "Technical apprenticeship",
            "name": "Data Analyst",
            "reference": "TA0012",
            "level": "L4"
        },
        {
            "type": "Management apprenticeship",
            "name": "Business Leadership",
            "reference": "MA0033",
            "level": "L6"
        }
    ]
    
    # Display as a table
    df = pd.DataFrame(apprenticeships)
    st.dataframe(df, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Add New Apprenticeship")
    
    with st.form("add_apprenticeship"):
        col1, col2 = st.columns(2)
        with col1:
            app_type = st.selectbox("Apprenticeship Type", [
                "Foundation apprenticeship", 
                "Technical apprenticeship", 
                "Management apprenticeship"
            ])
            app_name = st.text_input("Apprenticeship Name")
        with col2:
            app_ref = st.text_input("Reference Code")
            app_level = st.selectbox("Level", ["L2", "L3", "L4", "L5", "L6", "L7"])
        
        if st.form_submit_button("Add Apprenticeship"):
            new_app = {
                "type": app_type,
                "name": app_name,
                "reference": app_ref,
                "level": app_level
            }
            apprenticeships.append(new_app)
            st.success(f"Added new apprenticeship: {app_name}")

def render_cohort_management(user_id: str):
    st.subheader("Cohort Management")
    
    # Initialize cohorts if not exists
    if 'cohorts' not in st.session_state:
        st.session_state.cohorts = {}
    
    # Sample cohorts data
    if not st.session_state.cohorts:
        st.session_state.cohorts = {
            "COH2024-001": {
                "name": "Software Foundation 2024",
                "duration": "12 months",
                "start_date": "2024-01-15",
                "delivery_date": "2024-12-15",
                "apprentices": {
                    "AP123": {
                        "first_name": "John", 
                        "last_name": "Doe", 
                        "email": "john@example.com", 
                        "training_ref": "FA0005", 
                        "level": "L2",
                        "assessment": {
                            "status": "Not Started",
                            "score": 0,
                            "answers": {},
                            "completed": False
                        }
                    },
                    "AP456": {
                        "first_name": "Jane", 
                        "last_name": "Smith", 
                        "email": "jane@example.com", 
                        "training_ref": "FA0005", 
                        "level": "L2",
                        "assessment": {
                            "status": "Completed",
                            "score": 85,
                            "answers": {
                                "K1": 2,
                                "K2": 1,
                                "K6": 1,
                                "K12": 1,
                                "K17": 1,
                                "K3": 1,
                                "K4": 1,
                                "K5": 1,
                                "K7": 1,
                                "K8": 1,
                                "K9": 1,
                                "K10": 1,
                                "K11": 1,
                                "K13": 1,
                                "K14": 1,
                                "K15": 1,
                                "K16": 0
                            },
                            "completed": True
                        }
                    }
                }
            },
            "COH2024-002": {
                "name": "Data Analysis Cohort",
                "duration": "18 months",
                "start_date": "2024-03-01",
                "delivery_date": "2025-08-01",
                "apprentices": {}
            }
        }
    
    # Create two columns - left for cohort list, right for cohort details
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Cohort List")
        
        # Add new cohort button
        if st.button("➕ Create New Cohort"):
            cohort_id = f"COH{datetime.now().strftime('%Y%m%d')}-{len(st.session_state.cohorts)+1}"
            st.session_state.cohorts[cohort_id] = {
                "name": "",
                "duration": "",
                "start_date": "",
                "delivery_date": "",
                "apprentices": {}
            }
            st.session_state.selected_cohort = cohort_id
            st.rerun()
        
        # Display cohorts list
        for cohort_id, cohort_data in st.session_state.cohorts.items():
            if st.button(f"{cohort_id}: {cohort_data['name']}", key=f"cohort_{cohort_id}"):
                st.session_state.selected_cohort = cohort_id
    
    with col2:
        if 'selected_cohort' in st.session_state and st.session_state.selected_cohort in st.session_state.cohorts:
            cohort_id = st.session_state.selected_cohort
            cohort_data = st.session_state.cohorts[cohort_id]
            
            st.markdown(f"### Cohort Details: {cohort_id}")
            
            # Cohort information form
            with st.form(f"cohort_form_{cohort_id}"):
                col2_1, col2_2 = st.columns(2)
                with col2_1:
                    name = st.text_input("Cohort Name", value=cohort_data['name'])
                    duration = st.text_input("Duration", value=cohort_data['duration'])
                with col2_2:
                    start_date = st.date_input("Start Date", 
                                              value=datetime.strptime(cohort_data['start_date'], '%Y-%m-%d') if cohort_data['start_date'] else datetime.today())
                    delivery_date = st.date_input("Delivery Date", 
                                                 value=datetime.strptime(cohort_data['delivery_date'], '%Y-%m-%d') if cohort_data['delivery_date'] else datetime.today() + timedelta(days=365))
                
                if st.form_submit_button("Save Cohort Details"):
                    cohort_data['name'] = name
                    cohort_data['duration'] = duration
                    cohort_data['start_date'] = start_date.strftime('%Y-%m-%d')
                    cohort_data['delivery_date'] = delivery_date.strftime('%Y-%m-%d')
                    st.success("Cohort details updated!")
            
            st.markdown("---")
            st.markdown("### Apprentices in Cohort")
            
            # Display apprentices table
            if cohort_data['apprentices']:
                st.markdown("""
                <table class="cohort-table">
                    <tr>
                        <th>First Name</th>
                        <th>Last Name</th>
                        <th>Email</th>
                        <th>Training Ref</th>
                        <th>Level</th>
                        <th>Assessment Status</th>
                        <th>Actions</th>
                    </tr>
                """, unsafe_allow_html=True)
                
                for apprentice_id, apprentice in cohort_data['apprentices'].items():
                    assessment_status = apprentice.get('assessment', {}).get('status', 'Not Started')
                    st.markdown(f"""
                    <tr>
                        <td>{apprentice['first_name']}</td>
                        <td>{apprentice['last_name']}</td>
                        <td>{apprentice['email']}</td>
                        <td>{apprentice['training_ref']}</td>
                        <td>{apprentice['level']}</td>
                        <td>{assessment_status}</td>
                        <td>
                            <button onclick="window.open('/?apprentice_id={apprentice_id}&cohort_id={cohort_id}', '_blank')">Take Assessment</button>
                        </td>
                    </tr>
                    """, unsafe_allow_html=True)
                
                st.markdown("</table>", unsafe_allow_html=True)
            else:
                st.info("No apprentices added to this cohort yet")
            
            # Add apprentice section
            st.markdown("---")
            st.markdown("### Add Apprentice to Cohort")
            
            add_option = st.radio("Add apprentice:", ["Select from existing", "Create new apprentice"])
            
            if add_option == "Select from existing":
                # Get all apprentices
                apprentices = list(st.session_state.apprentices.values())
                if apprentices:
                    apprentice_names = [f"{a.get('name', 'Unknown')} ({a.get('email', 'No email')})" for a in apprentices]
                    selected_apprentice = st.selectbox("Select Apprentice", apprentice_names)
                    
                    if st.button("Add Selected Apprentice"):
                        # Add to cohort
                        st.success("Apprentice added to cohort!")
                else:
                    st.info("No apprentices available in the system")
            else:
                with st.form("add_new_apprentice"):
                    col3_1, col3_2 = st.columns(2)
                    with col3_1:
                        first_name = st.text_input("First Name")
                        training_ref = st.selectbox("Training Ref", ["FA0005", "TA0012", "MA0033"])
                    with col3_2:
                        last_name = st.text_input("Last Name")
                        level = st.selectbox("Level", ["L2", "L3", "L4", "L5", "L6", "L7"])
                    email = st.text_input("Email")
                    
                    if st.form_submit_button("Add New Apprentice"):
                        new_apprentice = {
                            "first_name": first_name,
                            "last_name": last_name,
                            "email": email,
                            "training_ref": training_ref,
                            "level": level,
                            "assessment": {
                                "status": "Not Started",
                                "score": 0,
                                "answers": {},
                                "completed": False
                            }
                        }
                        apprentice_id = f"AP{generate_id()}"
                        cohort_data['apprentices'][apprentice_id] = new_apprentice
                        st.success("New apprentice added to cohort!")
            
            st.markdown("---")
            st.markdown("### Cohort Assessment")
            
            # Assessment section
            if st.button("Send Assessment Links to Cohort"):
                st.success("Assessment links sent to all apprentices in this cohort!")
            
            if cohort_data['apprentices']:
                if st.button("View Assessment Results"):
                    st.session_state.show_assessment = True
            
            if 'show_assessment' in st.session_state and st.session_state.show_assessment:
                render_cohort_assessment(cohort_id)
        else:
            st.info("Please select a cohort from the list")

def render_cohort_assessment(cohort_id: str):
    st.subheader("Cohort Assessment Results")
    
    cohort_data = st.session_state.cohorts[cohort_id]
    assessment_results = {}
    
    for apprentice_id, apprentice in cohort_data['apprentices'].items():
        if 'assessment' in apprentice and apprentice['assessment']['completed']:
            assessment_results[apprentice_id] = {
                "name": f"{apprentice['first_name']} {apprentice['last_name']}",
                "score": apprentice['assessment']['score'],
                "status": apprentice['assessment']['status'],
                "details": apprentice['assessment']['answers']
            }
    
    if not assessment_results:
        st.info("No assessment results available yet")
        return
    
    # Overall stats
    scores = [result['score'] for result in assessment_results.values()]
    avg_score = sum(scores) / len(scores) if scores else 0
    pass_rate = sum(1 for score in scores if score >= 70) / len(scores) * 100 if scores else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Apprentices", len(assessment_results))
    col2.metric("Average Score", f"{avg_score:.1f}%")
    col3.metric("Pass Rate", f"{pass_rate:.1f}%")
    
    # Display assessment questions and results
    assessment = st.session_state.assessments['FA0005']
    
    st.markdown("### Assessment Questions")
    for i, question in enumerate(assessment['questions']):
        st.markdown(f"**{question['id']}:** {question['question']}")
        
        # Display options
        for j, option in enumerate(question['options']):
            st.markdown(f"<div class='mcq-option'>{option}</div>", unsafe_allow_html=True)
        
        # Display correct answer
        st.markdown(f"**Correct Answer:** {question['options'][question['correct']]}")
        st.markdown(f"**Explanation:** {question['explanation']}")
        st.markdown("---")
    
    # Results per apprentice
    st.markdown("### Apprentice Results")
    for apprentice_id, result in assessment_results.items():
        with st.expander(f"{result['name']} - Score: {result['score']}%"):
            st.write(f"**Status:** {result['status']}")
            
            # Display answers
            for q_id, q_result in result['details'].items():
                q_data = next(q for q in assessment['questions'] if q['id'] == q_id)
                st.markdown(f"**{q_id}:** {q_data['question']}")
                st.markdown(f"**Selected:** {q_data['options'][q_result]}")
                st.markdown(f"**Correct:** {'✅' if q_result == q_data['correct'] else '❌'}")
                st.markdown("---")

def render_apprentice_management(user_id: str):
    st.subheader("Apprentice Management")
    st.info("Manage apprentices in your programs")
    
    # Add new apprentice button
    if st.button("➕ Add New Apprentice", key="add_new_apprentice"):
        # Implementation for adding new apprentice
        st.success("New apprentice added!")
    
    # Sample apprentices data
    apprentices = [
        {"name": "John Doe", "program": "Digital Marketing", "progress": 75, "status": "Active"},
        {"name": "Jane Smith", "program": "Software Development", "progress": 60, "status": "Active"},
        {"name": "Mike Johnson", "program": "Data Analysis", "progress": 90, "status": "Near Completion"}
    ]
    
    # Search and filter
    col1, col2 = st.columns(2)
    with col1:
        search_name = st.text_input("Search by Name")
    with col2:
        filter_status = st.selectbox("Filter by Status", ["All", "Active", "Completed", "Withdrawn"])
    
    # Display apprentices
    for apprentice in apprentices:
        if search_name and search_name.lower() not in apprentice['name'].lower():
            continue
        if filter_status != "All" and filter_status != apprentice['status']:
            continue
            
        with st.expander(f"{apprentice['name']} - {apprentice['program']}"):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**Program:** {apprentice['program']}")
                st.progress(apprentice['progress'] / 100)
                st.write(f"Progress: {apprentice['progress']}%")
            with col2:
                st.write(f"**Status:** {apprentice['status']}")
            with col3:
                if st.button("View Details", key=f"view_{apprentice['name']}"):
                    st.info("Detailed view opened!")
                if st.button("Edit", key=f"edit_{apprentice['name']}"):
                    # Implementation for editing apprentice
                    st.success("Apprentice details updated!")

def render_employers_collaboration():
    st.subheader("Employer Collaboration")
    st.info("Companies you're collaborating with")
    
    # Sample employers data
    employers = [
        {"name": "Tech Corp", "city": "London", "cohorts": ["COH2024-001"], "apprentices": 15},
        {"name": "Marketing Plus", "city": "Manchester", "cohorts": ["COH2024-002"], "apprentices": 8},
        {"name": "Data Solutions", "city": "Birmingham", "cohorts": ["COH2023-005"], "apprentices": 12}
    ]
    
    # Display employers
    for employer in employers:
        with st.expander(f"{employer['name']} - {employer['city']}"):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**Cohorts:** {', '.join(employer['cohorts'])}")
                st.write(f"**Number of Apprentices:** {employer['apprentices']}")
            with col2:
                if st.button("Contact", key=f"contact_{employer['name']}"):
                    st.success("Message sent to company!")
                if st.button("View Details", key=f"details_{employer['name']}"):
                    # Implementation for viewing employer details
                    st.info("Employer details opened!")

def render_tp_analytics():
    st.subheader("Training Provider Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Cohort Performance")
        cohorts = ["Software Foundation 2024", "Data Analysis 2024", "Business Leadership 2023"]
        completion_rates = [85, 78, 92]
        fig = px.bar(x=cohorts, y=completion_rates, title="Cohort Completion Rates (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Program Distribution")
        programs = ['Digital Marketing', 'Software Dev', 'Data Analysis', 'Engineering']
        students = [35, 42, 28, 25]
        fig = px.pie(values=students, names=programs, title="Students by Program")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Assessment Performance")
    
    # Assessment analytics
    col3, col4 = st.columns(2)
    with col3:
        questions = ["K1", "K2", "K6", "K12", "K17"]
        correct_rates = [85, 92, 78, 88, 82]
        fig = px.bar(x=questions, y=correct_rates, title="Question Correct Rates (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        scores = [65, 72, 85, 78, 90, 82, 88, 75, 80, 85]
        fig = px.histogram(x=scores, nbins=10, title="Score Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Candidate Progression")
    
    # Mock progression data
    progression_data = pd.DataFrame({
        "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        "Started": [25, 30, 35, 40, 45, 50],
        "Completed": [0, 5, 10, 18, 28, 40]
    })
    
    fig = px.line(progression_data, x="Month", y=["Started", "Completed"], 
                 title="Apprentice Progression Over Time")
    st.plotly_chart(fig, use_container_width=True)

def render_admin_dashboard():
    st.title("⚙️ Admin Dashboard")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["System Config", "User Management", "Analytics", "Content Management", "Reports"])
    
    with tab1:
        render_system_config()
    
    with tab2:
        render_user_management()
    
    with tab3:
        render_admin_analytics()
    
    with tab4:
        render_content_management()
    
    with tab5:
        render_admin_reports()

def render_system_config():
    st.subheader("System Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Regional Settings**")
        default_country = st.selectbox("Default Country", ["UK", "US", "Canada", "Australia"])
        supported_languages = st.multiselect(
            "Supported Languages", 
            ["English", "Punjabi", "Urdu", "Hindi", "Mirpuri", "Arabic"],
            default=["English", "Punjabi", "Urdu", "Hindi"]
        )
        
        st.write("**Payment Settings**")
        payment_gateway = st.selectbox("Payment Gateway", ["Stripe", "PayPal", "Square"])
        currency = st.selectbox("Default Currency", ["GBP", "USD", "EUR"])
    
    with col2:
        st.write("**Platform Settings**")
        max_video_duration = st.number_input("Max Video Duration (minutes)", value=5)
        max_file_size = st.number_input("Max File Size (MB)", value=50)
        
        st.write("**Notification Settings**")
        email_notifications = st.checkbox("Email Notifications", value=True)
        sms_notifications = st.checkbox("SMS Notifications", value=False)
    
    if st.button("Save Configuration"):
        st.success("Configuration saved successfully!")

def render_user_management():
    st.subheader("User Management")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Users", len(st.session_state.users))
    with col2:
        apprentice_count = len([u for u in st.session_state.users.values() if u['user_type'] == 'apprentice'])
        st.metric("Apprentices", apprentice_count)
    with col3:
        company_count = len([u for u in st.session_state.users.values() if u['user_type'] == 'company'])
        st.metric("Companies", company_count)
    with col4:
        tp_count = len([u for u in st.session_state.users.values() if u['user_type'] == 'training_provider'])
        st.metric("Training Providers", tp_count)
    
    st.write("**Recent Users:**")
    users_data = []
    for username, user_data in st.session_state.users.items():
        users_data.append({
            'Username': username,
            'Type': user_data['user_type'].title(),
            'Created': user_data['created_at'].strftime('%Y-%m-%d'),
            'Status': 'Active'
        })
    
    if users_data:
        df = pd.DataFrame(users_data)
        st.dataframe(df, use_container_width=True)

def render_admin_analytics():
    st.subheader("Platform Analytics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Daily Active Users", "1,234", "+5.2%")
    with col2:
        st.metric("Job Applications", "856", "+12.4%")
    with col3:
        st.metric("Successful Matches", "142", "+8.1%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
        users = np.cumsum(np.random.randint(50, 200, 12))
        fig = px.line(x=dates, y=users, title="User Growth Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        regions = ['London', 'Manchester', 'Birmingham', 'Glasgow']
        success_rates = [78, 65, 72, 69]
        fig = px.bar(x=regions, y=success_rates, title="Success Rate by Region (%)")
        st.plotly_chart(fig, use_container_width=True)

def render_content_management():
    st.subheader("Content Management")
    
    st.write("**Video Recording Questions:**")
    questions = st.session_state.video_questions.copy()
    
    for i, question in enumerate(questions):
        col1, col2 = st.columns([4, 1])
        with col1:
            new_question = st.text_input(f"Question {i+1}", value=question, key=f"q_{i}")
            questions[i] = new_question
        with col2:
            if st.button("Delete", key=f"del_q_{i}"):
                questions.pop(i)
                st.session_state.video_questions = questions
                st.rerun()
    
    new_question = st.text_input("Add New Question")
    if st.button("Add Question") and new_question:
        st.session_state.video_questions.append(new_question)
        st.rerun()
    
    if st.button("Save Questions"):
        st.session_state.video_questions = questions
        st.success("Questions updated!")
    
    st.write("**Subscription Plans:**")
    plans = st.session_state.subscription_plans.copy()
    
    for plan_name, plan_data in plans.items():
        with st.expander(f"{plan_name} Plan"):
            col1, col2 = st.columns(2)
            with col1:
                new_price = st.number_input(f"Price (£)", value=plan_data['price'], key=f"price_{plan_name}")
                new_candidates = st.number_input(f"Candidate Limit", value=plan_data['candidates'], key=f"candidates_{plan_name}")
            with col2:
                features_text = '\n'.join(plan_data['features'])
                new_features = st.text_area(f"Features (one per line)", value=features_text, key=f"features_{plan_name}")
            
            if st.button(f"Update {plan_name}", key=f"update_{plan_name}"):
                plans[plan_name] = {
                    'price': new_price,
                    'candidates': new_candidates,
                    'features': new_features.strip().split('\n')
                }
                st.session_state.subscription_plans = plans
                st.success(f"{plan_name} plan updated!")

def render_admin_reports():
    st.subheader("System Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox("Report Type", [
            "User Activity Report",
            "Subscription Revenue Report", 
            "Matching Success Report",
            "Regional Performance Report"
        ])
        
        date_from = st.date_input("From Date", datetime.now() - timedelta(days=30))
        date_to = st.date_input("To Date", datetime.now())
    
    with col2:
        export_format = st.selectbox("Export Format", ["CSV", "PDF", "Excel"])
        
        if st.button("Generate Report"):
            if report_type == "User Activity Report":
                data = {
                    'Date': pd.date_range(start=date_from, end=date_to, freq='D'),
                    'New Users': np.random.randint(5, 25, (date_to - date_from).days + 1),
                    'Active Users': np.random.randint(50, 200, (date_to - date_from).days + 1),
                    'Applications': np.random.randint(10, 50, (date_to - date_from).days + 1)
                }
            elif report_type == "Subscription Revenue Report":
                data = {
                    'Plan': ['BASIC', 'STANDARD', 'PROFESSIONAL'],
                    'Subscribers': [45, 23, 12],
                    'Revenue': [4455, 4577, 3588]
                }
            elif report_type == "Matching Success Report":
                data = {
                    'Industry': ['Technology', 'Marketing', 'Engineering', 'Healthcare'],
                    'Matches': [156, 89, 134, 78],
                    'Success Rate': [0.78, 0.65, 0.72, 0.69]
                }
            else:
                data = {
                    'Region': ['London', 'Manchester', 'Birmingham', 'Glasgow', 'Edinburgh'],
                    'Users': [450, 234, 189, 167, 123],
                    'Companies': [67, 34, 28, 22, 18],
                    'Success Rate': [0.78, 0.65, 0.72, 0.69, 0.71]
                }
            
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
            st.success(f"Report generated! Download link sent to admin email.")

def render_assessment():
    if 'cohort_id' not in st.query_params or 'apprentice_id' not in st.query_params:
        st.error("Invalid assessment link")
        return
    
    cohort_id = st.query_params['cohort_id']
    apprentice_id = st.query_params['apprentice_id']
    
    if cohort_id not in st.session_state.cohorts or apprentice_id not in st.session_state.cohorts[cohort_id]['apprentices']:
        st.error("Cohort or apprentice not found")
        return
    
    apprentice = st.session_state.cohorts[cohort_id]['apprentices'][apprentice_id]
    assessment = apprentice['assessment']
    
    st.title("🎓 Apprenticeship Assessment")
    st.info(f"Complete this assessment for the {apprentice['training_ref']} apprenticeship")
    
    assessment_data = st.session_state.assessments['FA0005']
    
    if assessment['completed']:
        st.success(f"You have already completed this assessment. Your score: {assessment['score']}%")
        return
    
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
        st.session_state.answers = {}
    
    current_question = st.session_state.current_question
    questions = assessment_data['questions']
    
    if current_question < len(questions):
        question = questions[current_question]
        
        st.markdown(f"### Question {current_question + 1}/{len(questions)}")
        st.markdown(f"**{question['id']}:** {question['question']}")
        
        selected_option = st.session_state.answers.get(question['id'], -1)
        
        for i, option in enumerate(question['options']):
            if st.button(option, key=f"option_{current_question}_{i}", 
                         use_container_width=True,
                         type="primary" if selected_option == i else "secondary"):
                st.session_state.answers[question['id']] = i
                st.session_state.current_question += 1
                st.rerun()
    else:
        # Calculate score
        score = 0
        for q in questions:
            if st.session_state.answers.get(q['id']) == q['correct']:
                score += 1
        
        score_percent = int((score / len(questions)) * 100)
        
        # Update apprentice assessment data
        st.session_state.cohorts[cohort_id]['apprentices'][apprentice_id]['assessment'] = {
            "status": "Completed",
            "score": score_percent,
            "answers": st.session_state.answers,
            "completed": True
        }
        
        st.success(f"Assessment completed! Your score: {score_percent}%")
        
        if st.button("Back to Dashboard"):
            st.switch_page("main.py")

# Main application logic
def main():
    init_session_state()
    
    # Sidebar for navigation
    with st.sidebar:
        st.title("Navigation")
        
        # Add Apprent.ai logo to sidebar
        st.image("https://github.com/Parimalsinfianfo/zedpro/blob/main/apprent.ai.png?raw=true", 
                 caption="Apprent.ai", 
                 width=200,
                 use_container_width=False)
        
        if st.session_state.current_user:
            st.write(f"Logged in as: **{st.session_state.current_user}**")
            st.write(f"Role: **{st.session_state.user_type.title()}**")
            
            if st.button("Logout"):
                logout_user()
                st.rerun()
        else:
            st.write("Please log in to continue")
    
    # Check if we're on an assessment page
    if 'apprentice_id' in st.query_params and 'cohort_id' in st.query_params:
        render_assessment()
    else:
        # Main content
        if not st.session_state.current_user:
            render_login_page()
        else:
            user_type = st.session_state.user_type
            
            if user_type == "apprentice":
                render_apprentice_dashboard()
            elif user_type == "company":
                render_company_dashboard()
            elif user_type == "training_provider":
                render_training_provider_dashboard()
            elif user_type == "admin":
                render_admin_dashboard()
            else:
                st.error("Unknown user type")

if __name__ == "__main__":
    main()  