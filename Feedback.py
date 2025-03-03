import streamlit as st
import google.generativeai as genai
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import re
import sqlite3

# Gemini API Configuration (Replace with your actual API key)
GEMINI_API_KEY = "AIzaSyD9NDUbjOJMIRwl0r01iJUt0TULSZb4U9k"
genai.configure(api_key=GEMINI_API_KEY)

# Subjects related to Ganga health
ALLOWED_SUBJECTS = [
    "Water Quality",
    "Pollution",
    "Ecological Impact",
    "Community Concerns",
    "Conservation Efforts",
    "Waste Management",
    "Industrial Discharge"
]

def create_database():
    """
    Create the database and feedback table if they don't exist.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(''' 
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            subject TEXT,
            feedback_text TEXT,
            relevance_score INTEGER,
            analysis TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_feedback_to_db(email, subject, feedback_text, relevance_score, analysis):
    """
    Save user feedback into the database.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO feedback (email, subject, feedback_text, relevance_score, analysis)
        VALUES (?, ?, ?, ?, ?)
    ''', (email, subject, feedback_text, relevance_score, analysis))
    conn.commit()
    conn.close()

def moderate_content(text, selected_subject):
    """
    Use Gemini to provide a concise content moderation analysis.
    """
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""Briefly assess this feedback about the Ganga river for {selected_subject}:
- Is the content relevant to Ganga river health?
- Provide a concise relevance score (0-100)
- Give a short explanation

Text: "{text}"

Respond in this exact format:
Relevance Score: [number]
Brief Analysis: [1-2 sentence explanation]
"""
        
        response = model.generate_content(prompt)
        full_analysis = response.text
        
        # Extract relevance score
        relevance_match = re.search(r'Relevance Score:\s*(\d+)', full_analysis)
        relevance_score = int(relevance_match.group(1)) if relevance_match else 0
        
        # Check if text is relevant
        is_relevant = relevance_score >= 50
        
        return {
            'is_appropriate': is_relevant,
            'relevance_score': relevance_score,
            'analysis': full_analysis
        }
    except Exception as e:
        return {
            'is_appropriate': False, 
            'relevance_score': 0, 
            'analysis': f"Moderation error: {str(e)}"
        }

def send_email(to_email, subject, body):
    """
    Send email with feedback confirmation
    """
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = to_email
        msg['Subject'] = "Ganga Health Feedback Confirmation"
        
        # Email body
        html_body = f"""
        <html>
        <body style="background-color: #f0f9ff; font-family: Arial, sans-serif;">
        <h2 style="color: #2a3d66;">Your Feedback Submission Confirmation</h2>
        <p style="color: #2a3d66;">Thank you for your submission about Ganga river health.</p>
        <hr>
        <h3 style="color: #2a3d66;">Submission Details:</h3>
        <p style="color: #2a3d66;"><strong>Subject:</strong> {subject}</p>
        <p style="color: #2a3d66;"><strong>Your Feedback:</strong></p>
        <blockquote style="background-color: #e1f5fe; border-left: 5px solid #039be5; padding: 10px;">
        {body}
        </blockquote>
        <p style="color: #2a3d66;">Your feedback is valuable in our mission to protect and monitor the Ganga river.</p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(html_body, 'html'))
        
        # Send email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.send_message(msg)
        
        return True
    except Exception as e:
        st.error(f"Email sending error: {e}")
        return False

def main():
    
    # Full page styling
    st.markdown("""
    <style>
        .main {
            background-color: #e0f7fa;
            color: #2a3d66;
            font-family: 'Arial', sans-serif;
        }
        .title {
            text-align: center;
            font-size: 36px;
            color: #01579b;
            margin-top: 20px;
            font-family: 'Iter', sans-serif;
            font-weight: bold;
        }
        .form-container {
            background-color: #ffffff;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .submit-btn {
            background-color: #039be5;
            color: white;
            border: none;
            padding: 12px 30px;
            font-size: 18px;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .submit-btn:hover {
            background-color: #0277bd;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="title">Ganga Health Feedback Forum</div>', unsafe_allow_html=True)

    # Feedback Form
    with st.form(key='feedback_form', clear_on_submit=True):
        st.write("Help us monitor and improve the health of the Ganga River by sharing your valuable feedback.")

        # Subject selection
        selected_subject = st.selectbox(
            "Select Feedback Subject", 
            ALLOWED_SUBJECTS, 
            index=None, 
            help="Choose a subject that best describes your feedback"
        )
        
        # Email input
        user_email = st.text_input("Your Email Address", placeholder="example@email.com")
        
        # Feedback text area
        feedback_text = st.text_area("Your Feedback", placeholder="Share your observations, concerns, or suggestions about Ganga river health")
        
        # Submit button
        submit_button = st.form_submit_button("Submit Feedback", use_container_width=True)
        
        if submit_button:
            # Validate inputs
            if not selected_subject:
                st.error("Please select a subject relevant to Ganga health")
                st.stop()
            
            if not user_email or '@' not in user_email:
                st.error("Please enter a valid email address")
                st.stop()
            
            if not feedback_text:
                st.error("Please provide your feedback")
                st.stop()
            
            # Content moderation using Gemini
            moderation_result = moderate_content(feedback_text, selected_subject)
            
            # More detailed relevance checking
            if not moderation_result['is_appropriate']:
                st.error("Your feedback does not seem sufficiently relevant to Ganga river health. Please adjust your text.")
                # Display the moderation analysis
                st.info(f"Moderation Analysis:\n{moderation_result['analysis']}")
                st.stop()
            
            # Save feedback to database
            save_feedback_to_db(user_email, selected_subject, feedback_text, moderation_result['relevance_score'], moderation_result['analysis'])
            
            # Send email
            email_sent = send_email(user_email, selected_subject, feedback_text)
            
            if email_sent:
                st.success("Feedback submitted successfully! A confirmation email has been sent.")
            else:
                st.error("Failed to send confirmation email. Please try again.")

if __name__ == "__main__":
    create_database()  # Create the database and table when the app starts
    main()
