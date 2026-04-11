from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import random
from flask_socketio import SocketIO

import sqlite3
import numpy as np
import pandas as pd
import pickle
import os
import requests
from dotenv import load_dotenv
import logging

# helper decorator
from functools import wraps

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'supersecret123')
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# database initialization
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'users.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        '''CREATE TABLE IF NOT EXISTS users (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               name TEXT NOT NULL,
               email TEXT NOT NULL UNIQUE,
               password TEXT NOT NULL
           )'''
    )
    c.execute(
        '''CREATE TABLE IF NOT EXISTS predictions (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               user_id INTEGER NOT NULL,
               disease TEXT NOT NULL,
               symptoms TEXT NOT NULL,
               date_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
               severity TEXT DEFAULT 'Medium',
               FOREIGN KEY (user_id) REFERENCES users(id)
           )'''
    )
    # Reports table for patient history / past reports
    c.execute(
        '''CREATE TABLE IF NOT EXISTS reports (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               userId INTEGER NOT NULL,
               disease TEXT NOT NULL,
               severity TEXT DEFAULT 'Medium',
               date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
               FOREIGN KEY (userId) REFERENCES users(id)
           )'''
    )
    conn.commit()
    conn.close()

init_db()

# authentication decorator

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# load databasedataset
def safe_read_csv(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        logging.warning(f"Failed to read {path}: {e}")
        return pd.DataFrame()

sym_des = safe_read_csv("dataset/symtoms_df.csv")
precautions = safe_read_csv("dataset/precautions_df.csv")
workout = safe_read_csv("dataset/workout_df.csv")
description = safe_read_csv("dataset/description.csv")
medications = safe_read_csv('dataset/medications.csv')
diets = safe_read_csv("dataset/diets.csv")


# load model..........................................................
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "svc.pkl")

svc = None
try:
    with open(MODEL_PATH, "rb") as f:
        svc = pickle.load(f)
    logging.info(f"Loaded model from {MODEL_PATH}")
except Exception as e:
    logging.error(f"Failed to load model {MODEL_PATH}: {e}")
    svc = None



# custome and helping functions.......................................................
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis] ['workout']


    return desc,pre,med,die,wrkout

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Model Prediction function...........................................................
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    valid_symptoms = []

    for item in patient_symptoms:
        item = item.lower().replace(" ", "_")
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
            valid_symptoms.append(item)

    # DEBUG (optional)....................................................................
    print("Valid Symptoms:", valid_symptoms)
    print("Vector Sum:", np.sum(input_vector))

    if np.sum(input_vector) == 0:
        return "No valid symptoms found"

    # Ensure model is loaded
    if svc is None:
        logging.error("Prediction requested but model is not loaded")
        return "Model not available"

    try:
        pred_index = svc.predict([input_vector])[0]
        return diseases_list.get(pred_index, "Unknown disease")
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return "Prediction error"

def save_prediction(user_id, disease, symptoms):
    """Save prediction to database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # Compute a simple severity label from severity score (0-100)
        try:
            score = calculate_severity_score(user_id)
        except Exception:
            score = None

        if score is None:
            severity_label = 'Medium'
        elif score >= 80:
            severity_label = 'High'
        elif score >= 40:
            severity_label = 'Medium'
        else:
            severity_label = 'Low'

        # Insert into legacy predictions table (keeps existing behavior)
        c.execute(
            "INSERT INTO predictions (user_id, disease, symptoms, severity) VALUES (?, ?, ?, ?)",
            (user_id, disease, symptoms, severity_label)
        )

        # Also insert a concise record into the new reports table
        c.execute(
            "INSERT INTO reports (userId, disease, severity) VALUES (?, ?, ?)",
            (user_id, disease, severity_label)
        )

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving prediction: {e}")
        return False

def get_user_predictions(user_id, limit=5):
    """Get user's prediction history"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "SELECT disease, symptoms, date_time FROM predictions WHERE user_id=? ORDER BY date_time DESC LIMIT ?",
            (user_id, limit)
        )
        predictions = c.fetchall()
        conn.close()
        return predictions
    except Exception as e:
        print(f"Error fetching predictions: {e}")
        return []


def get_reports(user_id, limit=None):
    """Return reports for a user from the reports table, latest first"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        if limit:
            c.execute("SELECT disease, severity, date FROM reports WHERE userId=? ORDER BY date DESC LIMIT ?", (user_id, limit))
        else:
            c.execute("SELECT disease, severity, date FROM reports WHERE userId=? ORDER BY date DESC", (user_id,))
        rows = c.fetchall()
        conn.close()
        return rows
    except Exception as e:
        print(f"Error fetching reports: {e}")
        return []

def get_dashboard_stats(user_id):
    """Get dashboard statistics"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Total predictions count
        c.execute("SELECT COUNT(*) FROM predictions WHERE user_id=?", (user_id,))
        total_predictions = c.fetchone()[0]
        
        # Last disease detected
        c.execute("SELECT disease FROM predictions WHERE user_id=? ORDER BY date_time DESC LIMIT 1", (user_id,))
        last_disease = c.fetchone()
        last_disease = last_disease[0] if last_disease else "None"
        
        # Last check date
        c.execute("SELECT date_time FROM predictions WHERE user_id=? ORDER BY date_time DESC LIMIT 1", (user_id,))
        last_check = c.fetchone()
        last_check = last_check[0] if last_check else "Never"
        
        conn.close()

        # Calculate risk level with frequency (avoid redundant calls)
        frequency = get_disease_frequency(user_id)
        risk_level, risk_count = calculate_risk_level(user_id, frequency)
        
        # Pass pre-computed values to avoid recursion
        severity_score = calculate_severity_score(
            user_id, 
            total_predictions=total_predictions,
            risk_level=risk_level,
            risk_count=risk_count
        )
        
        return {
            'total_predictions': total_predictions,
            'last_disease': last_disease,
            'last_check': last_check,
            'risk_level': risk_level,
            'risk_count': risk_count,
            'severity_score': severity_score
        }
    except Exception as e:
        print(f"Error getting dashboard stats: {e}")
        return {
            'total_predictions': 0,
            'last_disease': 'None',
            'last_check': 'Never',
            'risk_level': 'Low',
            'risk_count': 0,
            'severity_score': 0
        }

def get_disease_frequency(user_id):
    """Calculate disease frequency - for insights"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT disease FROM predictions WHERE user_id=?", (user_id,))
        diseases = [row[0] for row in c.fetchall()]
        conn.close()
        
        frequency = {}
        for disease in diseases:
            frequency[disease] = frequency.get(disease, 0) + 1
        
        return frequency
    except Exception as e:
        print(f"Error calculating disease frequency: {e}")
        return {}

def calculate_risk_level(user_id, frequency=None):
    """Calculate risk level based on prediction frequency
    
    Args:
        user_id: The user ID
        frequency: Optional pre-computed frequency dict to avoid redundant calls
    """
    try:
        # Use provided frequency or calculate it
        if frequency is None:
            frequency = get_disease_frequency(user_id)
        
        if not frequency:
            return "Low", 0
        
        # Get max frequency
        max_count = max(frequency.values())
        
        # Risk calculation logic
        if max_count >= 3:
            return "High", max_count
        elif max_count == 2:
            return "Medium", max_count
        else:
            return "Low", max_count
    except Exception as e:
        print(f"Error calculating risk: {e}")
        return "Low", 0


def calculate_severity_score(user_id, total_predictions=None, risk_level=None, risk_count=None):
    """Calculate numeric severity score (0-100) based on risk and history
    
    Args:
        user_id: The user ID
        total_predictions: Optional pre-computed total (avoids redundant DB call)
        risk_level: Optional pre-computed risk level
        risk_count: Optional pre-computed risk count
    """
    try:
        # Avoid recursion: Get total from DB only if not provided
        if total_predictions is None:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM predictions WHERE user_id=?", (user_id,))
            total_predictions = c.fetchone()[0]
            conn.close()
        
        if total_predictions == 0:
            return 0

        # Use provided risk info or calculate it
        if risk_level is None or risk_count is None:
            frequency = get_disease_frequency(user_id)
            risk_level, risk_count = calculate_risk_level(user_id, frequency)

        # Severity score mapping:
        # High risk => 80-100, Medium => 40-79, Low => 0-39 (scaled by frequency share)
        base = 0
        if risk_level == 'High':
            base = 80
        elif risk_level == 'Medium':
            base = 50
        else:
            base = 25

        # Add a proportional component based on the most frequent disease share
        severity = base + int((risk_count / total_predictions) * 20)
        return min(100, max(0, severity))
    except Exception as e:
        print(f"Error calculating severity score: {e}")
        return 0


def get_symptom_frequency(user_id):
    """Calculate most frequent symptoms"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT symptoms FROM predictions WHERE user_id=?", (user_id,))
        symptoms_list = [row[0] for row in c.fetchall()]
        conn.close()
        
        symptom_count = {}
        for symptoms_str in symptoms_list:
            symptoms = [s.strip() for s in symptoms_str.split(',')]
            for symptom in symptoms:
                symptom_count[symptom] = symptom_count.get(symptom, 0) + 1
        
        # Return top 5 symptoms
        sorted_symptoms = sorted(symptom_count.items(), key=lambda x: x[1], reverse=True)[:5]
        return sorted_symptoms
    except Exception as e:
        print(f"Error calculating symptom frequency: {e}")
        return []

def get_health_trends(user_id, days=7):
    """Get health trends for the last N days"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Get predictions from last N days
        c.execute(
            "SELECT date_time, disease FROM predictions WHERE user_id=? ORDER BY date_time DESC LIMIT 30",
            (user_id,)
        )
        predictions = c.fetchall()
        conn.close()
        
        # Group by date
        trend = {}
        for date_time, disease in predictions:
            date = date_time[:10] if date_time else "Unknown"
            trend[date] = trend.get(date, 0) + 1
        
        # Sort by date
        sorted_trend = sorted(trend.items())
        return sorted_trend[-days:] if len(sorted_trend) > days else sorted_trend
    except Exception as e:
        print(f"Error getting trends: {e}")
        return []

def generate_ai_insights(user_id):
    """Generate AI-powered health insights personalized to user"""
    insights = []
    
    try:
        # Get user name for personalization
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT name FROM users WHERE id=?", (user_id,))
        user_row = c.fetchone()
        user_name = user_row[0] if user_row else "User"
        conn.close()
        
        # 1. Symptom-based insights
        top_symptoms = get_symptom_frequency(user_id)
        if top_symptoms:
            primary_symptom = top_symptoms[0][0]
            if 'fever' in primary_symptom.lower():
                insights.append({
                    'type': 'symptom',
                    'icon': '🌡️',
                    'title': 'Frequent Fever Alert',
                    'message': f'{user_name}, you frequently report {primary_symptom.replace("_", " ")}. Stay hydrated and monitor your temperature regularly. Consult a doctor if fever persists.',
                    'priority': 'high'
                })
            elif 'cough' in primary_symptom.lower():
                insights.append({
                    'type': 'symptom',
                    'icon': '🫁',
                    'title': 'Cough Monitoring',
                    'message': f'Your health pattern shows frequent {primary_symptom.replace("_", " ")}. Try steam inhalation and honey-based remedies. Track duration.',
                    'priority': 'medium'
                })
            elif 'headache' in primary_symptom.lower():
                insights.append({
                    'type': 'symptom',
                    'icon': '🧠',
                    'title': 'Recurring Headaches',
                    'message': f'You experience {primary_symptom.replace("_", " ")} frequently. Ensure 7-8 hours sleep, stay hydrated, and reduce screen time.',
                    'priority': 'medium'
                })
        
        # 2. Disease frequency insights
        disease_freq = get_disease_frequency(user_id)
        if disease_freq:
            most_common_disease = max(disease_freq, key=disease_freq.get)
            count = disease_freq[most_common_disease]
            
            if count >= 2:
                insights.append({
                    'type': 'disease',
                    'icon': '⚕️',
                    'title': f'Recurring {most_common_disease}',
                    'message': f'Your records show {most_common_disease} has appeared {count} times. We recommend scheduling a doctor\'s appointment for proper diagnosis.',
                    'priority': 'high' if count >= 3 else 'medium'
                })
        
        # 3. Risk-based insights
        risk_level, risk_count = calculate_risk_level(user_id)
        if risk_level == "High":
            insights.append({
                'type': 'risk',
                'icon': '⚠️',
                'title': 'High Health Risk Alert',
                'message': f'{user_name}, your health patterns show elevated risk. Schedule a comprehensive health checkup with your doctor immediately.',
                'priority': 'high'
            })
        elif risk_level == "Medium":
            insights.append({
                'type': 'risk',
                'icon': '📋',
                'title': 'Monitor Your Health',
                'message': f'You\'ve had {risk_count} predictions recently. Keep track of patterns and consult a doctor if symptoms persist.',
                'priority': 'medium'
            })
        
        # 4. General wellness insights based on activity
        total = len(disease_freq)
        if total > 0:
            insights.append({
                'type': 'wellness',
                'icon': '💪',
                'title': 'Boost Your Immunity',
                'message': 'Regular exercise (30 mins daily), balanced nutrition, and 7-8 hours of quality sleep strengthen your immune system.',
                'priority': 'low'
            })
        
            insights.append({
                'type': 'wellness',
                'icon': '💧',
                'title': 'Stay Hydrated',
                'message': 'Drink 8-10 glasses of water daily. Proper hydration supports immune function and helps prevent many illnesses.',
                'priority': 'low'
            })
        
        return insights[:4]  # Return top 4 insights
    
    except Exception as e:
        print(f"Error generating insights: {e}")
        return []


# ---------------- AI-DOCTOR ------------------ #
@app.route("/ai-doctor")
@login_required
def ai_doctor():
    return render_template("ai_doctor.html")

@app.route("/ask", methods=["POST"])
def ask_ai():
    user_input = request.json.get("message", "")
    prompt = f"""
You are my personal AI doctor and medical assistant. 
I am your patient. I will describe my symptoms or health issues, and you will respond in a friendly, professional, and caring doctor style. 
Give clear, dynamic, and unique advice every time. 
Ask follow-up questions only if needed to better understand my symptoms.
Patient's input: {user_input}
"""



    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openrouter/free",   # Free model
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        ai_response = response.json()
        reply = ai_response["choices"][0]["message"]["content"]
    except Exception as e:
        reply = f"Error contacting AI: {str(e)}"

    return jsonify({"reply": reply})


# creating routes========================================
@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    # simple password reset flow: user submits email + new password
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm = request.form.get('confirm')
        if password != confirm:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('forgot_password'))
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE email=?", (email,))
        row = c.fetchone()
        if not row:
            conn.close()
            flash('Email not registered', 'danger')
            return redirect(url_for('forgot_password'))
        hashed = generate_password_hash(password)
        c.execute("UPDATE users SET password=? WHERE email=?", (hashed, email))
        conn.commit()
        conn.close()
        flash('Password updated successfully, please login', 'success')
        return redirect(url_for('login'))
    # GET request
    return render_template('forgot_password.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    # already logged in
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, password, name FROM users WHERE email=?", (email,))
        row = c.fetchone()
        conn.close()
        if row and check_password_hash(row[1], password):
            session['user_id'] = row[0]
            session['user_name'] = row[2]
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'danger')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm = request.form.get('confirm')
        if password != confirm:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('register'))
        hashed = generate_password_hash(password)
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                      (name, email, hashed))
            conn.commit()
            conn.close()
            flash('Registration successful, please login', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'info')
    return redirect(url_for('login'))

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route('/dashboard')
@login_required
def dashboard():
    stats = get_dashboard_stats(session['user_id'])
    predictions = get_user_predictions(session['user_id'], 5)
    return render_template('dashboard.html', stats=stats, predictions=predictions, user_id=session['user_id'])

@app.route('/api/dashboard-stats')
@login_required
def api_dashboard_stats():
    """API endpoint for dashboard data with personalized user insights
    
    Optimized to avoid redundant function calls and recursion.
    """
    try:
        user_id = session['user_id']
        
        # Get all data in one pass - Avoid redundant calls
        stats = get_dashboard_stats(user_id)  # This now internally handles everything without recursion
        predictions = get_user_predictions(user_id, 10)
        all_predictions = get_user_predictions(user_id, 100)  # Get full history
        disease_frequency = get_disease_frequency(user_id)
        trends = get_health_trends(user_id, 7)
        ai_insights = generate_ai_insights(user_id)
        top_symptoms = get_symptom_frequency(user_id)
        
        # Reuse stats instead of recalculating
        risk_level = stats.get('risk_level', 'Low')
        risk_count = stats.get('risk_count', 0)
        severity_score = stats.get('severity_score', 0)
        
        # Get last prediction details
        last_prediction = None
        if predictions:
            disease, symptoms, date_time = predictions[0]
            last_prediction = {
                'disease': disease,
                'symptoms': symptoms,
                'date_time': date_time,
                'severity': 'Medium'  # Default, can be calculated if needed
            }
        
        # Format predictions for chart
        prediction_list = []
        for idx, (disease, symptoms, date_time) in enumerate(predictions):
            prediction_list.append({
                'date': date_time[:10] if date_time else 'Unknown',
                'disease': disease,
                'symptoms_count': len(symptoms.split(',')),
                'index': idx + 1
            })
        
        # Format all predictions for history table
        history_list = []
        for disease, symptoms, date_time in all_predictions:
            history_list.append({
                'disease': disease,
                'symptoms': symptoms,
                'date_time': date_time,
                'severity': 'Medium'  # Can be enhanced
            })
        
        # Format trends
        trend_data = []
        for date, count in trends:
            trend_data.append({'date': date, 'count': count})
        
        # Format disease frequency for pie chart
        disease_data = []
        for disease, count in sorted(disease_frequency.items(), key=lambda x: x[1], reverse=True)[:6]:
            disease_data.append({'disease': disease, 'count': count})
        
        # Format top symptoms
        symptoms_data = [
            {'symptom': s[0].replace('_', ' ').title(), 'count': s[1]} 
            for s in top_symptoms
        ]
        
        response = {
            'total_predictions': stats['total_predictions'],
            'last_disease': stats['last_disease'],
            'last_check': stats['last_check'],
            'last_prediction': last_prediction,
            'recent_predictions': prediction_list,
            'all_predictions': history_list,
            'risk_level': risk_level,
            'risk_count': risk_count,
            'severity_score': severity_score,
            'disease_frequency': disease_data,
            'trends': trend_data,
            'ai_insights': ai_insights,
            'top_symptoms': symptoms_data
        }
        
        return jsonify(response)
    
    except RecursionError as e:
        print(f"Recursion error in dashboard stats: {e}")
        return jsonify({
            'error': 'Recursion error',
            'message': 'Failed to calculate dashboard stats due to recursion',
            'total_predictions': 0
        }), 500
    
    except Exception as e:
        print(f"Error in api_dashboard_stats: {e}")
        return jsonify({
            'error': str(type(e).__name__),
            'message': str(e),
            'total_predictions': 0
        }), 500


@app.route("/")
def front():
    # front page removed - redirect straight to login
    return redirect(url_for('login'))


@app.route('/health')
def health_check():
    """Health-check endpoint for readiness checks."""
    status = {
        'status': 'ok',
        'model_loaded': svc is not None,
        'datasets_loaded': bool(not description.empty and not precautions.empty)
    }
    return jsonify(status)

@app.route("/healthcare")
@login_required
def index():
    return render_template("index.html")


@app.route('/api/history/<int:userId>', methods=['GET'])
@login_required
def api_history(userId):
    """Return user's past reports (latest first). Requires authenticated user.

    Only the logged-in user may fetch their own history.
    """
    try:
        # Ensure users can only fetch their own history
        if session.get('user_id') != userId:
            return jsonify({'error': 'forbidden'}), 403

        rows = get_reports(userId)
        reports = []
        for disease, severity, date in rows:
            reports.append({'disease': disease, 'severity': severity, 'date': date})

        return jsonify({'reports': reports})
    except Exception as e:
        print(f"Error in api_history: {e}")
        return jsonify({'error': 'internal_error', 'message': str(e)}), 500

# Define a route for the home page
@app.route('/predict', methods=['GET', 'POST'])
@login_required
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        print(symptoms)
        if symptoms =="Symptoms":
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template('index.html', message=message)
        else:
            # Split the user's input into a list of symptoms (assuming they are comma-separated)
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            # Remove any extra characters, if any
            user_symptoms = [s.strip().lower().replace(" ", "_") for s in symptoms.split(',')]
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

            my_precautions = []
            for i in precautions[0]:
                my_precautions.append(i)

            # Save prediction to database
            save_prediction(session['user_id'], predicted_disease, symptoms)

            return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des,
                                   my_precautions=my_precautions, medications=medications, my_diet=rec_diet,
                                   workout=workout)

    return render_template('index.html')



# about view funtion and path
@app.route('/about')
@login_required
def about():
    return render_template("about.html")
# contact view funtion and path
@app.route('/contact')
@login_required
def contact():
    return render_template("contact.html")

# developer view funtion and path
@app.route('/developer')
@login_required
def developer():
    return render_template("developer.html")

# about view funtion and path
@app.route('/blog')
@login_required
def blog():
    return render_template("blog.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)