"""Seed demo user and sample predictions for dashboard demo.

Run:
    python scripts/seed_demo.py

This will create a demo user (demo@local / demo) and insert a few sample predictions.
"""
import os
import sqlite3
from werkzeug.security import generate_password_hash
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(__file__)) if os.path.basename(os.getcwd()) != 'scripts' else os.path.dirname(__file__)
DB_PATH = os.path.join(os.path.dirname(BASE_DIR), 'users.db') if False else os.path.join(os.path.dirname(__file__), '..', 'users.db')
# Resolve to project root reliably
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DB_PATH = os.path.join(PROJECT_ROOT, 'users.db')

print(f"Using DB: {DB_PATH}")

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Ensure tables exist
c.execute('''CREATE TABLE IF NOT EXISTS users (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               name TEXT NOT NULL,
               email TEXT NOT NULL UNIQUE,
               password TEXT NOT NULL
           )''')

c.execute('''CREATE TABLE IF NOT EXISTS predictions (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               user_id INTEGER NOT NULL,
               disease TEXT NOT NULL,
               symptoms TEXT NOT NULL,
               date_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
               severity TEXT DEFAULT 'Medium',
               FOREIGN KEY (user_id) REFERENCES users(id)
           )''')
conn.commit()

# Create demo user if not exists
email = 'demo@local'
c.execute('SELECT id FROM users WHERE email=?', (email,))
row = c.fetchone()
if row:
    user_id = row[0]
    print('Demo user already exists, id=', user_id)
else:
    hashed = generate_password_hash('demo')
    c.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)', ('Demo User', email, hashed))
    conn.commit()
    user_id = c.lastrowid
    print('Created demo user id=', user_id)

# Insert sample predictions for last 7 days
sample_diseases = ['Common Cold', 'Migraine', 'Hypertension ', 'Common Cold', 'Bronchial Asthma']
sample_symptoms = ['cough, fever', 'headache, nausea', 'high_blood_pressure', 'cough, sore throat', 'wheezing, breathlessness']
now = datetime.now()

# Clean previous demo predictions
c.execute('DELETE FROM predictions WHERE user_id=?', (user_id,))
conn.commit()

for i, (d, s) in enumerate(zip(sample_diseases, sample_symptoms)):
    dt = now - timedelta(days=(len(sample_diseases) - i))
    c.execute('INSERT INTO predictions (user_id, disease, symptoms, date_time, severity) VALUES (?, ?, ?, ?, ?)',
              (user_id, d, s, dt.strftime('%Y-%m-%d %H:%M:%S'), 'Medium'))

conn.commit()
print('Inserted sample predictions')
print('\nDemo credentials:')
print('  email: demo@local')
print('  password: demo')
print('\nRun the app and log in with the demo account to view the dashboard.')

conn.close()
