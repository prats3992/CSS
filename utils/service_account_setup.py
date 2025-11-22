"""
Helper script to set up Firebase service account credentials

Follow these steps:
1. Go to Firebase Console (https://console.firebase.google.com/)
2. Select your project: pre-eclampsia-analysis
3. Click the gear icon > Project settings
4. Go to "Service accounts" tab
5. Click "Generate new private key"
6. Save the JSON file as 'firebase-credentials.json' in this directory

Alternatively, you can use the Firebase Admin SDK with web credentials,
but a service account is recommended for server-side operations.
"""

import json
import os

def create_credentials_template():
    """Create a template for Firebase credentials"""
    template = {
        "type": "service_account",
        "project_id": "pre-eclampsia-analysis",
        "private_key_id": "YOUR_PRIVATE_KEY_ID",
        "private_key": "YOUR_PRIVATE_KEY",
        "client_email": "firebase-adminsdk@pre-eclampsia-analysis.iam.gserviceaccount.com",
        "client_id": "YOUR_CLIENT_ID",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "YOUR_CERT_URL"
    }
    
    with open('firebase-credentials-template.json', 'w') as f:
        json.dump(template, f, indent=2)
    
    print("✓ Created firebase-credentials-template.json")
    print("\nInstructions:")
    print("1. Download your service account JSON from Firebase Console")
    print("2. Save it as 'firebase-credentials.json' in this directory")
    print("3. Update firebase_manager.py to use: cred = credentials.Certificate('firebase-credentials.json')")

def check_credentials():
    """Check if credentials file exists"""
    if os.path.exists('firebase-credentials.json'):
        print("✓ firebase-credentials.json found")
        return True
    else:
        print("✗ firebase-credentials.json not found")
        print("Please download it from Firebase Console")
        return False

if __name__ == "__main__":
    print("Firebase Service Account Setup")
    print("="*50)
    
    if not check_credentials():
        create_credentials_template()
