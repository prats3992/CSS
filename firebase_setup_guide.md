# Firebase Setup Guide for Pre-eclampsia Reddit Analysis

## Overview
This guide will help you set up Firebase Realtime Database for the pre-eclampsia Reddit analysis project. Firebase provides a cloud-based NoSQL database that's perfect for team collaboration.

## Prerequisites
- Google account
- Firebase project (we'll create this)
- Python environment with required packages

## Step 1: Create Firebase Project

1. **Go to Firebase Console**
   - Visit [https://console.firebase.google.com/](https://console.firebase.google.com/)
   - Sign in with your Google account

2. **Create New Project**
   - Click "Create a project"
   - Enter project name (e.g., "pre-eclampsia-analysis")
   - Choose whether to enable Google Analytics (optional)
   - Click "Create project"

## Step 2: Set Up Realtime Database

1. **Navigate to Database**
   - In the Firebase console, go to "Realtime Database" in the left sidebar
   - Click "Create Database"

2. **Choose Security Rules**
   - Start in "locked mode" for security
   - We'll configure rules later

3. **Select Database Location**
   - Choose a location close to your team
   - Note the database URL (e.g., `https://your-project-default-rtdb.firebaseio.com/`)

## Step 3: Configure Database Security Rules

1. **Go to Rules Tab**
   - In Realtime Database, click on "Rules" tab

2. **Set Development Rules** (for initial development):
```json
{
  "rules": {
    ".read": "auth != null",
    ".write": "auth != null"
  }
}
```

3. **Production Rules** (more secure for actual research):
```json
{
  "rules": {
    "posts": {
      ".read": "auth != null",
      ".write": "auth != null"
    },
    "processed_data": {
      ".read": "auth != null", 
      ".write": "auth != null"
    },
    "analysis_results": {
      ".read": "auth != null",
      ".write": "auth != null"
    }
  }
}
```

## Step 4: Get Firebase Configuration

### For Admin SDK (Server-side)

1. **Create Service Account**
   - Go to Project Settings (gear icon) → Service Accounts
   - Click "Generate new private key"
   - Download the JSON file
   - Rename it to `firebase-service-account.json`
   - Place it in your project root directory

2. **Note Database URL**
   - Copy the database URL from the Realtime Database page

### For Web SDK (Client-side, optional)

1. **Get Web App Config**
   - Go to Project Settings → General
   - In "Your apps" section, click "Add app" → Web
   - Register the app and copy the config object

## Step 5: Environment Variables Setup

Create a `.env` file in your project root:

```env
# Reddit API
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret

# Firebase Admin SDK
FIREBASE_SERVICE_ACCOUNT_PATH=firebase-service-account.json
FIREBASE_DATABASE_URL=https://your-project-default-rtdb.firebaseio.com/

# Firebase Web SDK (optional)
FIREBASE_API_KEY=your_api_key
FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_STORAGE_BUCKET=your-project.appspot.com
FIREBASE_MESSAGING_SENDER_ID=123456789
FIREBASE_APP_ID=1:123456789:web:abcdef123456
```

## Step 6: Update Config Files

1. **Update `config.py`**
   - Replace placeholder values with your actual Firebase configuration
   - Set the correct database URL

2. **Test Connection**
   ```python
   from firebase_config import FirebaseConfig

   config = FirebaseConfig()
   config.initialize_firebase()
   if config.test_connection():
       print("✅ Firebase connected successfully!")
   else:
       print("❌ Firebase connection failed")
   ```

## Step 7: Database Structure

The application will create this structure in your Firebase Realtime Database:

```
your-database/
├── posts/
│   ├── post_id_1/
│   │   ├── title: "Post title"
│   │   ├── selftext: "Post content"
│   │   ├── subreddit: "subreddit_name"
│   │   ├── created_utc: "2025-01-01T00:00:00"
│   │   └── comments/
│   │       ├── comment_0/
│   │       └── comment_1/
│   └── post_id_2/...
├── processed_data/
│   ├── post_id_1/
│   │   ├── tokens_lemmatized: [...]
│   │   ├── sentiment_compound: 0.5
│   │   └── medical_terms: {...}
│   └── ...
└── analysis_results/
    ├── topic_modeling/
    │   ├── topics: [...]
    │   ├── coherence_score: 0.8
    │   └── timestamp: "2025-01-01T12:00:00"
    └── sentiment_analysis/
        ├── overall_sentiment: 0.2
        ├── sentiment_distribution: {...}
        └── timestamp: "2025-01-01T12:00:00"
```

## Step 8: Team Access Management

### Add Team Members

1. **Go to Project Settings**
   - Click on "Users and permissions"

2. **Add Members**
   - Click "Add member"
   - Enter team member's Google email
   - Assign appropriate role:
     - **Editor**: Can read/write data and modify settings
     - **Viewer**: Can only read data
     - **Owner**: Full access (be careful with this)

### Authentication Setup (Optional)

If you want user authentication:

1. **Enable Authentication**
   - Go to Authentication → Sign-in method
   - Enable desired sign-in providers (Email/Password, Google, etc.)

2. **Update Security Rules**
   ```json
   {
     "rules": {
       ".read": "auth != null",
       ".write": "auth != null"
     }
   }
   ```

## Step 9: Testing the Setup

Run this test script to verify everything works:

```python
# test_firebase.py
from firebase_config import FirebaseConfig, FirebaseDataManager
import pandas as pd
from datetime import datetime

# Test Firebase connection
config = FirebaseConfig()
config.initialize_firebase()

if config.test_connection():
    print("✅ Firebase connection successful!")

    # Test data operations
    manager = FirebaseDataManager(config)

    # Test saving data
    test_data = pd.DataFrame([{
        'post_id': 'test_post',
        'title': 'Test Post',
        'created_utc': datetime.now()
    }])

    manager.save_posts(test_data)
    print("✅ Data save successful!")

    # Test loading data
    loaded_data = manager.load_posts()
    print(f"✅ Data load successful! Found {len(loaded_data)} posts")

    # Clean up test data
    manager.clear_data('posts')
    print("✅ Cleanup successful!")

else:
    print("❌ Firebase connection failed!")
```

## Step 10: Best Practices

### Security
- Never commit `firebase-service-account.json` to version control
- Use environment variables for sensitive configuration
- Regularly review and update database security rules
- Monitor database access in Firebase console

### Data Management
- Use batch operations for large datasets
- Implement proper error handling
- Keep local backups of important data
- Monitor database usage and costs

### Team Collaboration
- Use consistent data structures
- Document any schema changes
- Set up proper access permissions
- Use Firebase console for monitoring and debugging

## Troubleshooting

### Common Issues

1. **"Permission denied" errors**
   - Check database security rules
   - Verify user authentication
   - Ensure service account has proper permissions

2. **"Service account file not found"**
   - Verify the path to `firebase-service-account.json`
   - Ensure the file is in the correct location

3. **Import errors**
   - Install Firebase packages: `pip install firebase-admin pyrebase4`
   - Check Python version compatibility

4. **Connection timeouts**
   - Check internet connection
   - Verify Firebase project is active
   - Try different database region

### Getting Help

- Firebase Documentation: [https://firebase.google.com/docs](https://firebase.google.com/docs)
- Stack Overflow: Search for "firebase python" issues
- Firebase Support: Available in Firebase console

## Cost Considerations

Firebase Realtime Database pricing:
- **Spark Plan (Free)**: 1GB storage, 10GB/month bandwidth
- **Blaze Plan (Pay-as-you-go)**: $5/GB storage, $1/GB bandwidth

For typical research projects with thousands of posts, the free tier should be sufficient.

## Summary

After completing this setup:
1. ✅ Firebase project created
2. ✅ Realtime Database configured
3. ✅ Service account set up
4. ✅ Environment variables configured
5. ✅ Team members added
6. ✅ Connection tested

You're now ready to run the pre-eclampsia analysis with cloud database storage!
