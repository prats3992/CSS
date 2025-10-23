# Reddit API Setup Guide for Pre-eclampsia Analysis Project

## Overview
This guide will help you create a Reddit application and obtain API credentials needed for the pre-eclampsia Reddit analysis project. You'll need these credentials to collect data from Reddit using PRAW (Python Reddit API Wrapper).

## Prerequisites
- Reddit account (create one at reddit.com if you don't have one)
- Verified email address on your Reddit account

## Step 1: Access Reddit Developer Portal

1. **Go to Reddit Apps Page**
   - Visit: https://www.reddit.com/prefs/apps
   - Make sure you're logged into your Reddit account

2. **Scroll to the bottom and click "Create App" or "Create Another App"**

## Step 2: Fill Out Application Form

Based on your project needs, here are the recommended settings:

### Application Type Selection
- **Choose "script"** - This is perfect for your research project since it's for personal use and data collection

### Application Details

**Name:** 
```
css
```
✅ This matches what you specified

**Application Type:**
- ✅ **script** (Select this option)
- ❌ web app (Don't select this)
- ❌ installed app (Don't select this)

**Description:**
```
Academic research project analyzing pre-eclampsia discussions on Reddit using NLP techniques including topic modeling and sentiment analysis. Data collection for CSS (Computational Social Science) course project.
```

**About URL:** (Optional)
```
Leave this blank or use: https://github.com/yourusername/pre-eclampsia-analysis
```

**Redirect URI:**
```
http://localhost:8080
```
(This is required but won't be used since you're creating a script application)

## Step 3: Create the Application

1. **Review your settings** - Make sure you selected "script"
2. **Click "Create app"**
3. **Reddit will create your application**

## Step 4: Get Your API Credentials

After creating the app, you'll see your application listed. Note down these important values:

### Client ID
- **Location:** Listed under your app name (short string of characters)
- **Format:** Usually 14 characters, looks like: `dJ2V7w8x9y2z3A`

### Client Secret  
- **Location:** Listed as "secret" 
- **Format:** Longer string, looks like: `xY9-wE2rT5yU8iO3pA7sD4fG6hJ1kL0`

## Step 5: Update Your Config File

Now update your `config.py` file with the credentials:

```python
# Replace these values in config.py
REDDIT_CLIENT_ID = 'your_actual_client_id_here'  # The 14-character ID
REDDIT_CLIENT_SECRET = 'your_actual_client_secret_here'  # The longer secret
REDDIT_USER_AGENT = 'css-preeclampsia-research/1.0 by yourusername'  # Update with your Reddit username
```

### Environment Variables (Recommended)
For better security, create a `.env` file in your project directory:

```env
REDDIT_CLIENT_ID=your_actual_client_id_here
REDDIT_CLIENT_SECRET=your_actual_client_secret_here
```

## Step 6: Test Your API Connection

Create a simple test to verify your credentials work:

```python
import praw
from config import Config

# Test Reddit connection
reddit = praw.Reddit(
    client_id=Config.REDDIT_CLIENT_ID,
    client_secret=Config.REDDIT_CLIENT_SECRET,
    user_agent=Config.REDDIT_USER_AGENT
)

# Test the connection
try:
    print("Testing Reddit API connection...")
    print(f"Read-only mode: {reddit.read_only}")
    print("✅ Successfully connected to Reddit API!")
    
    # Test a simple subreddit access
    subreddit = reddit.subreddit("pregnancy")
    print(f"✅ Successfully accessed r/{subreddit.display_name}")
    
except Exception as e:
    print(f"❌ Error connecting to Reddit API: {e}")
```

## Step 7: Verify Setup

Run your test script to make sure everything works:

```bash
python test_reddit_connection.py
```

You should see:
```
Testing Reddit API connection...
Read-only mode: True
✅ Successfully connected to Reddit API!
✅ Successfully accessed r/pregnancy
```

## Common Issues and Solutions

### Issue: "Invalid credentials"
- **Solution:** Double-check your Client ID and Client Secret
- Make sure there are no extra spaces or characters

### Issue: "Too Many Requests"  
- **Solution:** Reddit has rate limits (60 requests per minute)
- Add delays between requests in your data collection script

### Issue: "User-Agent required"
- **Solution:** Make sure your user_agent string is descriptive and unique
- Format: `appname/version by username`

## Rate Limits and Best Practices

### Reddit API Limits
- **60 requests per minute** for OAuth applications
- **600 requests per 10 minutes** 
- Respect these limits to avoid being blocked

### Best Practices
1. **Use descriptive User-Agent string**
2. **Add delays between requests** (1-2 seconds recommended)
3. **Handle errors gracefully** with try-catch blocks
4. **Cache results** to avoid repeated requests
5. **Respect subreddit rules** and Reddit's content policy

## Data Collection Ethics

### Important Guidelines
- ✅ Use only public posts and comments
- ✅ Anonymize all personal information
- ✅ Remove medical identifiers
- ✅ Aggregate data for analysis
- ❌ Don't share individual posts/comments
- ❌ Don't attempt to identify users
- ❌ Don't scrape private subreddits

## Next Steps

Once your Reddit API is set up:

1. **Test data collection** with a small sample
2. **Run the preprocessing pipeline** 
3. **Verify text cleaning** is working properly
4. **Start with topic modeling** on sample data
5. **Scale up** to full data collection

## Troubleshooting

If you encounter issues:

1. **Check Reddit Developer Documentation:** https://www.reddit.com/dev/api/
2. **PRAW Documentation:** https://praw.readthedocs.io/
3. **Verify your Reddit account** is in good standing
4. **Make sure your app type is "script"** not web app

## Security Notes

- **Never commit API keys** to version control
- **Use environment variables** for credentials  
- **Don't share your Client Secret** with others
- **Regenerate keys** if they're compromised

## Contact

If you need help with the setup, refer to:
- Reddit API documentation
- PRAW documentation  
- Course materials and TAs
- Project team members

---

**Remember:** This setup is specifically for academic research purposes. Make sure to follow your institution's IRB guidelines for social media research if required.