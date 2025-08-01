#!/usr/bin/env python3
"""
Google Drive Setup Helper
Helps set up Google Drive integration for Railway deployment
"""

import os
import json
import logging
import base64

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()  # Load environment variables from .env

def check_environment():
    """Check environment configuration"""
    logger.info("🔧 Checking environment configuration...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check Google Drive settings
    use_drive = os.getenv('USE_GOOGLE_DRIVE', 'false').lower() == 'true'
    folder_id = os.getenv('GOOGLE_DRIVE_FOLDER_ID', '')
    
    logger.info(f"☁️ Google Drive enabled: {use_drive}")
    logger.info(f"📁 Folder ID: {folder_id}")
    
    if not use_drive:
        logger.warning("⚠️ Google Drive is disabled in .env file")
        logger.info("   Set USE_GOOGLE_DRIVE=true to enable")
        return False
    
    if not folder_id:
        logger.error("❌ GOOGLE_DRIVE_FOLDER_ID not set in .env file")
        return False
    
    return True

def check_service_account_key():
    """Check if service account key exists either as a file or in environment variable"""
    logger.info("🔑 Checking service account key...")

    key_path = "secrets/service_account.json"
    key_data = None

    # Try reading from local file
    if os.path.exists(key_path):
        logger.info(f"✅ Service account key found: {key_path}")
        try:
            with open(key_path, 'r') as f:
                key_data = json.load(f)
        except json.JSONDecodeError:
            logger.error("❌ Service account key file is not valid JSON")
            return False
        except Exception as e:
            logger.error(f"❌ Error reading service account key file: {e}")
            return False
    else:
        logger.warning(f"⚠️ Service account key not found at: {key_path}")
        logger.info("🔎 Trying to load from environment variable: GOOGLE_SERVICE_ACCOUNT_JSON")
        encoded_key = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        if not encoded_key:
            logger.error("❌ GOOGLE_SERVICE_ACCOUNT_JSON not found in environment variables.")
            _print_setup_instructions(key_path)
            return False
        try:
            decoded = base64.b64decode(encoded_key).decode("utf-8")
            key_data = json.loads(decoded)
            logger.info("✅ Service account key loaded from environment variable.")
        except (base64.binascii.Error, UnicodeDecodeError):
            logger.error("❌ Failed to decode base64 service account key from environment.")
            return False
        except json.JSONDecodeError:
            logger.error("❌ Decoded service account key is not valid JSON.")
            return False

    # Validate key data
    required_fields = ['type', 'project_id', 'private_key_id', 'private_key', 'client_email']
    missing_fields = [field for field in required_fields if field not in key_data]

    if missing_fields:
        logger.error(f"❌ Invalid service account key. Missing fields: {missing_fields}")
        return False

    logger.info(f"✅ Service account: {key_data.get('client_email', 'Unknown')}")
    logger.info(f"✅ Project: {key_data.get('project_id', 'Unknown')}")
    return True

def _print_setup_instructions(key_path):
    logger.info("\n📋 To set up Google Drive integration:")
    logger.info("1. Go to Google Cloud Console (https://console.cloud.google.com/)")
    logger.info("2. Create a new project or select existing one")
    logger.info("3. Enable Google Drive API")
    logger.info("4. Create a Service Account:")
    logger.info("   - Go to IAM & Admin > Service Accounts")
    logger.info("   - Click 'Create Service Account'")
    logger.info("   - Give it a name (e.g., 'money-printer-drive')")
    logger.info("   - Click 'Create and Continue'")
    logger.info("   - Skip role assignment (click 'Continue')")
    logger.info("   - Click 'Done'")
    logger.info("5. Create and download key:")
    logger.info("   - Click on the service account you created")
    logger.info("   - Go to 'Keys' tab")
    logger.info("   - Click 'Add Key' > 'Create new key'")
    logger.info("   - Choose JSON format")
    logger.info("   - Download the file")
    logger.info(f"6. Place the downloaded file at: {key_path}")
    logger.info("   OR encode it as base64 and set it as the env var: GOOGLE_SERVICE_ACCOUNT_JSON")
    logger.info("7. Share your Google Drive folder with the service account email")
    logger.info("   - Right-click on your folder in Google Drive")
    logger.info("   - Click 'Share'")
    logger.info("   - Add the service account email with Editor permissions")

def create_secrets_directory():
    """Create secrets directory if it doesn't exist"""
    secrets_dir = "secrets"
    
    if not os.path.exists(secrets_dir):
        os.makedirs(secrets_dir, exist_ok=True)
        logger.info(f"📁 Created secrets directory: {secrets_dir}")
        
        # Create .gitignore to prevent committing secrets
        gitignore_path = os.path.join(secrets_dir, ".gitignore")
        with open(gitignore_path, 'w') as f:
            f.write("# Ignore all files in secrets directory\n")
            f.write("*\n")
            f.write("!.gitignore\n")
        
        logger.info(f"🔒 Created .gitignore in secrets directory")
    else:
        logger.info(f"✅ Secrets directory exists: {secrets_dir}")

def test_basic_drive_access():
    """Test basic Google Drive access without service account"""
    logger.info("🔗 Testing basic Drive API access...")
    
    try:
        # Test if we can import the required modules
        from googleapiclient.discovery import build
        logger.info("✅ Google API client library available")
        return True
    except ImportError as e:
        logger.error(f"❌ Google API client library not available: {e}")
        logger.info("💡 Install with: pip install google-api-python-client google-auth")
        return False

def show_railway_deployment_notes():
    """Show notes specific to Railway deployment"""
    logger.info("\n🚀 Railway Deployment Notes:")
    logger.info("=" * 50)
    logger.info("For Railway deployment, you need to:")
    logger.info("1. ✅ Add service account JSON as Railway environment variable")
    logger.info("   - Go to your Railway project")
    logger.info("   - Go to Variables tab")
    logger.info("   - Add variable: GOOGLE_SERVICE_ACCOUNT_JSON")
    logger.info("   - Paste the entire JSON content as the value")
    logger.info("2. ✅ Update the drive manager to read from environment variable")
    logger.info("3. ✅ Set USE_GOOGLE_DRIVE=true in Railway variables")
    logger.info("4. ✅ Set GOOGLE_DRIVE_FOLDER_ID in Railway variables")
    logger.info("\n⚠️ DO NOT commit the service account JSON file to git!")

def main():
    """Main setup function"""
    logger.info("🏁 Google Drive Setup Helper")
    logger.info("🚀 Railway Deployment Configuration")
    logger.info("=" * 60)
    
    # Create secrets directory
    create_secrets_directory()
    
    # Check environment
    env_ok = check_environment()
    
    # Test API access
    api_ok = test_basic_drive_access()
    
    # Check service account key
    key_ok = check_service_account_key()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Setup Status Summary:")
    logger.info(f"  🔧 Environment Config: {'✅ OK' if env_ok else '❌ NEEDS SETUP'}")
    logger.info(f"  📚 API Libraries: {'✅ OK' if api_ok else '❌ NEEDS INSTALL'}")
    logger.info(f"  🔑 Service Account: {'✅ OK' if key_ok else '❌ NEEDS SETUP'}")
    
    if env_ok and api_ok and key_ok:
        logger.info("\n🎉 Google Drive integration is ready!")
        logger.info("You can now run data collection with Drive storage.")
    else:
        logger.warning("\n⚠️ Google Drive integration needs setup.")
        logger.info("Follow the instructions above to complete setup.")
    
    # Show Railway notes
    show_railway_deployment_notes()

if __name__ == "__main__":
    main()
