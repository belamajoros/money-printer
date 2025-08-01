#!/usr/bin/env python3
"""
Quick Setup Helper for Enhanced Drive Integration
"""

import os
import json
import base64
from pathlib import Path
from dotenv import load_dotenv

def main():
    print("🔧 Enhanced Google Drive Setup Helper")
    print("=" * 50)

    secrets_dir = Path("secrets")
    service_account_path = secrets_dir / "service_account.json"

    print(f"\n📁 Checking secrets directory: {secrets_dir.absolute()}")
    if not secrets_dir.exists():
        secrets_dir.mkdir(parents=True)
        print("✅ Created secrets directory")

    print(f"\n📋 Service Account Key Location:")
    print(f"   Expected: {service_account_path.absolute()}")

    key_data = None

    if service_account_path.exists():
        print("✅ Service account key file found!")
        try:
            with open(service_account_path, 'r') as f:
                key_data = json.load(f)
        except Exception as e:
            print(f"❌ Error reading service account key: {e}")
    else:
        print("❌ Service account key file not found")
        print("🔎 Trying to load from .env as base64-encoded JSON...")

        load_dotenv()
        encoded_key = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        if encoded_key:
            try:
                decoded = base64.b64decode(encoded_key).decode("utf-8")
                key_data = json.loads(decoded)
                print("✅ Loaded service account key from environment variable!")
            except Exception as e:
                print(f"❌ Failed to decode or parse GOOGLE_SERVICE_ACCOUNT_JSON: {e}")
        else:
            print("❌ GOOGLE_SERVICE_ACCOUNT_JSON not found in .env")

    if key_data:
        if 'client_email' in key_data:
            print(f"📧 Service account email: {key_data['client_email']}")
            print("\n📂 Share your Google Drive folder with this email!")
        else:
            print("❌ Invalid service account key format (missing 'client_email')")
    else:
        print("\n💡 To fix this:")
        print(f"   1. Download your service account JSON key from Google Cloud Console")
        print(f"   2. Save it as: {service_account_path.absolute()}")
        print(f"   OR encode it with base64 and set it in your .env as GOOGLE_SERVICE_ACCOUNT_JSON")

    # Check .env configuration
    print(f"\n⚙️ Environment Configuration:")

    env_path = Path(".env")
    if env_path.exists():
        with open(env_path, 'r') as f:
            env_content = f.read()

        if 'USE_GOOGLE_DRIVE=true' in env_content:
            print("✅ USE_GOOGLE_DRIVE=true found")
        else:
            print("❌ Add USE_GOOGLE_DRIVE=true to .env")

        if 'GOOGLE_DRIVE_FOLDER_ID=' in env_content:
            print("✅ GOOGLE_DRIVE_FOLDER_ID found")
        else:
            print("❌ Add GOOGLE_DRIVE_FOLDER_ID=your_folder_id to .env")
    else:
        print("❌ .env file not found")
        print("💡 Create .env file with:")
        print("   USE_GOOGLE_DRIVE=true")
        print("   GOOGLE_DRIVE_FOLDER_ID=your_folder_id_here")
        print("   GOOGLE_SERVICE_ACCOUNT_JSON=your_base64_encoded_key")

    print(f"\n🧪 Test when ready:")
    print(f"   python src/drive_manager.py --status")
    print(f"   python test_enhanced_integration.py")


if __name__ == "__main__":
    main()
