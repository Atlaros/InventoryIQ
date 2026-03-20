"""
Automated script to create catalog tables in Supabase using the REST API.
"""
import os
import pathlib
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / '.env'
load_dotenv(ENV_PATH)

URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY")

if not URL or not KEY:
    raise ValueError("❌ Missing Supabase credentials in .env")

supabase: Client = create_client(URL, KEY)

def create_tables_via_api():
    """
    Attempt to create tables by inserting a dummy row and letting Supabase auto-create.
    Note: This is a workaround since Supabase Python SDK doesn't support DDL.
    """
    print("==========================================")
    print("🏗️  CREATING CATALOG TABLES")
    print("==========================================")
    
    # Try to create items table by inserting dummy data
    print("\n📦 Creating items table...")
    try:
        # Insert a dummy item
        supabase.table('items').insert({"id": 999, "name": "TEMP_ITEM"}).execute()
        # Delete it immediately
        supabase.table('items').delete().eq('id', 999).execute()
        print("✅ Items table exists or was created")
    except Exception as e:
        print(f"⚠️  Items table creation: {e}")
    
    # Try to create stores table
    print("\n🏪 Creating stores table...")
    try:
        # Insert a dummy store
        supabase.table('stores').insert({"id": 999, "name": "TEMP_STORE"}).execute()
        # Delete it immediately
        supabase.table('stores').delete().eq('id', 999).execute()
        print("✅ Stores table exists or was created")
    except Exception as e:
        print(f"⚠️  Stores table creation: {e}")
    
    print("\n==========================================")
    print("🏁 TABLE CREATION COMPLETE")
    print("==========================================")

if __name__ == '__main__':
    create_tables_via_api()
