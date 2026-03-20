"""
Script to create and seed catalog tables (items and stores) in Supabase.
This adds human-readable names to the numeric IDs.
"""
import os
import pathlib
from supabase import create_client, Client
from dotenv import load_dotenv
from catalog import PRODUCTS, STORES

# Load environment
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / '.env'
load_dotenv(ENV_PATH)

URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY")

if not URL or not KEY:
    raise ValueError("❌ Missing Supabase credentials in .env")

supabase: Client = create_client(URL, KEY)

def create_tables():
    """
    Note: Supabase doesn't support DDL via Python SDK.
    You need to run these SQL commands in the Supabase SQL Editor:
    
    CREATE TABLE IF NOT EXISTS items (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL
    );
    
    CREATE TABLE IF NOT EXISTS stores (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL
    );
    """
    print("⚠️  Please create tables manually in Supabase SQL Editor:")
    print("""
    CREATE TABLE IF NOT EXISTS items (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL
    );
    
    CREATE TABLE IF NOT EXISTS stores (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL
    );
    """)

def seed_items():
    print("\n📦 Seeding items table...")
    items_data = [{"id": id, "name": name} for id, name in PRODUCTS.items()]
    
    try:
        result = supabase.table('items').upsert(items_data).execute()
        print(f"✅ Inserted {len(items_data)} products")
    except Exception as e:
        print(f"❌ Error seeding items: {e}")

def seed_stores():
    print("\n🏪 Seeding stores table...")
    stores_data = [{"id": id, "name": name} for id, name in STORES.items()]
    
    try:
        result = supabase.table('stores').upsert(stores_data).execute()
        print(f"✅ Inserted {len(stores_data)} stores")
    except Exception as e:
        print(f"❌ Error seeding stores: {e}")

if __name__ == '__main__':
    print("==========================================")
    print("🏗️  CATALOG SEEDER")
    print("==========================================")
    
    seed_items()
    seed_stores()
    
    print("\n==========================================")
    print("🏁 CATALOG SEEDING COMPLETE")
    print("==========================================")
