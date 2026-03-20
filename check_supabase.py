from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()
supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

# Check tables
result = supabase.table('sales').select('*').limit(5).execute()
print(f'Total rows in sales table: {len(result.data)}')
print('Sample data:')
for row in result.data[:2]:
    print(row)
