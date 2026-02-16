import os
try:
    import bcrypt
    from pymongo import MongoClient
    AUTH_LIBS_AVAILABLE = True
except ImportError:
    AUTH_LIBS_AVAILABLE = False
    # Define dummy MongoClient if pymongo is missing
    class MongoClient:
        def __init__(self, *args, **kwargs): pass
        def close(self): pass
        def admin(self): return self
        def command(self, *args): pass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "carecost_predictor"
COLLECTION_NAME = "users"

def get_db_client():
    """Initialize and return MongoDB client."""
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        # Force a connection check
        client.admin.command('ping')
        return client
    except Exception as e:
        print(f"MongoDB Connection Error: {e}")
        return None

def hash_password(password):
    """Hash a password for security."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt)

def check_password(password, hashed):
    """Check if the provided password matches the hashed one."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def sign_up_user(username, password, email):
    """Register a new user in MongoDB."""
    client = get_db_client()
    if not client:
        return False, "Could not connect to database"
    
    db = client[DB_NAME]
    users_col = db[COLLECTION_NAME]
    
    # Check if user already exists
    if users_col.find_one({"username": username}):
        client.close()
        return False, "Username already exists"
    
    # Hash and save user
    hashed_pw = hash_password(password)
    users_col.insert_one({
        "username": username,
        "email": email,
        "password": hashed_pw
    })
    
    client.close()
    return True, "User registered successfully"

def login_user(username, password):
    """Validate user credentials against MongoDB."""
    client = get_db_client()
    if not client:
        return False, "Could not connect to database", None
    
    db = client[DB_NAME]
    users_col = db[COLLECTION_NAME]
    
    user = users_col.find_one({"username": username})
    client.close()
    
    if user and check_password(password, user['password']):
        return True, "Login successful", user.get('email')
    
    return False, "Invalid username or password", None

def get_all_users():
    """Retrieve all registered users (Admin only)."""
    client = get_db_client()
    if not client:
        return []
    
    db = client[DB_NAME]
    users_col = db[COLLECTION_NAME]
    
    # Return username and email only, exclude password
    users = list(users_col.find({}, {"username": 1, "email": 1, "_id": 0}))
    client.close()
    return users
