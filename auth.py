from pydantic import BaseModel
from typing import Dict, List, Optional
import jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
from database import db_manager

# User model
class User(BaseModel):
    email: str
    first_name: str
    last_name: str
    password: str

# User manager class
class UserManager:
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def hash_password(self, password: str) -> str:
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return self.pwd_context.verify(plain_password, hashed_password)

    def register_user(self, user: User):
        """Register a new user in the database"""
        conn = db_manager.get_connection()
        cursor = conn.cursor()

        # Check if user already exists
        cursor.execute("SELECT id FROM users WHERE email = ?", (user.email,))
        if cursor.fetchone():
            raise ValueError("User already exists")

        # Hash password and insert user
        hashed_password = self.hash_password(user.password)
        cursor.execute("""
            INSERT INTO users (email, first_name, last_name, password_hash)
            VALUES (?, ?, ?, ?)
        """, (user.email, user.first_name, user.last_name, hashed_password))

        conn.commit()

    def authenticate_user(self, email: str, password: str) -> Optional[Dict]:
        """Authenticate user and return user data if valid"""
        conn = db_manager.get_connection()
        cursor = conn.cursor()

        # Get user from database
        cursor.execute("""
            SELECT id, email, first_name, last_name, password_hash, created_at
            FROM users WHERE email = ?
        """, (email,))

        user_row = cursor.fetchone()
        if not user_row:
            return None

        # Verify password
        if not self.verify_password(password, user_row['password_hash']):
            return None

        # Return user data
        return {
            "id": user_row['id'],
            "email": user_row['email'],
            "first_name": user_row['first_name'],
            "last_name": user_row['last_name'],
            "created_at": user_row['created_at']
        }

    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get user data by email"""
        conn = db_manager.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, email, first_name, last_name, created_at
            FROM users WHERE email = ?
        """, (email,))

        user_row = cursor.fetchone()
        if not user_row:
            return None

        return {
            "id": user_row['id'],
            "email": user_row['email'],
            "first_name": user_row['first_name'],
            "last_name": user_row['last_name'],
            "created_at": user_row['created_at']
        }

    def add_analysis_history(self, user_email: str, traits: Dict, confidence: float):
        """Add analysis history for a user"""
        conn = db_manager.get_connection()
        cursor = conn.cursor()

        # Get user ID
        cursor.execute("SELECT id FROM users WHERE email = ?", (user_email,))
        user_row = cursor.fetchone()
        if not user_row:
            return

        # Insert analysis history
        cursor.execute("""
            INSERT INTO analysis_history (user_id, traits, confidence)
            VALUES (?, ?, ?)
        """, (user_row['id'], str(traits), confidence))

        conn.commit()

# JWT functions
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Create user manager instance
user_manager = UserManager()
