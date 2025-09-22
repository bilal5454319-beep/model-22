# Import authentication
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auth import (
    user_manager,
    User,
    create_access_token
)
# Import helper functions
from your_model_loader import (
    load_binary_model,
    load_personality_model,
    predict_handwriting,
    predict_personality
)
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from PIL import Image
import io
from fastapi.security import OAuth2PasswordBearer
from fastapi.security import OAuth2PasswordRequestForm
from auth import SECRET_KEY, ALGORITHM
from jose import JWTError, jwt
from typing import Optional

# Load models (update paths to your actual .h5 files)
binary_model = load_binary_model("binary_handwriting_classifier.h5")
personality_model = load_personality_model("cnn_personality_model.h5")

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_token(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = user_manager.get_user_by_email(email)
    if user is None:
        raise credentials_exception
    return user

@app.post("/predict")
async def predict_personality_endpoint(file: UploadFile = File(...), user: dict = Depends(verify_token)):
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        # Step 1: Handwriting detection
        is_handwriting = predict_handwriting(image, binary_model)
        if not is_handwriting:
            raise HTTPException(status_code=422, detail="Uploaded image does not appear to be handwriting")
        # Step 2: Personality prediction
        traits = predict_personality(image, personality_model)
        # Add analysis history for user
        user_manager.add_analysis_history(user['email'], traits, 95)
        # Return in format expected by frontend
        return {"personality": traits, "confidence": 95}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# User registration model
class UserRegister(BaseModel):
    email: str
    first_name: str
    last_name: str
    password: str

# User login model
class UserLogin(BaseModel):
    email: str
    password: str

@app.post("/signup")
async def register_user(user: UserRegister):
    try:
        user_obj = User(**user.dict())
        user_manager.register_user(user_obj)
        access_token = create_access_token(data={"sub": user.email})
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "first_name": user.first_name,
                "last_name": user.last_name,
                "email": user.email
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/login")
async def login_user(user: UserLogin):
    try:
        user_data = user_manager.authenticate_user(user.email, user.password)
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        access_token = create_access_token(data={"sub": user.email})
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "first_name": user_data["first_name"],
                "last_name": user_data["last_name"],
                "email": user.email
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid credentials")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
