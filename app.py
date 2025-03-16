from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import random
import os
import time
import uuid
import logging
import re
from difflib import SequenceMatcher
from contextlib import asynccontextmanager
from datetime import datetime
from dotenv import load_dotenv
import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("lyric_match_api")

# Load environment variables
load_dotenv()

# Set up OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logger.error("OpenAI API key not found! Please set the OPENAI_API_KEY environment variable.")
    raise ValueError("OpenAI API key not found")

# Difficulty levels
difficulty_levels = ["easy", "medium", "hard"]

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("✅ API server starting up")
    yield
    # Shutdown
    logger.info("Shutting down API server")

app = FastAPI(
    title="Lyric Match API",
    description="Advanced API for the Lyric Match song guessing game",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware for logging and performance tracking
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log the request details
        logger.info(
            f"Path: {request.url.path} | "
            f"Method: {request.method} | "
            f"Time: {process_time:.4f}s | "
            f"Status: {response.status_code}"
        )
        return response
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        process_time = time.time() - start_time
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(e), "process_time": process_time},
        )

# Models
class Genre(BaseModel):
    id: str
    name: str

class SongDetails(BaseModel):
    title: str
    artist: str
    year: Optional[int] = None
    genres: Optional[List[str]] = None
    
    @property
    def full_name(self):
        return f"{self.title} - {self.artist}"

class GameSettings(BaseModel):
    difficulty: str = Field(default="medium", description="Difficulty level: easy, medium, or hard")
    genre_filter: Optional[List[str]] = Field(default=None, description="List of genre IDs to filter by")
    line_count: Optional[int] = Field(default=None, description="Number of lyric lines to generate")

class SongGuess(BaseModel):
    user_guess: str
    session_id: str


class HintResponse(BaseModel):
    hint: str
    hints_remaining: int


class HintRequest(BaseModel):
    session_id: str
    hint_type: str = Field(default="context", description="Hint type: artist, year, genre, first_letter, word_count")

class LyricResponse(BaseModel):
    lyric_snippet: str
    session_id: str
    difficulty: str
    hints_available: int

class GuessResult(BaseModel):
    is_correct: bool
    similarity_score: float
    feedback: str
    correct_song: Optional[str] = None  # Only provided when guess is correct or game is over

class StatsResponse(BaseModel):
    total_games: int
    correct_guesses: int
    accuracy_rate: float
    most_guessed_songs: List[Dict[str, Union[str, int]]]

# Song database
songs = [
    SongDetails(
        title="Blinding Lights",
        artist="The Weeknd",
        year=2019,
        genres=["synth-pop", "new wave"]
    ),
    SongDetails(
        title="Shape of You",
        artist="Ed Sheeran",
        year=2017,
        genres=["pop", "dancehall"]
    ),
    SongDetails(
        title="Someone Like You",
        artist="Adele",
        year=2011,
        genres=["pop", "soul"]
    ),
    SongDetails(
        title="Bohemian Rhapsody",
        artist="Queen",
        year=1975,
        genres=["rock", "progressive rock"]
    ),
    SongDetails(
        title="Billie Jean",
        artist="Michael Jackson",
        year=1983,
        genres=["pop", "dance"]
    ),
    SongDetails(
        title="Sweet Child O' Mine",
        artist="Guns N' Roses",
        year=1987,
        genres=["hard rock"]
    ),
    SongDetails(
        title="Rolling in the Deep",
        artist="Adele",
        year=2010,
        genres=["soul", "blues"]
    ),
    SongDetails(
        title="Uptown Funk",
        artist="Mark Ronson ft. Bruno Mars",
        year=2014,
        genres=["funk", "pop"]
    ),
    SongDetails(
        title="Hotel California",
        artist="Eagles",
        year=1977,
        genres=["rock"]
    ),
    SongDetails(
        title="Bitter Sweet Symphony",
        artist="The Verve",
        year=1997,
        genres=["alternative rock"]
    ),
    SongDetails(
        title="Smells Like Teen Spirit",
        artist="Nirvana",
        year=1991,
        genres=["grunge", "alternative rock"]
    ),
    SongDetails(
        title="Thinking Out Loud",
        artist="Ed Sheeran",
        year=2014,
        genres=["soul", "pop"]
    ),
    SongDetails(
        title="Thriller",
        artist="Michael Jackson",
        year=1982,
        genres=["pop", "funk", "disco"]
    ),
    SongDetails(
        title="Bad Guy",
        artist="Billie Eilish",
        year=2019,
        genres=["electropop", "trap"]
    ),
    SongDetails(
        title="Starboy",
        artist="The Weeknd",
        year=2016,
        genres=["R&B", "electropop"]
    ),
    SongDetails(
        title="Watermelon Sugar",
        artist="Harry Styles",
        year=2019,
        genres=["pop rock", "indie pop"]
    ),
    SongDetails(
        title="Don't Start Now",
        artist="Dua Lipa",
        year=2019,
        genres=["nu-disco", "dance-pop"]
    ),
    SongDetails(
        title="Levitating",
        artist="Dua Lipa",
        year=2020,
        genres=["disco", "pop"]
    ),
    SongDetails(
        title="Save Your Tears",
        artist="The Weeknd",
        year=2020,
        genres=["synth-pop", "new wave"]
    ),
    SongDetails(
        title="Dance Monkey",
        artist="Tones and I",
        year=2019,
        genres=["electropop", "dance-pop"]
    ),
    SongDetails(
        title="Shallow",
        artist="Lady Gaga & Bradley Cooper",
        year=2018,
        genres=["country rock", "folk rock"]
    ),
    SongDetails(
        title="All of Me",
        artist="John Legend",
        year=2013,
        genres=["R&B", "soul"]
    ),
    SongDetails(
        title="Despacito",
        artist="Luis Fonsi ft. Daddy Yankee",
        year=2017,
        genres=["reggaeton", "latin pop"]
    ),
    SongDetails(
        title="Happy",
        artist="Pharrell Williams",
        year=2013,
        genres=["soul", "funk"]
    ),
    SongDetails(
        title="As It Was",
        artist="Harry Styles",
        year=2022,
        genres=["synth-pop", "indie pop"]
    ),
    SongDetails(
        title="Stay",
        artist="The Kid LAROI & Justin Bieber",
        year=2021,
        genres=["pop", "hyperpop"]
    ),
    SongDetails(
        title="Circles",
        artist="Post Malone",
        year=2019,
        genres=["pop", "alternative"]
    ),
    SongDetails(
        title="Believer",
        artist="Imagine Dragons",
        year=2017,
        genres=["pop rock", "electropop"]
    ),
    SongDetails(
        title="Unstoppable",
        artist="Sia",
        year=2016,
        genres=["electropop", "pop"]
    ),
    SongDetails(
        title="Take Me to Church",
        artist="Hozier",
        year=2013,
        genres=["indie rock", "blues rock"]
    )
]

# Track active game sessions
active_sessions = {}

# Game statistics
game_stats = {
    "total_games": 0,
    "correct_guesses": 0,
    "song_attempts": {},
}

# Helper Functions
def get_difficulty_parameters(difficulty: str):
    """Return parameters based on difficulty level"""
    if difficulty not in difficulty_levels:
        difficulty = "medium"  # Default to medium if invalid
        
    if difficulty.lower() == "easy":
        return {
            "line_count": 4,
            "provide_context": True,
            "hints_available": 3,
            "temperature": 0.5,
            "max_attempts": 5
        }
    elif difficulty.lower() == "hard":
        return {
            "line_count": 2,
            "provide_context": False,
            "hints_available": 1,
            "temperature": 0.9,
            "max_attempts": 3
        }
    else:  # medium (default)
        return {
            "line_count": 3,
            "provide_context": False,
            "hints_available": 2,
            "temperature": 0.7,
            "max_attempts": 4
        }

def select_song(genre_filter=None):
    """Select a random song, optionally filtered by genre"""
    filtered_songs = songs
    if genre_filter:
        filtered_songs = [song for song in songs if any(genre in song.genres for genre in genre_filter)]
        if not filtered_songs:
            filtered_songs = songs  # Fallback if no songs match filter
    
    return random.choice(filtered_songs)

def calculate_similarity(guess: str, correct: str) -> float:
    """Calculate string similarity between guess and correct answer"""
    # Clean up the strings
    guess = re.sub(r'[^\w\s]', '', guess.lower()).strip()
    correct = re.sub(r'[^\w\s]', '', correct.lower()).strip()
    
    # Use SequenceMatcher for better similarity detection
    ratio = SequenceMatcher(None, guess, correct).ratio()
    
    # Also check if words from guess appear in correct
    guess_words = set(guess.split())
    correct_words = set(correct.split())
    
    # Calculate Jaccard similarity
    union = len(guess_words.union(correct_words))
    intersection = len(guess_words.intersection(correct_words))
    jaccard = intersection / union if union > 0 else 0.0
    
    # Combine both metrics with a weight
    return (ratio * 0.7) + (jaccard * 0.3)

def generate_feedback(similarity_score: float, is_correct: bool, attempts_left: int) -> str:
    """Generate feedback based on similarity score and remaining attempts"""
    if is_correct:
        return "Correct! Well done!"
    
    if attempts_left == 0:
        return "Game over! You've used all your attempts."
    
    if similarity_score > 0.7:
        return f"Very close! You almost had it. {attempts_left} attempts remaining."
    elif similarity_score > 0.4:
        return f"You're on the right track, but not quite there. {attempts_left} attempts remaining."
    elif similarity_score > 0.2:
        return f"Your guess is somewhat related, but pretty far off. {attempts_left} attempts remaining."
    else:
        return f"Not even close. Try again! {attempts_left} attempts remaining."

def generate_hint(song: SongDetails, hint_type: str) -> str:
    """Generate a hint based on hint type"""
    if hint_type == "artist":
        return f"This song is performed by {song.artist}."
    elif hint_type == "year":
        return f"This song was released in {song.year}."
    elif hint_type == "genre":
        if song.genres and len(song.genres) > 0:
            genres_str = ", ".join(g.title() for g in song.genres)
            return f"This song's genres include {genres_str}."
        return "Genre information is not available for this song."
    elif hint_type == "first_letter":
        return f"The song title starts with the letter '{song.title[0]}'."
    elif hint_type == "word_count":
        word_count = len(song.title.split())
        return f"The song title contains {word_count} word{'s' if word_count > 1 else ''}."
    else:  # context hint - more lyrics or context
        return "Try one of these hint types: 'artist', 'year', 'genre', 'first_letter', or 'word_count'."

def cleanup_old_sessions():
    """Remove sessions older than 1 hour"""
    current_time = datetime.now()
    sessions_to_remove = []
    
    for session_id, session in active_sessions.items():
        session_time = datetime.fromisoformat(session["timestamp"])
        if (current_time - session_time).total_seconds() > 3600:  # 1 hour
            sessions_to_remove.append(session_id)
    
    for session_id in sessions_to_remove:
        del active_sessions[session_id]
    
    if sessions_to_remove:
        logger.info(f"Cleaned up {len(sessions_to_remove)} expired sessions")

# API Routes
@app.get("/")
def read_root():
    return {
        "message": "Welcome to Lyric Match API v2.0",
        "documentation": "/docs",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/genres", response_model=List[Genre])
async def get_genres():
    """Get available genres for filtering"""
    all_genres = set()
    for song in songs:
        if song.genres:
            all_genres.update(song.genres)
    
    return [
        {"id": genre, "name": genre.title()}
        for genre in sorted(all_genres)
    ]

@app.post("/api/start", response_model=LyricResponse)
async def start_game(settings: GameSettings):
    """Start a new song guessing game"""
    # Clean up old sessions periodically
    cleanup_old_sessions()
    
    # Generate a new session ID
    session_id = str(uuid.uuid4())
    
    # Get difficulty parameters
    difficulty = settings.difficulty.lower()
    params = get_difficulty_parameters(difficulty)
    
    # Override line count if specified
    if settings.line_count:
        params["line_count"] = settings.line_count
    
    # Select a random song based on genre filter
    selected_song = select_song(settings.genre_filter)
    
    # Generate lyrics using OpenAI
    try:
        system_prompt = f"""You are a lyrics expert. Generate {params['line_count']} consecutive lines from the song '{selected_song.title}' by {selected_song.artist}.
        -> Make sure you doesnt give any song name or artist name included in the response.
        -> DO NOT give any greetings or anything else. Just the lyrics. """
        if params["provide_context"]:
            system_prompt += " Include enough context that someone familiar with the song would recognize it, but not the most famous or obvious lyrics."
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please provide {params['line_count']} consecutive lines from the song '{selected_song.title}' by {selected_song.artist}."}
            ],
            temperature=params["temperature"],
            max_tokens=150
        )
        
        lyric_snippet = response.choices[0].message.content.strip()
        
        # Create session data
        active_sessions[session_id] = {
            "song": selected_song,
            "lyric_snippet": lyric_snippet,
            "difficulty": difficulty,
            "hints_available": params["hints_available"],
            "max_attempts": params["max_attempts"],
            "attempts_made": 0,
            "timestamp": datetime.now().isoformat(),
            "hints_used": []
        }
        
        # Update game stats
        game_stats["total_games"] += 1
        
        # Track song usage
        song_key = selected_song.full_name
        if song_key in game_stats["song_attempts"]:
            game_stats["song_attempts"][song_key]["uses"] += 1
        else:
            game_stats["song_attempts"][song_key] = {
                "uses": 1,
                "correct_guesses": 0
            }
        
        logger.info(f"New game started with session ID: {session_id}, difficulty: {difficulty}")
        
        return LyricResponse(
            lyric_snippet=lyric_snippet,
            session_id=session_id,
            difficulty=difficulty,
            hints_available=params["hints_available"]
        )
        
    except Exception as e:
        logger.error(f"Error generating lyrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate lyrics: {str(e)}"
        )

@app.post("/api/guess", response_model=GuessResult)
async def make_guess(guess_data: SongGuess):
    """Submit a guess for the current song"""
    session_id = guess_data.session_id
    user_guess = guess_data.user_guess
    
    # Check if session exists
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=404,
            detail="Game session not found. Please start a new game."
        )
    
    session = active_sessions[session_id]
    correct_song = session["song"]
    
    # Check if already out of attempts
    if session["attempts_made"] >= session["max_attempts"]:
        return GuessResult(
            is_correct=False,
            similarity_score=0.0,
            feedback="Game over! You've used all your attempts.",
            correct_song=correct_song.full_name
        )
    
    # Increment attempt counter
    session["attempts_made"] += 1
    attempts_left = session["max_attempts"] - session["attempts_made"]
    
    # Calculate similarity between guess and correct answer
    similarity_score = calculate_similarity(user_guess, correct_song.full_name)
    
    # Determine if guess is correct (allowing for some flexibility)
    is_correct = similarity_score > 0.8
    
    # Generate feedback
    feedback = generate_feedback(similarity_score, is_correct, attempts_left)
    
    # Update stats if correct or out of attempts
    if is_correct:
        game_stats["correct_guesses"] += 1
        song_key = correct_song.full_name
        game_stats["song_attempts"][song_key]["correct_guesses"] += 1
        
        # Clean up session if correct
        if session_id in active_sessions:
            del active_sessions[session_id]
            
        return GuessResult(
            is_correct=True,
            similarity_score=similarity_score,
            feedback=feedback,
            correct_song=correct_song.full_name
        )
    
    # Out of attempts
    if attempts_left <= 0:
        # Clean up session if out of attempts
        if session_id in active_sessions:
            del active_sessions[session_id]
            
        return GuessResult(
            is_correct=False,
            similarity_score=similarity_score,
            feedback=feedback,
            correct_song=correct_song.full_name
        )
    
    # Still have attempts left
    return GuessResult(
        is_correct=False,
        similarity_score=similarity_score,
        feedback=feedback
    )

@app.post("/api/hint", response_model=HintResponse)
async def get_hint(hint_request: HintRequest):
    """Get a hint for the current song"""
    session_id = hint_request.session_id
    hint_type = hint_request.hint_type
    
    # Check if session exists
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=404,
            detail="Game session not found. Please start a new game."
        )
    
    session = active_sessions[session_id]
    
    # Check if hints are available
    if session["hints_available"] <= 0:
        raise HTTPException(
            status_code=400,
            detail="No hints available for this session."
        )
    
    # Check if this hint type was already used
    if hint_type in session["hints_used"]:
        raise HTTPException(
            status_code=400,
            detail=f"You've already used a '{hint_type}' hint in this session."
        )
    
    # Generate hint
    hint_text = generate_hint(session["song"], hint_type)
    
    # Update hint usage
    session["hints_available"] -= 1
    session["hints_used"].append(hint_type)
    
    return HintResponse(
        hint=hint_text,
        hints_remaining=session["hints_available"]
    )

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get game statistics"""
    total_games = game_stats["total_games"]
    correct_guesses = game_stats["correct_guesses"]
    accuracy_rate = (correct_guesses / total_games) * 100 if total_games > 0 else 0
    
    # Get most guessed songs
    most_guessed = []
    if game_stats["song_attempts"]:
        sorted_songs = sorted(
            game_stats["song_attempts"].items(),
            key=lambda x: x[1]["uses"],
            reverse=True
        )
        
        # Get top 10 songs
        top_songs = sorted_songs[:10]
        
        for song_name, data in top_songs:
            most_guessed.append({
                "song": song_name,
                "times_used": data["uses"],
                "times_guessed": data["correct_guesses"],
                "success_rate": (data["correct_guesses"] / data["uses"]) * 100 if data["uses"] > 0 else 0
            })
    
    return StatsResponse(
        total_games=total_games,
        correct_guesses=correct_guesses,
        accuracy_rate=round(accuracy_rate, 2),
        most_guessed_songs=most_guessed
    )

@app.delete("/api/session/{session_id}")
async def cancel_session(session_id: str):
    """Cancel an active game session"""
    if session_id in active_sessions:
        del active_sessions[session_id]
        return {"message": "Session cancelled successfully"}
    else:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )

@app.get("/api/health")
async def health_check():
    """API health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "active_sessions": len(active_sessions),
        "total_games_played": game_stats["total_games"],
        "uptime": "Unknown"  # You would calculate this from a start time variable
    }

@app.get("/api/session/{session_id}", response_model=Dict)
async def get_session_info(session_id: str):
    """Get information about an active session (for debugging)"""
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )
    
    session = active_sessions[session_id]
    
    # Create a safe copy without exposing the answer directly
    safe_session = {
        "difficulty": session["difficulty"],
        "hints_available": session["hints_available"],
        "attempts_made": session["attempts_made"],
        "max_attempts": session["max_attempts"],
        "timestamp": session["timestamp"],
        "lyric_snippet": session["lyric_snippet"],
        "hints_used": session["hints_used"],
        "song_genres": session["song"].genres
    }
    
    return safe_session

# Run the app
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)