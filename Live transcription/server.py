import os, json, uuid, tempfile, time, re
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect
from vosk import Model, KaldiRecognizer
import numpy as np
import soundfile as sf
from word2number import w2n  
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

vosk_en = Model(os.path.join(MODELS_DIR, "vosk-model-en-in-0.5"))
vosk_hi = Model(os.path.join(MODELS_DIR, "vosk-model-small-hi-0.22"))

# ===================== TEXT NORMALIZATION FUNCTION =====================

def normalize_dates_times(text: str) -> str:
    original = text
    text = text.lower()
    text = re.sub(r"\ba\s*m\b", "am", text)
    text = re.sub(r"\bp\s*m\b", "pm", text)

    try:
        # =========================================================
        #  STEP 0 : HANDLE TIMES FIRST 
        # =========================================================
        def time_replacer_words(match):
            phrase = match.group(0)          
            words = phrase.split()

            try:
                hour = w2n.word_to_num(words[0])            
                minute = w2n.word_to_num(" ".join(words[1:-1]))  
                period = words[-1].upper()                  

                return f"{hour}:{minute:02d} {period}"
            except:
                return phrase  

        text = re.sub(
            r"\b"
            r"(?:one|two|three|four|five|six|seven|eight|nine|ten|"
            r"eleven|twelve)\s+"
            r"(?:oh\s+)?"
            r"(?:one|two|three|four|five|six|seven|eight|nine|ten|"
            r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
            r"eighteen|nineteen|twenty|twenty\s+one|twenty\s+two|twenty\s+three|"
            r"twenty\s+four|twenty\s+five|twenty\s+six|twenty\s+seven|twenty\s+eight|"
            r"twenty\s+nine|thirty|thirty\s+one|thirty\s+two|thirty\s+three|"
            r"thirty\s+four|thirty\s+five|thirty\s+six|thirty\s+seven|thirty\s+eight|"
            r"thirty\s+nine|forty|forty\s+one|forty\s+two|forty\s+three|"
            r"forty\s+four|forty\s+five|forty\s+six|forty\s+seven|forty\s+eight|"
            r"forty\s+nine|fifty|fifty\s+one|fifty\s+two|fifty\s+three|"
            r"fifty\s+four|fifty\s+five|fifty\s+six|fifty\s+seven|fifty\s+eight|"
            r"fifty\s+nine)"
            r"\s+(am|pm)\b",
            time_replacer_words,
            text
        )


        # =========================================================
        # STEP 1: TEEN YEARS 
        # =========================================================
        def fix_teen_year(match):
            words = match.group(0).split()

            if len(words) >= 2 and words[0] in [
                "nineteen", "eighteen", "seventeen",
                "sixteen", "fifteen", "fourteen",
                "thirteen", "twelve", "eleven"
            ]:
                first = w2n.word_to_num(words[0])     
                rest = w2n.word_to_num(" ".join(words[1:]))  
                return str(first * 100 + rest)        

            return str(w2n.word_to_num(match.group(0)))

        text = re.sub(
            r"\b(nineteen|eighteen|seventeen|sixteen|fifteen|"
            r"fourteen|thirteen|twelve|eleven)(?:\s+\w+)+\b",
            fix_teen_year,
            text
        )
        
        def fix_repeated_decade(match):
            word = match.group(1) 
            try:
                decade = w2n.word_to_num(word)   
                return str(decade * 100 + decade) 
            except:
                return match.group(0)

        
        text = re.sub(
            r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+\1\b(?!\s*(am|pm))",
            fix_repeated_decade,
            text
        )


        # =========================================================
        # STEP 2: CONVERT ALL REMAINING NUMBER PHRASES 
        # =========================================================
        number_words_pattern = (
            r"\b("
            r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
            r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
            r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
            r"eighty|ninety|hundred|thousand|million|and)"
            r"(?:\s+|$)"
            r")+"
            r"\b"
        )

        def convert_span(match):
            phrase = match.group(0).strip()
            try:
                return str(w2n.word_to_num(phrase))
            except:
                return phrase

        text = re.sub(number_words_pattern, convert_span, text)

        # =========================================================
        # STEP 3: CLEAN UP TIMES WRITTEN AS DIGITS 
        # =========================================================
        def time_replacer(match):
            h, m, p = match.groups()
            return f"{h}:{int(m):02d} {p.upper()}"

        text = re.sub(
            r"\b(\d+)\s+(\d+)\s+(am|pm)\b",
            time_replacer,
            text
        )

        # =========================================================
        # STEP 4: CAPITALIZE MONTHS 
        # =========================================================
        months = [
            "january","february","march","april","may","june",
            "july","august","september","october","november","december"
        ]
        for m in months:
            text = re.sub(rf"\b{m}\b", m.capitalize(), text)

        return text

    except Exception as e:
        print(f"[normalize_dates_times ERROR]: {e}")
        return original



# ===========================================================================
def remove_repeated_words(text: str) -> str:
    words=text.split()
    if not words:
        return text
    cleaned=[words[0]]
    for w in words[1:]:
        if w.lower() != cleaned[-1].lower():
            cleaned.append(w)
    return " ".join(cleaned)
# ===========================================================================
def insert_punctuation(text: str) -> str:
    text=text.strip()
    if not text:
        return text
    question_words=["who","what","when","where","why","how","is","are","do","does","did","can","could","would","will","shall","may","might"]
    if text.lower().startswith(tuple(question_words)):
        if not text.endswith("?"):
            return text + "?"
        return text
    if text[-1].isalnum():
        return text + "."
    return text
# ===========================================================================
def insert_commas(text: str) -> str:
    conjunctions=["and","but","so","because","however","therefore","moreover","meanwhile","otherwise"]
    for conj in conjunctions:
        text=text.replace(conj," , " + conj)
    return text
# ===========================================================================
def capitalization(text: str) -> str:
    if not text:
        return text
    text=text[0].upper() + text[1:]
    text=re.sub(r"\bi\b","I",text)
    return text
# ===========================================================================
def full_post_process(text: str) -> str:
    text=normalize_dates_times(text)
    text=remove_repeated_words(text)
    text=insert_commas(text)
    text=insert_punctuation(text) 
    text=capitalization(text)
    return text
# ===========================================================================
HI_NUM_WORDS = {
    "शून्य":0, "एक":1, "दो":2, "तीन":3, "चार":4, "पाँच":5, "छह":6,
    "सात":7, "आठ":8, "नौ":9, "दस":10, "ग्यारह":11, "बारह":12,
    "तेरह":13, "चौदह":14, "पंद्रह":15, "सोलह":16, "सत्रह":17,
    "अठारह":18, "उन्नीस":19, "बीस":20, "तीस":30, "चालीस":40,
    "पचास":50, "साठ":60, "सत्तर":70, "अस्सी":80, "नब्बे":90,
    "सौ":100, "हज़ार":1000, "हजार":1000
}
def normalize_hindi_numbers(text: str) -> str:
    words=text.split()
    result=[]
    current=0
    total=0
    
    for w in words:
        if w in HI_NUM_WORDS:
            val=HI_NUM_WORDS[w]
            if val==100:
                current*=100
            elif val==1000:
                total+=current*1000
                current=0
            else:
                current+=val
        else:
            if current or total:
                result.append(str(total + current))
                current=total=0
            result.append(w)
    if current or total:
        result.append(str(total+current))
    return " ".join(result)
# ===========================================================================
def normalize_hindi_time(text: str) -> str:
    text = text.lower()

    text = normalize_hindi_numbers(text)

    text = re.sub(
        r"(?:साढ़े|साडे)\s*(\d+)",
        lambda m: f"{m.group(1)}:30",
        text
    )

    def paune_repl(m):
        hour = int(m.group(1)) - 1
        return f"{hour}:45"

    text = re.sub(
        r"(?:पौने|पोने)\s*(\d+)",
        paune_repl,
        text
    )

    
    text = re.sub(
        r"\b(\d+)(?!:)\s*(?:बजे|बजें|बजे़|बाजे)\b",
        r"\1:00",
        text
    )


    return text




# ===========================================================================
def insert_hindi_punctuation(text: str) -> str:
    text = text.strip()
    if not text:
        return text

    question_words = ("क्या", "क्यों", "कैसे", "कब", "कहाँ", "कौन")

    if any(q in text for q in question_words):
        if not text.endswith("?"):
            return text + "?"
        return text

    if not text.endswith("।"):
        return text + "।"

    return text

# ===========================================================================
def post_process_hi(text: str) -> str:
    text = normalize_hindi_numbers(text)
    text = normalize_hindi_time(text)
    text = remove_repeated_words(text)
    text = insert_hindi_punctuation(text)
    return text
# ===========================================================================

@app.get("/")
async def root():
    with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    print(f"User connected: {session_id}")

    try:
        # -------------------------------
        # STEP 1: Wait for language config
        # -------------------------------
        lang = "en"

        while True:
            msg = await websocket.receive()

            if msg["type"] == "websocket.disconnect":
                print(f"Session {session_id} disconnected before language selection")
                return

            if "text" in msg:
                try:
                    config = json.loads(msg["text"])
                    lang = config.get("language", "en")
                    print(f"Session {session_id} selected language: {lang}")
                except:
                    print("Invalid language config, defaulting to English")
                break

        # -------------------------------
        # STEP 2: Load correct model
        # -------------------------------
        sr = 16000

        if lang == "hi":
            recognizer = KaldiRecognizer(vosk_hi, sr)
            print(f"Session {session_id} using Hindi model")
        else:
            recognizer = KaldiRecognizer(vosk_en, sr)
            print(f"Session {session_id} using English model")

        recognizer.SetWords(True)

        # -------------------------------
        # STEP 3: Audio transcription loop
        # -------------------------------
        while True:
            msg = await websocket.receive()

            if msg["type"] == "websocket.disconnect":
                print(f"User disconnected: {session_id}")
                break

            if "bytes" not in msg:
                continue

            data = msg["bytes"]

            if recognizer.AcceptWaveform(data):
                raw_text = json.loads(recognizer.Result()).get("text", "")

                if lang == "hi":
                    clean_text = post_process_hi(raw_text)
                else:
                    clean_text = full_post_process(raw_text)

                await websocket.send_json({
                    "text": clean_text,
                    "final": True
                })
            else:
                raw_partial = json.loads(recognizer.PartialResult()).get("partial", "")

                await websocket.send_json({
                    "text": raw_partial,
                    "final": False
                })

    except WebSocketDisconnect:
        print(f"WebSocketDisconnect: {session_id}")



