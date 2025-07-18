import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model
import time
from io import BytesIO
from audiorecorder import audiorecorder
from dotenv import dotenv_values
from openai import OpenAI
from instructor import Instructor
from pydantic import BaseModel
from hashlib import md5
from typing import Optional

st.set_page_config(page_title="Kalkulator ubezpieczeń", layout="centered")

# Wczytywanie zmiennych środowiskowych z pliku .env
env = dotenv_values(".env")

# Modele i stałe
AUDIO_TRANSCRIBE_MODEL = "whisper-1"
CONVERT_TO_JSON_MODEL = "gpt-4o-mini"
PREDICTION_CHARGE_MODEL = 'v5_insurance_charge_regression'
CURRENCY = "USD"

def is_valid_api_key(api_key):
    # Sprawdź długość klucza
    if len(api_key) != 164 or not api_key.startswith("sk-"):
        return False
    return True

def get_openai_client():
    key = st.session_state.get("openai_api_key")
    if not key or not isinstance(key, str):
        st.warning("Brakuje klucza OpenAI API.")
        st.stop()
    return OpenAI(api_key=key)


@st.cache_data
def get_model():
    return load_model(PREDICTION_CHARGE_MODEL)

# Funkcja do transkrypcji audio
def transcribe_audio(audio_bytes):
    openai_client = get_openai_client()
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "audio.mp3"
    transcript = openai_client.audio.transcriptions.create(
        file=audio_file,
        model=AUDIO_TRANSCRIBE_MODEL,
        response_format="verbose_json",
    )
    return transcript.text


# Konwersja tekstu do formatu json
def retrieve_structure(text: str, response_model: BaseModel):
    openai_client = get_openai_client()
    instructor_openai_client = Instructor.from_openai(openai_client)
    response = instructor_openai_client.chat.completions.create(
        model=CONVERT_TO_JSON_MODEL,
        temperature=0,
        response_model=response_model,
        messages=[
            {
                "role": "user",
                "content": text,
            },
        ],
    )
    
    return response.model_dump()

def validate_response_data(person):
    if (person.gender in [None, ''] or 
        person.age is None or 
        person.weight is None or 
        person.height is None or 
        person.children is None or
        person.smoker in [None, ''] or
        person.region in [None, ''] or
        (person.region.lower() != 'północny wschód' and person.region.lower() != 'południowy wschód' and person.region.lower() != 'północny zachód' and person.region.lower() != 'południowy zachód')): 
        return None
    else:
        # Upewnij się, że wartości są poprawne przed porównaniem
        if (person.age > 0 and person.weight > 0 and person.height > 0 and person.children >= 0):
            return person
        else:
            st.error("🛑 Niepoprawne wartości dla jednego z pól!")
            return None

def reduce_cost_and_display_message(person_df, weight, height, weight_changes, direction):
    valid_weight_change = None  # Zmienne do przechowywania znalezionej wagi
    min_cost = float('inf')  # Ustaw minimalny koszt na nieskończoność

    for weight_change in weight_changes:
        new_weight = weight - weight_change
        new_bmi = new_weight / (height ** 2)
        new_df = person_df.copy()
        new_df["bmi"] = round(new_bmi, 2)
        new_cost = predict_cost(model, new_df)

        # Sprawdź warunek dla nowego kosztu
        if (direction == "lower" and new_cost < predicted_charge) or \
           (direction == "increase" and new_cost < predicted_charge):

            # Sprawdź, czy nowy koszt jest mniejszy od minimalnego kosztu
            if new_cost < min_cost:
                min_cost = new_cost  # Zaktualizuj minimalny koszt
                valid_weight_change = weight_change  # Przechowaj zmianę wagę

    # Wypisz komunikat, jeśli znaleziono prawidłowy wynik
    if valid_weight_change is not None:
        with st.spinner("Jeszcze chwila..."):
            if direction == "lower":
                st.markdown(f"- Zredukuj masę o {valid_weight_change} kg, a Twój koszt ubezpieczenia wyniesie {min_cost} {CURRENCY}!")
            elif direction == "increase":
                st.markdown(f"- Zwiększ masę o {-valid_weight_change} kg, a Twój koszt ubezpieczenia wyniesie {min_cost} {CURRENCY}!")

    return valid_weight_change  # Zwróć wartość zmian wagi, jeśli znalazłeś właściwą


# Session state initialization
if "note_audio_bytes" not in st.session_state:
    st.session_state["note_audio_bytes"] = None

if "note_audio_text" not in st.session_state:
    st.session_state["note_audio_text"] = ""

# Funkcja do przewidywania kosztów
def predict_cost(model, data):
    prediction = predict_model(model, data=data)
    return round(prediction["prediction_label"][0], 2)

# CSS do stylizacji przycisków
st.markdown("""
    <style>
    div.stButton > button {
        background-color: #66CDAA;
        color: black;
        font-size: 20px;
        width: 50%;
        border-radius: 5%;
        margin-top: 20px;
    }
    div.stButton > button:hover {
        background-color: green;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


# OpenAI API key protection
if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in env:
        st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]

    else:
        st.title("Zaloguj do OpenAI")
        
        # Dodanie instrukcji dla użytkownika z kolorową ramką
        instruction_html = """
        <div style="background-color: #003366; padding: 10px; border-radius: 5px; border: 1px solid #0073e6; margin-bottom: 10px;">
            <h4>Instrukcje uzyskania klucza API</h4>
            <ol>
                <li>Załóż konto na stronie <a href="https://platform.openai.com/signup" target="_blank">OpenAI</a>.</li>
                <li>Wygeneruj swój klucz API w sekcji API Keys.</li>
                <li>Wklej go poniżej.</li>
            </ol>
        </div>
        """
        st.markdown(instruction_html, unsafe_allow_html=True)
        
        api_key_input = st.text_input("Klucz API", type="password")

        if api_key_input:
            if is_valid_api_key(api_key_input):
                st.session_state["openai_api_key"] = api_key_input
                st.rerun()
            else:
                st.error("Podany klucz API jest niepoprawny. Upewnij się, że klucz zaczyna się od 'sk-' i ma 164 znaki długości.")

if not st.session_state.get("openai_api_key"):
    st.stop()

# Pobieranie modelu
model = get_model()

# Interfejs użytkownika
cols = st.columns([2, 5])  # Użyj kolumn 1:4 dla obsługi obrazu i tekstu
with cols[0]:
    st.image('insurance_logo200x200.png', width=200)
with cols[1]:
    st.markdown(
"<h1 style='font-weight: bold; font-size: 1.8rem;'>Kalkulator ubezpieczeń</h1 style='font-weight: bold; font-size: 1.8rem;'>",
unsafe_allow_html=True
)

st.write('Aby obliczyć koszt ubezpieczenia, wypełnij formularz po lewej stronie lub nagraj notatkę głosową odnoszącą się do pól formularza!')

# Session state initialization
if "note_audio_bytes_md5" not in st.session_state:
    st.session_state["note_audio_bytes_md5"] = None

if "note_audio_bytes" not in st.session_state:
    st.session_state["note_audio_bytes"] = None

if "note_audio_text" not in st.session_state:
    st.session_state["note_audio_text"] = ""

if "note_text" not in st.session_state:
    st.session_state["note_text"] = ""

# Notatka audio
note_audio = audiorecorder(
start_prompt="Nagraj notatkę",
stop_prompt="Zatrzymaj nagrywanie",
)

# Ustalanie szerokości st.sidebar za pomocą CSS
st.markdown(
"""
<style>
/* Ustaw szerokość sidebaru na 300px */
.css-1lcbmhc {
    width: 00px;
}
.css-1lcbmhc.e1yy0g3g0 {
    width: 00px;
}
</style>
""",
unsafe_allow_html=True
)

with st.sidebar:
    if note_audio:
        audio = BytesIO()
        note_audio.export(audio, format="mp3")
        st.session_state["note_audio_bytes"] = audio.getvalue()
        current_md5 = md5(st.session_state["note_audio_bytes"]).hexdigest()
        if st.session_state["note_audio_bytes_md5"] != current_md5:
            st.session_state["note_audio_text"] = ""
            st.session_state["note_audio_bytes_md5"] = current_md5

        st.audio(st.session_state["note_audio_bytes"], format="audio/mp3")

        if st.button("Wyświetl notatkę"):
            with st.spinner("Czekaj, przetwarzam..."):
                st.session_state["note_audio_text"] = transcribe_audio(st.session_state["note_audio_bytes"])
                
        if st.session_state["note_audio_text"]:
            # Ustaw wartość text_area i zaktualizuj stan
            st.session_state["note_text"] = st.text_area("Twoja notatka", value=st.session_state["note_audio_text"])

            time.sleep(1)
            guide_info = st.info("Jeśli tekst jest zgodny z wypowiedzią to oblicz koszt! W przeciwnym razie edytuj notatkę lub nagraj nową!")

            st.session_state["transcription_ready"] = True

            

        if st.session_state.get("transcription_ready"):
            if st.button("Oblicz koszt"):
                with st.spinner("Czekaj, sprawdzam dane..."):
                    guide_info.empty()

                    class SpeechPerson(BaseModel):
                        gender: Optional[str]
                        age: Optional[int]
                        weight: Optional[int]
                        height: Optional[float]
                        children: Optional[int]
                        smoker: Optional[str]
                        region: Optional[str]

                    #st.write("Zawartość note_text:", st.session_state.get("note_text"))
                    result_data = retrieve_structure(
                        text=st.session_state.get("note_text"),  # Użyj zaktualizowanej wartości
                        response_model=SpeechPerson,
                    )
                           
                    # Mapowanie płci i regionu
                    region_mapping = {
                        "southeast": "południowy wschód",
                        "northeast": "północny wschód",
                        "southwest": "południowy zachód",
                        "northwest": "północny zachód"
                    }

                    gender_mapping = {
                        "female": "kobieta",
                        "male": "mężczyzna",
                    }

                    smoker_mapping = {
                        "yes": "tak",
                        "no": "nie",
                    }

                    # Dopasowanie wartości
                    result_data["gender"] = gender_mapping.get(result_data["gender"], result_data["gender"])
                    result_data["region"] = region_mapping.get(result_data["region"], result_data["region"])
                    result_data["smoker"] = smoker_mapping.get(result_data["smoker"], result_data["smoker"])

                    # Create an instance of response_model to leverage Pydantic validation
                    person = SpeechPerson(**result_data)
                
                    # Sprawdzamy kompletność pól formularza
                    validated_person = validate_response_data(person=person)
                    #st.write(validated_person)

                if validated_person is None:
                    st.error("🛑 Dane niekompletne! Edytuj notatkę lub nagraj jeszcze raz!")
                    st.stop()
                else:
                    success_message = st.success("Dane są kompletne!")
                    time.sleep(2)  
                    success_message.empty()
                    with st.spinner("Czekaj, obliczam..."):
                        time.sleep(1.5)
                        if validated_person.height > 3.0:
                            speech_bmi = validated_person.weight / ((validated_person.height / 100) ** 2)
                        else:
                            speech_bmi = validated_person.weight / (validated_person.height ** 2)
                            
                        if "person" not in st.session_state:
                            st.session_state["person"] = None
                            
                        if "speech_bmi" not in st.session_state:
                            st.session_state["speech_bmi"] = None
                            
                        st.session_state["speech_bmi"] = speech_bmi
                        st.session_state["person"] = validated_person  # Teraz przypisujesz do sesji

                        if "speech_person_df" not in st.session_state:
                            st.session_state["speech_person_df"] = None
                            
                        st.session_state["speech_person_df"] = pd.DataFrame([{
                            "age": validated_person.age,
                            "sex": validated_person.gender.lower(),
                            "bmi": round(speech_bmi, 2),
                            "children": validated_person.children,
                            "smoker": validated_person.smoker.lower(),
                            "region": validated_person.region.lower(),
                        }])
                        #st.write(st.session_state["speech_person_df"])
                        predicted_charge = predict_cost(model, st.session_state["speech_person_df"])
                        st.header(f"Koszt Twojego ubezpieczenia wyniesie {predicted_charge} {CURRENCY}")
                
                # Wskazówki do obniżenia kosztów
                if st.session_state["person"] is not None:
                    # Wskazówki do obniżenia kosztów
                    person = st.session_state["person"]  # pobieranie obiektu person z sesji
                    speech_bmi = st.session_state["speech_bmi"]
                    if person.smoker:
                        # kod do obliczeń i wskazówek, używając person...
                        speech_person_df_no_smoke = st.session_state["speech_person_df"].copy()
                        speech_person_df_no_smoke["smoker"] = "no"
                        cost_no_smoke = predict_cost(model, speech_person_df_no_smoke)
                        if cost_no_smoke < predicted_charge:
                            with st.spinner("Czekaj, generuję wskazówki do obniżenia kosztu..."):
                                time.sleep(2.5)
                                st.markdown(f"- Przestań palić, a Twój koszt ubezpieczenia wyniesie {cost_no_smoke} {CURRENCY}!")
                if st.session_state["speech_bmi"] is not None:
                    speech_bmi = st.session_state["speech_bmi"]
                    if person.height > 3.0:
                        person.height = person.height / 100
                    if speech_bmi > 24.9:
                        lose_weight_changes = [5, 10, 15, 20]  # Możliwe zmiany wagi
                        reduce_cost_and_display_message(st.session_state["speech_person_df"], person.weight, person.height, lose_weight_changes, "lower") 
                    elif speech_bmi < 18.5:
                        increase_weight_changes = [-5, -10, -15, -20]
                        reduce_cost_and_display_message(st.session_state["speech_person_df"], person.weight, person.height, increase_weight_changes, "increase")
                    else:
                        st.markdown("- Twoje BMI jest w normie i nie podwyższa kosztu ubezpieczenia.")                  
    else:
        st.markdown(
        "<h5>Wypełnij formularz</h5>",
        unsafe_allow_html=True
    )
        
        if 'age' not in st.session_state:
            st.session_state.age = 18
        if 'gender' not in st.session_state:
            st.session_state.gender = 'mężczyzna'
        if 'height' not in st.session_state:
            st.session_state.height = 1.7
        if 'weight' not in st.session_state:
            st.session_state.weight = 70
        if 'children' not in st.session_state:
            st.session_state.children = 0
        if 'smoker' not in st.session_state:
            st.session_state.smoker = 'nie'
        if 'live_place' not in st.session_state:
            st.session_state.live_place = 'południowy wschód'

        st.session_state.age = st.number_input("Wiek", min_value=0, max_value=100, value=st.session_state.age, step=1)
        st.session_state.gender = st.radio('Płeć', ['mężczyzna', 'kobieta'], index=0 if st.session_state.gender == 'mężczyzna' else 1)
        st.session_state.height = st.number_input("Wzrost", min_value=0.1, max_value=2.30, step=0.01, value=st.session_state.height)
        st.session_state.weight = st.number_input("Waga", min_value=2, max_value=180, step=1, value=st.session_state.weight)
        st.session_state.children = st.number_input("Dzieci", min_value=0, max_value=12, step=1, value=st.session_state.children)
        st.session_state.smoker = st.radio('Jestem osobą palącą', ['tak', 'nie'], index=1 if st.session_state.smoker == 'nie' else 0)
        st.session_state.live_place = st.selectbox('Region', ['południowy wschód', 'południowy zachód', 'północny wschód', 'północny zachód'], index=['południowy wschód', 'południowy zachód', 'północny wschód', 'północny zachód'].index(st.session_state.live_place))

        write_bmi = st.session_state.weight / (st.session_state.height ** 2)

        # Tworzenie DataFrame dla modelu
        write_person_df = pd.DataFrame([{
            "age": st.session_state.age,
            "sex": st.session_state.gender,
            "bmi": round(write_bmi, 2),
            "children": st.session_state.children,
            "smoker": st.session_state.smoker,
            "region": st.session_state.live_place,
        }])

        #with st.sidebar:
        # Obliczanie kosztu i dostarczanie rekomendacji
        if st.button("Oblicz koszt"):
            with st.spinner("Czekaj, obliczam..."):
                predicted_charge = predict_cost(model, write_person_df)
                # Sztuczne opóźnienie dla demonstrowania spinnera działania
                
                st.header(f"Koszt Twojego ubezpieczenia wyniesie {predicted_charge} {CURRENCY}")

            weight = st.session_state.weight
            height = st.session_state.height
            smoker = st.session_state.smoker

            # Wskazówki do obniżenia kosztów
            if smoker == "tak":
                write_person_df_no_smoke = write_person_df.copy()
                write_person_df_no_smoke["smoker"] = "no"
                cost_no_smoke = predict_cost(model, write_person_df_no_smoke)
                with st.spinner("Czekaj, generuję wskazówki do obniżenia kosztu..."):
                    time.sleep(2.5)
                    st.markdown(f"- Przestań palić, a Twój koszt ubezpieczenia wyniesie {cost_no_smoke} {CURRENCY}!")

            if write_bmi > 24.9:
                lose_weight_changes = [5, 10, 15, 20]  # Możliwe zmiany wagi
                reduce_cost_and_display_message(write_person_df, weight, height, lose_weight_changes, "lower") 
            elif write_bmi < 18.5:
                increase_weight_changes = [-5, -10, -15, -20]
                reduce_cost_and_display_message(write_person_df, weight, height, increase_weight_changes, "increase")
            else:
                st.markdown("- Twoje BMI jest w normie i nie podwyższa kosztu ubezpieczenia.") 



