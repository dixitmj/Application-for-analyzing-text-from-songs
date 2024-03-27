import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
from transformers import pipeline

# Initialize the question-answering pipeline
qa_pipeline = pipeline("question-answering")

# Function to convert MP3 to WAV
def convert_mp3_to_wav(mp3_file, wav_file):
    sound = AudioSegment.from_mp3(mp3_file)
    sound.export(wav_file, format="wav")

# Function to transcribe audio to text using Sphinx
def transcribe_audio(wav_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_file) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_sphinx(audio_data)
        return text
    except sr.RequestError as e:
        st.error(f"Recognition request failed: {e}")
    except sr.UnknownValueError:
        st.error("Speech recognition could not understand audio")
    return None

# Function to use a QA model to answer questions based on the transcribed text
def get_answer(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Main function to create the Streamlit app
def main():
    st.title("👾 Audio Transcription 👾")

    # File upload section
    uploaded_file = st.file_uploader("Upload an audio file (MP3)", type=["mp3"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/mp3')
        st.write("Uploaded audio file")

        # Convert MP3 to WAV and transcribe audio to text
        wav_file = "temp_audio.wav"
        convert_mp3_to_wav(uploaded_file, wav_file)
        transcribed_text = transcribe_audio(wav_file)

        # Display transcribed text
        if transcribed_text:
            st.subheader("Transcribed Text:")
            st.write(transcribed_text)

            # Ask user for questions
            user_question = st.text_input("Ask any question about the audio:")

            # Answer user's question based on transcribed text
            if user_question:
                answer = get_answer(user_question, transcribed_text)
                st.subheader("Answer:")
                st.write(answer)
        else:
            st.error("Transcription failed. Please try again.")

if __name__ == "__main__":
    main()
