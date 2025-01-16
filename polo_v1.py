import ollama
import chromadb
import psycopg
from psycopg.rows import dict_row
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import pyttsx3

class VoiceTextProcessor:
    def __init__(self, sample_rate=44100, duration=10, output_file="recorded_audio.wav"):
        self.sample_rate = sample_rate
        self.duration = duration
        self.output_file = output_file

    def record_audio(self):
        print(f"Recording for {self.duration} seconds...")
        audio = sd.rec(int(self.sample_rate * self.duration), samplerate=self.sample_rate, channels=1, dtype="int16")
        sd.wait()  # Wait until the recording is finished
        write(self.output_file, self.sample_rate, audio)  # Save as WAV file
        print(f"Recording saved to {self.output_file}")

    def transcribe_audio(self, file_path):
        print("Loading Whisper model...")
        model = whisper.load_model("base")
        print("Transcribing audio...")
        result = model.transcribe(file_path)
        print("Transcription:")
        print(result["text"])
        return result["text"]

    def text_to_speech_spanish(self, text, voice_name=""):
        engine = pyttsx3.init()

        # Set a specific voice (e.g., for Spanish)
        voices = engine.getProperty("voices")
        for voice in voices:
            if voice_name.lower() in voice.name.lower():
                engine.setProperty("voice", voice.id)
                print(f"Using voice: {voice.name}")
                break
        engine.say(text)
        engine.runAndWait()


class MemoryAgent:
    def __init__(self):
        self.client = chromadb.Client()
        self.system_prompt = (
            'You are an AI assistant that has memory of every conversation you have ever had with this user. '
            'On every prompt from the user, the system has checked for any relevant messages you have had with the user. '
            'If any embedded previous conversations are attached, use them for context to responding to the user,'
            'if the context is relevant and useful to responding. If the recalled conversations are irrelevant'
            'disregard speaking about them and respond normally as an AI assistant. Do not talk about recalling conversations'
            'Just use any useful data from the previous conversations and respond normally as an intelligent Al assistant.'
        )
        self.convo = [{"role": "system", "content": self.system_prompt}]
        self.DB_PARAMS = {
            "dbname": "memory_agent",
            "user": "example_user",
            "password": "123456",
            # TODO: Change this to the IP address of the database
            "host": "192.168.1.142",
            "port": "5432"
        }

    def connect_db(self):
        conn = psycopg.connect(**self.DB_PARAMS)
        return conn

    def fetch_conversations(self):
        conn = self.connect_db()
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute("SELECT * FROM conversations")
            conversations = cursor.fetchall()
        conn.close()
        return conversations

    def store_conversations(self, prompt, response):
        conn = self.connect_db()
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO conversations (timestamp, prompt, response) VALUES (CURRENT_TIMESTAMP, %s, %s)", 
                (prompt, response)
            )
            conn.commit()
        conn.close()

    def strem_response(self, prompt):
        self.convo.append({"role": "user", "content": prompt})
        response = ''
        stream = ollama.chat(model="llama3.2", messages=self.convo, stream=True)
        print("ASSISTANT: \n")

        for chuck in stream:
            content = chuck["message"]["content"]
            response += content
            print(content, end="", flush=True)

        # add response to the conversation with audio
        processor = VoiceTextProcessor()
        processor.text_to_speech_spanish(text=str(response), voice_name="Spanish")

        print("\n")
        self.store_conversations(prompt=prompt, response=response)
        self.convo.append({"role": "assistant", "content": response})

    def create_vector_db(self, conversations):
        vector_db_name = "conversations"

        try:
            self.client.delete_collection(name=vector_db_name)
        except ValueError:
            pass

        vectir_db = self.client.create_collection(name=vector_db_name)

        for c in conversations:
            serialized_convo = f'prompt: {c["prompt"]}, response: {c["response"]}'
            response = ollama.embeddings(model='nomic-embed-text', prompt=serialized_convo)
            embedding = response["embedding"]

            vectir_db.add(
                ids=[str(c["id"])],
                embeddings=[embedding],
                documents=[serialized_convo]
            )

    def retrieve_embeddings(self, prompt):
        response = ollama.embeddings(model='nomic-embed-text', prompt=prompt)
        prompt_embedding = response["embedding"]

        vector_db = self.client.get_collection(name="conversations")
        results = vector_db.query(query_embeddings=[prompt_embedding], n_results=1)
        best_embedding = results["documents"][0][0]

        return best_embedding

agent = MemoryAgent()
conversations = agent.fetch_conversations()
agent.create_vector_db(conversations=conversations)
# print(fetch_conversations())

while True:
    voice_text_processor = VoiceTextProcessor()
    voice_text_processor.record_audio()
    prompt = voice_text_processor.transcribe_audio(voice_text_processor.output_file)
    # prompt = input("USER: \n")
    context = agent.retrieve_embeddings(prompt=prompt)
    prompt = f'USER PROMPT: {prompt} \nCONTEXT FROM EMEDDINGS: {context}'
    agent.strem_response(prompt=prompt)


