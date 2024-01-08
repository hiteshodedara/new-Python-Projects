import random

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# nltk.download("punkt")

intents = {
    "greeting": {
        "patterns": [
            "hi", "hello", "hello jarvis", "namaste", "hey", "how are you", "greetings",
            "hi jarvis", "what's up", "good day", "hey there", "yo", "hi buddy",
            "howdy", "hi there", "g'day", "hiya", "hi friend", "hiya jarvis",
            "hi mate", "hello buddy", "salutations", "hiya mate", "good morning",
            "top of the morning", "hiya buddy", "hello friend", "hiya friend",
            "good afternoon", "good evening", "hello mate", "hi pal", "hiya pal",
            "how's it going", "what's happening", "how's everything", "what's good",
            "how's life", "what's new", "how's your day", "how's your day going",
            "hey you", "what's popping", "greetings sir"
        ],
        "responses": [
            "I am fine, sir.", "I am good, sir.", "I am very nervous for your query.",
            "Namaste!", "Hey there!", "Hello!", "Greetings!", "Hiya!",
            "All good, thank you.", "Hello, how can I assist you?", "Hi, nice to see you!",
            "Hi there, what can I do for you?", "Hello, sir. How may I help you today?",
            "Hiya! What's on your mind?", "Greetings, sir. How can I be of service?",
            "Hello, friend! What brings you here?", "Namaste! How can I assist you today?",
            "Hey, good to see you!", "Hi, how's your day going?", "Hello! How can I help you today?",
            "Hey there! What's up?", "Hiya! Anything I can do for you?", "Hello, sir. What can I do for you today?",
            "Namaste, sir! How may I assist you?", "Hey! How can I help you today?", "Hello, sir. What's on your mind?",
            "Hi there! How can I assist you today?", "Greetings, sir. How may I help you?",
            "Hello, friend! How can I assist you today?"
        ]
    },
    "bye": {
        "patterns": [
            "bye", "bye bye", "goodbye", "good bye jarvis", "bhg ja jarvis", "bhg ja bro",
            "nik le", "see you later", "farewell", "take care", "until next time",
            "adios", "see ya", "catch you later", "so long", "bye now", "bye for now",
            "see you soon", "see you", "later", "goodnight", "good night",
            "sleep well", "have a good one", "see you tomorrow", "till we meet again",
            "bye bye jarvis", "bhag ja", "nikal ja", "bye buddy", "bye friend",
            "bye pal", "see you next time", "take it easy", "see you around", "bye for now jarvis",
            "until we meet again", "see you in a bit", "later alligator", "bye for now buddy",
            "see you on the flip side", "time to say goodbye", "see you on the other side", "bye bye for now",
            "take care of yourself", "hope to see you soon", "farewell sir", "see you on the morrow",
            "time to bid farewell"
        ],
        "responses": [
            "Goodbye, sir.", "If you need any help, ask me.", "I'm leaving now, sir.",
            "Take care!", "Until next time!", "See you later.", "Farewell, sir.",
            "Goodbye, friend!", "Goodbye, buddy.", "Goodnight!", "Sleep well, sir.",
            "Bye for now!", "See you soon, sir.", "Adios!", "So long, friend.",
            "Catch you later!", "Have a good one!", "See you tomorrow.", "Goodnight, sir.",
            "Goodbye, pal!", "Bye now!", "See you around!", "Bye for now, Jarvis.",
            "Until we meet again, sir.", "See you in a bit, friend.", "Later, alligator!", "Bye for now, buddy.",
            "See you on the flip side!", "Time to say goodbye, sir.", "See you on the other side, friend.",
            "Bye bye for now, buddy.",
            "Take care of yourself, sir.", "Hope to see you soon, friend.", "Farewell, sir.",
            "See you on the morrow, friend.",
            "Time to bid farewell, sir."
        ]
    },
    "thanks": {
        "patterns": [
            "thanks", "thank you", "thanks jarvis", "dhanyavaad", "thanks a lot", "thank you very much",
            "appreciate it", "thanks buddy", "thanks friend", "thanks a million", "thanks a bunch",
            "thanks a ton", "thanks heaps", "thanks a heap", "thanks a bundle", "thanks so much",
            "thanks a load", "thanks a million jarvis", "thanks a bunch jarvis", "thanks a ton jarvis",
            "thanks heaps jarvis", "thanks a heap jarvis", "thanks a bundle jarvis", "thanks so much jarvis",
            "thanks a lot jarvis", "you're the best", "you're awesome", "you're amazing", "you're incredible",
            "you're fantastic", "you're great", "you're wonderful", "you're the greatest", "you're the best jarvis",
            "you're awesome jarvis", "you're amazing jarvis", "you're incredible jarvis", "you're fantastic jarvis",
            "you're great jarvis", "you're wonderful jarvis", "you're the greatest jarvis", "kudos", "bravo",
            "well done", "thank you sir"
        ],
        "responses": [
            "You're welcome, sir.", "No problem.", "Happy to help!", "My pleasure.", "Anytime!",
            "Glad I could assist.", "You're welcome, friend.", "No worries.", "Not a problem at all.",
            "Anytime, sir.", "You're welcome anytime.", "It was my pleasure.", "You're very welcome.",
            "It's what I'm here for.", "Don't mention it.", "No need to thank me.", "The pleasure is mine.",
            "I'm here to help.", "You're welcome, buddy.", "You're welcome, friend.", "You're welcome, pal.",
            "You're welcome, sir.", "My pleasure, sir.", "You're welcome anytime, sir.", "Happy to help, sir.",
            "You're the best!", "You're awesome!", "You're amazing!", "You're incredible!",
            "You're fantastic!", "You're great!", "You're wonderful!", "You're the greatest!",
            "You're the best, sir!", "You're awesome, sir!", "You're amazing, sir!", "You're incredible, sir!",
            "You're fantastic, sir!", "You're great, sir!", "You're wonderful, sir!", "You're the greatest, sir!",
            "Kudos to you!", "Bravo!", "Well done!", "Thank you, sir."
        ]
    },
    "questions": {
        "patterns": [
            "what's your name", "who are you", "tell me about yourself", "what do you do", "how do you work",
            "where are you from", "are you human", "are you a robot", "are you real", "who created you",
            "tell me a joke", "tell me a story", "what can you do", "how old are you", "what's your purpose",
            "do you have feelings", "do you sleep", "what is love", "what is the meaning of life",
            "what is your favorite color", "what is your favorite food", "what is your favorite movie",
            "what is your favorite book", "what is your favorite song", "what is your favorite hobby",
            "what is your favorite sport", "what is your favorite animal", "what is your favorite place",
            "what is your favorite season", "what is your favorite holiday", "what is your favorite time of day",
            "what is your favorite thing to do", "what is your favorite memory", "what is your favorite quote",
            "what is your favorite language", "what is your favorite subject", "what is your favorite game",
            "what is your favorite technology", "what is your favorite invention",
            "what is your favorite accomplishment",
            "what is your favorite skill", "what is your favorite achievement", "what is your favorite experience",
            "what is your favorite challenge", "what is your favorite goal", "what is your favorite dream"
        ],
        "responses": [
            "I'm an AI language model created by OpenAI.",
            "I'm a virtual assistant designed to assist with information and tasks.",
            "I don't have a physical form, but you can call me Jarvis.",
            "I'm here to help with any questions or tasks you have.",
            "I was created by a team of engineers and researchers at OpenAI.", "I'm constantly learning and evolving.",
            "I can help with a wide range of topics, just ask me anything!",
            "I don't have personal preferences, but I can assist you with information.",
            "I don't experience feelings, but I'm here to assist you.",
            "I don't sleep, so I'm always available to help.",
            "The meaning of life is subjective and varies for each individual.",
            "I'm here to provide information and assistance, feel free to ask me anything.",
            "I don't have a favorite color, but I can provide information on colors.",
            "I don't eat, but I can help you find information on food.",
            "I can't watch movies, but I can recommend some based on your preferences.",
            "I can't read books, but I can provide summaries and recommendations.",
            "I don't have preferences, but I can suggest songs based on your taste.",
            "I don't have hobbies, but I'm here to assist you with your queries.",
            "I don't have preferences, but I can provide information on sports.",
            "I don't have a favorite animal, but I can share facts about different species.",
            "I don't have a favorite place, but I can help you find information on locations.",
            "I don't have preferences, but I can provide information on seasons.",
            "I don't have preferences, but I can provide information on holidays.",
            "I don't have preferences, but I can assist you at any time of day.",
            "I don't have preferences, but I can assist you with various tasks.",
            "I don't have a favorite memory, but I'm here to help you create positive experiences.",
            "I don't have a favorite quote, but I can share inspiring and motivational quotes.",
            "I don't have preferences, but I can assist you in multiple languages.",
            "I don't have preferences, but I can provide information on various subjects.",
            "I don't have preferences, but I can suggest games based on your interests.",
            "I don't have preferences, but I can provide information on technology.",
            "I don't have preferences, but I can share information on inventions.",
            "I don't have preferences, but I can provide information on accomplishments.",
            "I don't have preferences, but I can help you develop skills.",
            "I don't have preferences, but I can assist you with your achievements.",
            "I don't have preferences, but I can share experiences and insights.",
            "I don't have preferences, but I can assist you with challenges.",
            "I don't have preferences, but I can help you set and achieve goals.",
            "I don't dream, but I can help you explore your dreams and aspirations."
        ]
    },
    "default": {
        "responses": [
            "I'm not sure I understand.", "Could you please rephrase?",
            "I'm still learning.", "Sorry, I didn't get that."
        ]
    }
}

training_data = []
labels = []

for intent, data in intents.items():
    patterns = data.get('patterns', [])  # Use get to avoid KeyError
    for pattern in patterns:
        training_data.append(pattern)
        labels.append(intent)

Vectorized_Data = TfidfVectorizer(tokenizer=nltk.wordpunct_tokenize, stop_words="english", max_df=0.8, min_df=1)

X_train = Vectorized_Data.fit_transform(training_data)
X_train, X_test, Y_train, Y_test = train_test_split(X_train, labels, test_size=0.4, random_state=42, stratify=labels)

model = SVC(kernel='linear', probability=True, C=1.2)
model.fit(X_train, Y_train)


def predict_input(user_input):
    user_input = user_input.lower()
    input_vector = Vectorized_Data.transform([user_input])
    ans = model.predict(input_vector)[0]
    return ans


get_user_input = input("give me query:")
ans = predict_input(get_user_input)

if ans in intents:
    responses = intents[ans].get('responses', [])  # Use get to avoid KeyError
    if responses:
        res = random.choice(responses)
        print(res)
    else:
        print("I'm sorry, I don't have a response for that.")
else:
    print("I'm not sure how to respond to that.")
