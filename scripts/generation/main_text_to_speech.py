import pyttsx3

def text_to_speech(text):
    try:

        engine = pyttsx3.init()
        if engine is None:
            print("Ошибка инициализации движка pyttsx3!")
            return

        engine.setProperty('rate', 250)
        engine.setProperty('volume', 1)

        voices = engine.getProperty('voices')

        for voice in voices:
            if "Irina" in voice.name:  # Проверка на наличие русского голоса
                engine.setProperty('voice', voice.id)
                break

        engine.say(text)
        engine.runAndWait()

    except Exception as e:
        print(f"Ошибка при выполнении text_to_speech: {e}")

def main():
    text_to_speech("Да капитан")

if __name__ == "__main__":
    main()
