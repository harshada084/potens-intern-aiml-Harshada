"""
src/translate.py

Multilingual flow: detects the language of the incoming question,
translates it to English if needed, runs it through the normal ask()
pipeline, then translates the answer back to the original language.

This is a translation-at-the-boundary approach (explicitly allowed by
the assignment brief for a 24-hour build) rather than a fully
multilingual embedding/retrieval pipeline.
"""

import os
from google import genai
from dotenv import load_dotenv
from langdetect import detect, LangDetectException
from qa import ask

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

GENERATION_MODEL = "gemini-2.5-flash"


def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "en"


def translate_text(text, target_language_name):
    prompt = (
        f"Translate the following text into {target_language_name}. "
        f"Only output the translation, nothing else.\n\n{text}"
    )
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )
    return response.text.strip()


# Minimal language-code to language-name map for common cases.
# langdetect returns ISO codes (e.g. "hi", "fr", "es"); Gemini works fine
# with either, but full names give more reliable translation results.
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh-cn": "Chinese",
    "ar": "Arabic",
    "ja": "Japanese",
    "pt": "Portuguese",
    "ru": "Russian",
}


def ask_multilingual(question):
    detected_code = detect_language(question)
    detected_name = LANGUAGE_NAMES.get(detected_code, detected_code)

    if detected_code == "en":
        # No translation needed
        result = ask(question)
        result["detected_language"] = "en"
        return result

    # Translate question -> English
    question_en = translate_text(question, "English")

    # Run the normal RAG pipeline in English
    result = ask(question_en)

    # Translate the answer back to the original language
    if not result["no_answer_found"]:
        result["answer"] = translate_text(result["answer"], detected_name)
    else:
        result["answer"] = translate_text(result["answer"], detected_name)

    result["detected_language"] = detected_code
    result["translated_question"] = question_en
    return result


if __name__ == "__main__":
    q = input("Ask a question (any language): ")
    result = ask_multilingual(q)
    print("\nDetected language:", result["detected_language"])
    if "translated_question" in result:
        print("Translated question:", result["translated_question"])
    print("\nANSWER:", result["answer"])
    print("\nCITATIONS:")
    for c in result.get("citations", []):
        print(f"  - {c['source']} ({c['chunk_ref']})")