# app.py
import streamlit as st
import time
from inference import get_context, query_model

st.set_page_config(page_title="THWS Chatbot (Prototyp)", layout="centered")

st.title("🎓 THWS Chatbot (Prototyp)")
st.write("Gib deine Frage ein, und der Bot sucht im THWS-Kontext und antwortet.")

# Eingabe
question = st.text_input("Frage:", placeholder="z. B. Wie melde ich mich für Prüfungen an?")

# Auswahl Modell (optional)
model = st.selectbox("Modell:", ["mistral:latest", "phi4:latest", "gemma3:27b", "deepseek-r1:32b"])

if st.button("Absenden") and question.strip():
    with st.spinner("Suche Kontext…"):
        context = get_context(question)
    if not context:
        st.warning("Kein Kontext gefunden. Frage bitte anders formulieren.")
    else:
        st.markdown("**⌛ Kontext:**")
        st.text_area("", context, height=200)

        with st.spinner("Hole Antwort…"):
            start_time = time.time()
            answer = query_model(question, context, model_name=model)
            end_time = time.time()
            duration = end_time - start_time

        st.markdown("**💬 Antwort:**")
        st.success(answer)
        st.write(f"Antwortgenerierung dauerte {duration:.2f} Sekunden")