import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# Configuración inicial
# --------------------------
st.set_page_config(page_title="Simulador Q-Learning vs Bayesiano", layout="wide")
st.title("Simulador: Cooperación vs Competencia (Q-Learning vs Bayesiano)")
st.markdown("""
Este simulador compara el rendimiento de dos modelos de aprendizaje:
- **Q-Learning**: aprende por refuerzo.
- **Bayesiano**: infiere la probabilidad de cooperación del oponente.

El entorno puede ser **estático** (el oponente se comporta igual siempre) o **dinámico** (cambia de estrategia cada cierto número de episodios).
""")

# --------------------------
# Controles de interfaz
# --------------------------
episodes = st.slider("Número de episodios", 500, 5000, 2000, step=500)
switch_interval = st.slider("Intervalo de cambio del oponente (si es dinámico)", 100, 1000, 500, step=100)
opponent_mode = st.selectbox("Modo del oponente", ["Estático", "Dinámico"])

run_simulation = st.button("Ejecutar simulación")

# --------------------------
# Simulación de modelos
# --------------------------
def opponent_behavior(step):
    if opponent_mode == "Estático":
        return "cooperate" if np.random.rand() < 0.4 else "defect"
    else:
        phase = (step // switch_interval) % 2
        prob = 0.7 if phase == 0 else 0.3
        return "cooperate" if np.random.rand() < prob else "defect"

if run_simulation:
    ACTIONS = ["cooperate", "defect"]

    # Q-Learning
    q_values = {"cooperate": 0.0, "defect": 0.0}
    alpha_q, gamma_q, epsilon = 0.1, 0.9, 0.1
    results_q = {"A_wins": 0, "loss": 0}

    # Bayesiano
    alpha_bayes = 1
    beta_bayes = 1
    results_bayes = {"A_wins": 0, "loss": 0}

    for step in range(episodes):
        # Q-learning
        a_q = np.random.choice(ACTIONS) if np.random.rand() < epsilon else max(q_values, key=q_values.get)
        o_q = opponent_behavior(step)
        reward_q = 1 if a_q == "defect" and o_q == "cooperate" else 0
        results_q["A_wins" if reward_q == 1 else "loss"] += 1
        q_values[a_q] += alpha_q * (reward_q + gamma_q * max(q_values.values()) - q_values[a_q])

        # Bayesiano
        p_coop = alpha_bayes / (alpha_bayes + beta_bayes)
        a_b = "defect" if p_coop > 0.0 else "cooperate"
        o_b = opponent_behavior(step)
        reward_b = 1 if a_b == "defect" and o_b == "cooperate" else 0
        results_bayes["A_wins" if reward_b == 1 else "loss"] += 1
        if o_b == "cooperate":
            alpha_bayes += 1
        else:
            beta_bayes += 1

    # --------------------------
    # Mostrar resultados
    # --------------------------
    st.subheader("Resultados")
    df_results = pd.DataFrame({
        "Modelo": ["Q-Learning", "Q-Learning", "Bayesiano", "Bayesiano"],
        "Resultado": ["A_wins", "loss", "A_wins", "loss"],
        "Conteo": [results_q["A_wins"], results_q["loss"], results_bayes["A_wins"], results_bayes["loss"]]
    })

    st.dataframe(df_results, use_container_width=True)

    fig, ax = plt.subplots()
    ax.bar(["Q-Learning", "Bayesiano"], [results_q["A_wins"], results_bayes["A_wins"]], color=["#1f77b4", "#ff7f0e"])
    ax.set_ylabel("A_wins")
    ax.set_title("Comparación de Ganancias entre Modelos")
    st.pyplot(fig)
