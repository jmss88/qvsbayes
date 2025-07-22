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
switch_interval = st.slider("Intervalo de cambio del oponente (si es dinámico)", 10, 1000, 100, step=10)
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

    q_values = {"cooperate": 0.0, "defect": 0.0}
    alpha_q, gamma_q, epsilon = 0.1, 0.9, 0.1
    results_q = {"A_wins": 0, "loss": 0}
    q_learning_rewards = []
    q_behaviors = []

    alpha_bayes = 1
    beta_bayes = 1
    results_bayes = {"A_wins": 0, "loss": 0}
    bayes_rewards = []
    bayes_beliefs = []
    bayes_behaviors = []
    switch_points = []

    def classify_behavior(a_action, b_action):
        if a_action == "defect" and b_action == "cooperate":
            return "Explotadora"
        elif a_action == "cooperate" and b_action == "defect":
            return "Sumisa"
        elif a_action == "defect" and b_action == "defect":
            return "Agresiva"
        elif a_action == "cooperate" and b_action == "cooperate":
            return "Cooperativa"
        else:
            return "Indefinida"

    for step in range(episodes):
        if opponent_mode == "Dinámico" and step % switch_interval == 0:
            switch_points.append(step)

        # Q-learning
        a_q = np.random.choice(ACTIONS) if np.random.rand() < epsilon else max(q_values, key=q_values.get)
        o_q = opponent_behavior(step)
        reward_q = 1 if a_q == "defect" and o_q == "cooperate" else 0
        results_q["A_wins" if reward_q == 1 else "loss"] += 1
        q_values[a_q] += alpha_q * (reward_q + gamma_q * max(q_values.values()) - q_values[a_q])
        q_learning_rewards.append(reward_q)
        q_behaviors.append(classify_behavior(a_q, o_q))

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
        bayes_rewards.append(reward_b)
        bayes_beliefs.append(p_coop)
        bayes_behaviors.append(classify_behavior(a_b, o_b))

    st.subheader("Resultados")
    df_results = pd.DataFrame({
        "Modelo": ["Q-Learning", "Q-Learning", "Bayesiano", "Bayesiano"],
        "Resultado": ["A_wins", "loss", "A_wins", "loss"],
        "Conteo": [results_q["A_wins"], results_q["loss"], results_bayes["A_wins"], results_bayes["loss"]]
    })
    st.dataframe(df_results, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        ax1.plot(np.cumsum(q_learning_rewards), label="Q-Learning (acumulado)")
        ax1.plot(np.cumsum(bayes_rewards), label="Bayesiano (acumulado)")
        for switch in switch_points:
            ax1.axvline(x=switch, color='gray', linestyle='--', linewidth=0.8)
        ax1.set_title("Curva de Ganancias Acumuladas")
        ax1.set_xlabel("Episodios")
        ax1.set_ylabel("Recompensa acumulada")
        ax1.legend()
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.plot(bayes_beliefs, color="orange")
        for switch in switch_points:
            ax2.axvline(x=switch, color='gray', linestyle='--', linewidth=0.8)
        ax2.set_title("Evolución de la creencia Bayesiana sobre cooperación")
        ax2.set_xlabel("Episodios")
        ax2.set_ylabel("P(cooperate)")
        st.pyplot(fig2)

    st.subheader("Comparación de A_wins")
    fig3, ax3 = plt.subplots()
    ax3.bar(["Q-Learning", "Bayesiano"], [results_q["A_wins"], results_bayes["A_wins"]], color=["#1f77b4", "#ff7f0e"])
    ax3.set_ylabel("A_wins")
    ax3.set_title("Comparación de Ganancias entre Modelos")
    st.pyplot(fig3)

    st.subheader("Perfiles conductuales acumulados")
    df_behavior = pd.DataFrame({
        "Q-Learning": pd.Series(q_behaviors).value_counts(),
        "Bayesiano": pd.Series(bayes_behaviors).value_counts()
    }).fillna(0).astype(int)
    st.dataframe(df_behavior)

    fig4, ax4 = plt.subplots()
    df_behavior.plot(kind='bar', ax=ax4)
    ax4.set_title("Comparación de estilos de conducta")
    ax4.set_ylabel("Frecuencia")
    st.pyplot(fig4)

    st.subheader("Evolución temporal de estilos de conducta")
    def temporal_profile(behaviors, label):
        chunks = [behaviors[i:i+100] for i in range(0, len(behaviors), 100)]
        summary = []
        for block in chunks:
            counts = pd.Series(block).value_counts()
            summary.append(counts)
        return pd.DataFrame(summary).fillna(0)

    q_temp = temporal_profile(q_behaviors, "Q-Learning")
    bayes_temp = temporal_profile(bayes_behaviors, "Bayesiano")

    fig5, ax5 = plt.subplots(figsize=(10, 4))
    q_temp.plot(ax=ax5)
    ax5.set_title("Q-Learning: evolución de estilos de conducta (por bloques de 100 episodios)")
    ax5.set_xlabel("Bloque de 100 episodios")
    ax5.set_ylabel("Frecuencia")
    st.pyplot(fig5)

    fig6, ax6 = plt.subplots(figsize=(10, 4))
    bayes_temp.plot(ax=ax6)
    ax6.set_title("Bayesiano: evolución de estilos de conducta (por bloques de 100 episodios)")
    ax6.set_xlabel("Bloque de 100 episodios")
    ax6.set_ylabel("Frecuencia")
    st.pyplot(fig6)
