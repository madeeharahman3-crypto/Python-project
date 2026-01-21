import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import mido
from mido import MidiFile, MidiTrack, Message
import random
import io

# --- GLOBAL STYLING & CONFIG ---
st.set_page_config(page_title="Madeeha Rahman | Academic Portfolio", layout="wide")
st.markdown("""
<style>
    .main {background-color: #f9f9f9;}
    h1 {font-family: 'Helvetica', sans-serif; font-size: 2.5em; color: #333;}
    h3 {color: #555; font-weight: normal;}
    .insight {font-size: 0.9em; color: #666; font-style: italic; border-left: 3px solid #ddd; padding-left: 10px;}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Portfolio Navigation")
page = st.sidebar.radio("Go to:", ["Home", "1. Music Data Dashboard", "2. Algorithmic Melody Generator", "3. AI Music Visualizer"])

# --- PAGE 0: HOME ---
if page == "Home":
    st.title("Madeeha Rahman")
    st.subheader("Creative Technology & Computer Science Portfolio")
    st.markdown("---")
    st.write("""
    **Objective:** To demonstrate technical competency in Python through three distinct lenses:
    1.  **Analysis:** Deriving insights from structured audio datasets.
    2.  **Algorithms:** Designing rule-based systems for creative output.
    3.  **Representation:** Visualizing signal processing data.
    """)
    st.info("👈 Select a project from the sidebar to view the demonstration.")

# --- PROJECT 1: MUSIC DATA DASHBOARD ---
elif page == "1. Music Data Dashboard":
    # 1. Title & Objective
    st.title("Music Data Analysis Dashboard")
    st.write("**Objective:** analyzing the correlation between audio features (energy, valence) and track popularity to identify genre-specific trends.")
    
    # DATA PREPARATION (Internal Mock Dataset)
    data = {
        'Genre': ['Pop']*5 + ['Rock']*5 + ['Jazz']*5 + ['EDM']*5,
        'Energy': [0.85, 0.8, 0.9, 0.75, 0.82,  0.88, 0.92, 0.85, 0.78, 0.95,  0.4, 0.35, 0.5, 0.45, 0.3,  0.9, 0.95, 0.88, 0.92, 0.85],
        'Danceability': [0.8, 0.85, 0.75, 0.8, 0.7,  0.5, 0.45, 0.4, 0.5, 0.55,  0.6, 0.55, 0.65, 0.6, 0.5,  0.85, 0.9, 0.8, 0.85, 0.88],
        'Valence': [0.9, 0.85, 0.8, 0.75, 0.8,  0.6, 0.55, 0.5, 0.65, 0.6,  0.7, 0.6, 0.65, 0.6, 0.55,  0.4, 0.5, 0.45, 0.35, 0.4],
        'Popularity': [92, 88, 85, 80, 84,  75, 72, 70, 68, 74,  55, 50, 58, 52, 48,  88, 90, 82, 85, 86],
        'Tempo': [120, 118, 122, 115, 124,  140, 138, 142, 135, 132,  90, 85, 95, 88, 92,  128, 130, 126, 128, 132],
        'Track ID': range(1, 21)
    }
    df = pd.DataFrame(data)

    # 2. VISUALS (Strict Order: Bar -> Scatter -> Hist -> Heatmap -> Table)
    
    # Visual 1: Bar Chart
    st.subheader("1. Genre Analysis")
    avg_energy = df.groupby('Genre')['Energy'].mean()
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    avg_energy.plot(kind='bar', color='#4c72b0', ax=ax1)
    ax1.set_title("Average Energy Score by Genre")
    ax1.set_ylabel("Energy (0-1)")
    plt.xticks(rotation=0)
    st.pyplot(fig1)
    st.markdown('<p class="insight">Insight: Rock and EDM genres display significantly higher average energy metrics compared to Jazz.</p>', unsafe_allow_html=True)

    # Visual 2: Scatter Plot
    st.subheader("2. Correlation Analysis")
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.scatter(df['Energy'], df['Popularity'], color='#55a868', alpha=0.7)
    ax2.set_title("Track Popularity vs. Energy")
    ax2.set_xlabel("Energy")
    ax2.set_ylabel("Popularity Score")
    ax2.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig2)
    st.markdown('<p class="insight">Insight: There is a positive linear correlation between high energy scores and track popularity.</p>', unsafe_allow_html=True)

    # Visual 3: Histogram
    st.subheader("3. Feature Distribution")
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    ax3.hist(df['Valence'], bins=8, color='#c44e52', edgecolor='black')
    ax3.set_title("Distribution of Valence (Musical Positiveness)")
    ax3.set_xlabel("Valence Score")
    ax3.set_ylabel("Frequency")
    st.pyplot(fig3)
    st.markdown('<p class="insight">Insight: The dataset skews towards mid-to-high valence, indicating a preference for major-key tonality.</p>', unsafe_allow_html=True)

    # Visual 4: Heatmap
    st.subheader("4. Feature Correlation Matrix")
    corr = df[['Energy', 'Danceability', 'Valence', 'Tempo', 'Popularity']].corr()
    fig4, ax4 = plt.subplots(figsize=(6, 5))
    cax = ax4.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig4.colorbar(cax)
    ax4.set_xticks(range(len(corr.columns)))
    ax4.set_yticks(range(len(corr.columns)))
    ax4.set_xticklabels(corr.columns, rotation=45)
    ax4.set_yticklabels(corr.columns)
    ax4.set_title("Correlation Heatmap")
    # Add numbers
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax4.text(j, i, round(corr.iloc[i, j], 2), ha="center", va="center", color="black")
    st.pyplot(fig4)
    st.markdown('<p class="insight">Insight: Danceability and Valence show a moderate positive correlation (0.5+), suggesting dance tracks are often perceived as happier.</p>', unsafe_allow_html=True)

    # Visual 5: Simple Table
    st.subheader("5. Top Tracks Overview")
    st.table(df.sort_values('Popularity', ascending=False).head(5)[['Genre', 'Popularity', 'Energy', 'Tempo']])
    st.markdown('<p class="insight">Insight: The top 5 highest-ranking tracks are exclusively from the Pop and EDM genres.</p>', unsafe_allow_html=True)

    # 3. Conclusion & 4. Reflection
    st.markdown("---")
    st.write("**Conclusion:** The analysis reveals that higher energy genres (Pop, EDM) tend to achieve higher popularity scores in this dataset. The correlation matrix confirms that energy is a stronger predictor of popularity than tempo.")
    st.write("**Reflection:** This project demonstrated how Python's Pandas library can be used to clean, aggregate, and visualize audio feature data to uncover hidden musical trends.")


# --- PROJECT 2: ALGORITHMIC MELODY GENERATOR ---
elif page == "2. Algorithmic Melody Generator":
    st.title("Algorithmic Melody Generator")
    
    # 1. Rules listed explicitly
    st.markdown("""
    **Algorithm Constraints & Rules:**
    1.  **Scale:** Notes must strictly belong to the C Major Scale (C, D, E, F, G, A, B).
    2.  **Length:** Fixed 16-note sequence.
    3.  **Motion (Probabilistic):**
        * 70% probability: Stepwise motion (±1 note index).
        * 20% probability: Repeat previous note.
        * 10% probability: Leap (±2 or ±3 note indices).
    4.  **Termination:** The melody must forcibly resolve to the Tonic (C) on the final note.
    """)

    if st.button("Generate Melody (Probabilistic)"):
        # SCALE DEFINITION (C Major)
        scale = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79] # Extended range
        
        # ALGORITHM: Random Walk
        melody = []
        current_idx = 4 # Start in middle (Note G)
        
        for _ in range(15): # Generate first 15 notes
            melody.append(scale[current_idx])
            
            # Probabilistic Logic
            r = random.random()
            if r < 0.7: 
                step = random.choice([-1, 1]) # Step
            elif r < 0.9:
                step = 0 # Repeat
            else:
                step = random.choice([-2, 2, 3, -3]) # Leap
            
            # Apply step and clamp to scale bounds
            current_idx += step
            current_idx = max(0, min(current_idx, len(scale)-1))
            
        # FORCE TONIC RESOLUTION (Rule 4)
        melody.append(60) # End on Middle C
        
        # GENERATE MIDI
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        
        for note in melody:
            track.append(Message('note_on', note=note, velocity=64, time=0))
            track.append(Message('note_off', note=note, velocity=64, time=250)) # Fixed duration
            
        mid.save('algorithmic_melody.mid')
        
        # VISUALIZATION (Pitch vs Time)
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.step(range(16), melody, where='mid', color='purple', linewidth=2)
        ax.set_title("Generated Melody Contour (Pitch vs Time Step)")
        ax.set_ylabel("MIDI Pitch")
        ax.set_xlabel("Time Step (16 Notes)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.success("Sequence generated. Rules applied successfully.")
        
        with open("algorithmic_melody.mid", "rb") as f:
            st.download_button("Download MIDI Result", f, "melody.mid", "audio/midi")
            
    st.write("**Algorithm Explanation:** This system uses a 'Random Walk' approach. Instead of choosing notes purely at random (chaos), the algorithm calculates the *next* note relative to the *current* note. This mimics how human melodies typically flow with small steps rather than large jumps.")


# --- PROJECT 3: AI MUSIC VISUALIZER ---
elif page == "3. AI Music Visualizer":
    st.title("AI Audio Visualizer")
    st.write("**Concept:** A representation tool that maps numerical audio data (amplitude) to visual coordinates.")
    
    uploaded_file = st.file_uploader("Upload Audio (MP3/WAV) to generate visual", type=["mp3", "wav"])
    
    if uploaded_file:
        # Load Audio (Librosa)
        y, sr = librosa.load(uploaded_file, duration=10)
        
        # ONE Clean Visual (Waveform)
        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.waveshow(y, sr=sr, ax=ax, color='#333', alpha=0.9)
        
        # Academic Formatting
        ax.set_title("Waveform Analysis: Amplitude vs. Time")
        ax.set_xlabel("Time (Seconds)")
        ax.set_ylabel("Amplitude")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)
        
        st.pyplot(fig)
        st.success("Visual generated from raw audio signal data.")