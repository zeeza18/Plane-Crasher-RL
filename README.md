# **Plane-Crasher-RL 🚀**  

This repository implements **Reinforcement Learning (RL)** for training an agent to navigate a flying plane through obstacles while adapting to progressively increasing difficulty. The project compares **Deep Q-Networks (DQN) and Double Deep Q-Networks (DDQN)** to analyze learning efficiency, adaptability, and stability in a dynamically evolving environment.

---

## **📌 Features**
- **Custom RL environment inspired by Flappy Bird but with major enhancements**
- **Progressive difficulty scaling** (speed increases, zig-zag motion)
- **Physics-based motion** (gravity, acceleration, velocity constraints)
- **Implemented DQN and DDQN models** for training & evaluation
- **Animated environment with moving obstacles, dynamic sky, and game-over effects**
- **Performance tracking with real-time plots and logs**

---

## **🔧 Setup and Installation**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/zeeza18/Plane-Crasher-RL.git
cd Plane-Crasher-RL
```

### **2️⃣ Create and Activate a Virtual Environment**  
```bash
# Create a virtual environment (optional but recommended)
python -m venv plane_rl_env

# Activate the virtual environment
# Windows
plane_rl_env\Scripts\activate
# MacOS/Linux
source plane_rl_env/bin/activate
```

### **3️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

The `requirements.txt` includes the following key libraries:  
- `gymnasium` - Reinforcement learning environment  
- `torch` - Deep learning framework for DQN and DDQN  
- `numpy` - Numerical computations  
- `matplotlib` - Data visualization  
- `pandas` - Data handling and analysis  

---

## **🚀 How to Run the Project**  

### **1️⃣ Train the Model**  
To train **DQN**, run:  
```bash
python train_dqn.py
```

To train **DDQN**, run:  
```bash
python train_ddqn.py
```
- This will start training the **DQN/DDQN models** on the plane environment.  
- Training logs and performance graphs will be saved in the `runs/` directory.  

### **2️⃣ Test the Trained Model**  
To test a trained **DQN** model:  
```bash
python test_dqn.py
```

To test a trained **DDQN** model:  
```bash
python test_ddqn.py
```
- Runs the trained model in inference mode.  
- The agent attempts to **navigate through obstacles using learned strategies**.  

### **3️⃣ Visualize Training Performance**  
```bash
python plot_results.py
```
- Generates **performance graphs** for training progression.  
- Saves plots in the `plots_dqn/` and `plots_ddqn/` directories.  

---

## **📂 Project Structure**
```
Plane-Crasher-RL/
│── animation/                  # Game animations (plane crash, explosion, movement)
│── assets/sprites/              # Visual assets (plane, buildings, background)
│── bird/                        # Bird-based environment components (legacy)
│── checkpoints/                 # Trained model checkpoints
│── checkpoints_dqn/             # DQN-specific saved models
│── checkpoints_ddqn/            # DDQN-specific saved models
│── environment/                 # Custom RL environment for plane navigation
│── models/                      # Neural network architectures for DQN & DDQN
│── plots_dqn/                   # Training performance graphs for DQN
│── plots_ddqn/                  # Training performance graphs for DDQN
│── runs/                        # Logs and TensorBoard files
│── utils/                       # Helper functions (sound, thrust effects, utilities)
│── .gitignore                   # Ignored files for Git
│── LICENSE                      # Project license
│── README.md                    # Project documentation
│── __init__.py                  # Initialization file
│── requirements.txt             # Required dependencies
│── sound_manager.py             # Manages in-game sound effects
│── test_dqn.py                  # Testing script for DQN
│── test_ddqn.py                 # Testing script for DDQN
│── thrust_effect.py             # Thrust effects for plane physics
│── train_dqn.py                 # Training script for DQN
│── train_ddqn.py                # Training script for DDQN
```

---

## **📊 Research Findings**

Our comparative analysis of DQN and DDQN revealed several key insights:

- **Performance Peaks**: DQN achieved a higher peak score of 30 buildings passed, compared to 27 buildings for DDQN
- **Adaptation Speed**: DDQN adapted to difficulty changes faster (100-150 episodes) than DQN (200-300 episodes)
- **Learning Stability**: DDQN exhibited smoother learning curves and better stability when facing new constraints
- **Recovery Patterns**: Both algorithms showed a mountain-like formation in performance graphs due to progressive difficulty scaling

The main difference in performance can be attributed to DQN's overestimation bias, which led to erratic decision-making when encountering new motion constraints. DDQN's separate action selection and evaluation mechanisms resulted in smoother transitions and quicker adaptation to changing conditions.

---

## **📈 Expected Results**

During training, the model will show a **mountain-like pattern** in performance graphs due to environmental updates:  
- **DQN achieves higher peak scores but takes longer to adapt**  
- **DDQN adapts faster but shows a steadier increase in scores**  
- **Both algorithms undergo performance drops and recover as they learn**  

## **🎥 Model Demonstrations**

### DQN Model Performance
[![DQN Model Performance](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID_HERE)

### DDQN Model Performance
[![DDQN Model Performance](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID_HERE)

---

## **🛠️ Future Enhancements**
- **Extend training beyond 2,500 episodes** to analyze long-term performance  
- **Introduce time-of-day variations** (morning and evening conditions) to assess adaptability under different visual scenarios
- **Incorporate advanced RL techniques** such as Genetic Algorithms (GA) or Proximal Policy Optimization (PPO)
- **Implement neural network optimizations** including Transformer-based RL models

---

## **🤝 Contributions**
Contributions are welcome! Feel free to:  
- Open an issue for **bugs or feature requests**  
- Submit **pull requests** to improve the model or environment  

---

## **📜 License**
This project is licensed under the **MIT License** – feel free to modify and distribute!  

---

### **🚀 Happy Reinforcement Learning!**