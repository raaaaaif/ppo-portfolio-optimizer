# PPO-Based Portfolio Optimization Bot

This project trains a deep reinforcement learning agent using Proximal Policy Optimization (PPO) to manage a stock portfolio. It includes a custom Gym-style environment, graph neural network architecture, and a data pipeline for Yahoo Finance.

---

## Features

### Reinforcement Learning
- Implements PPO using Stable Baselines3 and PyTorch
- Custom agent adapted for portfolio weight optimization

### Custom Trading Environment
- Built from scratch using `gym`
- Includes reward shaping, weight constraints, and cash balance tracking

### Graph Neural Network Architecture
- Uses PyTorch Geometric and RGCNConv
- EIIE-inspired layout for multi-asset learning
- Encodes relationships between stocks (e.g., industry sectors)

### Financial Data Pipeline
- Loads historical OHLCV data via Yahoo Finance
- Designed for flexibility and clean preprocessing

---

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Launch the notebook:
   ```bash
   jupyter notebook ppo_stock_bot_demo.ipynb
   ```

---

## File Structure

```
traderlib/
├── core/
│   ├── data_loader.py
│   ├── agent.py
│   ├── model_arch.py
│   └── portfolio_env.py
├── settings/
│   ├── settings.py
│   └── market_lists.py
```

---

## Notes

This was built as a personal experiment in applying RL to finance. The code is modular and intended for further extension (e.g., transaction cost modeling, live deployment).
