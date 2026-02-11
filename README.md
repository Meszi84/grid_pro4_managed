# Grid Pro 4 – Managed Trading Engine (Bybit / CCXT)

⚠️ **Advanced / Experimental trading engine**  
This project is **not a beginner bot**. It is a **state-driven trading engine** designed to run safely on live exchanges with **minimal API usage**, **deterministic decisions**, and **clear separation between GRID and TREND regimes**.

# Disclaimer: This software is for educational purposes only. Use it at your own risk. The author is not responsible for any financial losses.
---

## 🔑 Key Principles

- **One decision per cycle** (no order spam, no hidden loops)
- **Explicit market state reconciliation on startup**
- **GRID mode ≠ TREND mode** (never mixed)
- **State machines instead of polling logic**
- **API-throttled execution** (safe for CCXT rate limits)
- **Fail-safe risk handling** (daily DD, liquidation safety, SL streak pause)

---

## 🧠 Strategy Overview

### GRID Mode (Sideways Market)
- Activated only when **ADX confirms range conditions**
- Grid is **planned**, not instantly flooded
- Orders are placed **gradually** (configurable)
- Grid rebuild happens **only on structural changes**
- No SL interference unless regime changes

### TREND Mode
- Triggered by **ADX trend detection**
- Grid orders are cancelled (LIMIT only, SL preserved)
- If position exists:
  - **Losing position → ATR-based Stop Loss (once)**
  - **Winning position → ATR%-based trailing**
- Position is allowed to **run naturally**
- No re-entry until regime stabilizes

---

## 🚀 Startup Reconciliation (Critical Feature)

On startup, the engine **does not trade immediately**.

It first checks:
- Existing positions
- Active Stop Loss / conditional orders
- Open grid orders
- Current market regime (ADX)
- Equity & risk limits

Only **after a coherent market state is established** does the engine decide what to do next.

This prevents:
- Accidental SL deletion
- Duplicate grid placement
- Immediate unwanted closes after restart

---

## 🛡 Risk Management

- Daily drawdown stop (hard pause)
- SL streak protection with cooldown
- Liquidation proximity detection
- Position-aware SL placement
- No blind market orders

---

## ⚙ Configuration

Key configuration groups:
- `exchange` – leverage, margin mode
- `market` – symbol, timeframe, polling
- `grid` – ATR logic, grid density, pacing
- `risk` – equity limits, drawdown rules
- `adx` – regime detection thresholds

All logic is **explicit and deterministic** – no magic numbers hidden in the code.

---

## 📦 Requirements

- Python 3.11
- Bybit Futures account
- API key with trading permissions

### Settings
500+$
```
"risk": {
  "risk_per_order_pct": 0.003,      // 0.3%
  "grid_equity_ratio": 0.75
},
"grid": {
  "order_count_cap": 24
}
```

400$-500$
```
"risk": {
  "risk_per_order_pct": 0.0025,     // 0.25%
  "grid_equity_ratio": 0.7
},
"grid": {
  "order_count_cap": 20
}
```

300$-400$
```
"risk": {
  "risk_per_order_pct": 0.002,      // 0.2%
  "grid_equity_ratio": 0.65
},
"grid": {
  "order_count_cap": 16
}
```

250$ setting:
```
risk:
  grid_equity_ratio: 0.6
  risk_per_order_pct: 0.003

grid:
  order_count_cap: 8
```

150$ setting:
```
risk:
  grid_equity_ratio: 0.55
grid:
  order_count_cap: 8
risk:
  risk_per_order_pct: 0.0015   # 0.15%
```
Minimum amount 80$
```
"risk": {
  "risk_per_order_pct": 0.0015,   // 0.15%
  "grid_equity_ratio": 0.5
},
"grid": {
  "order_count_cap": 6
}
```
---
<img width="2217" height="951" alt="Képernyőkép 2026-02-09 190337" src="https://github.com/user-attachments/assets/493975ad-60bd-4ced-ae06-10239255528f" />

## Requirements
- Python 3.11
- Bybit Futures account
- API key with trading permissions

### `.env.example`
```env
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here

DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

ENV=production
```
## Start.sh
```
#!/bin/bash
source .venv/bin/activate
python grid_pro4_managed.py
```
## 2
```
chmod +x start.sh
```
## Restart.sh
```
#!/bin/bash
pkill -f grid_pro4_managed.py
sleep 2
./start.sh
```
## Systemd
```
[Unit]
Description=Grid Pro Trading Bot
After=network.target

[Service]
User=root
WorkingDirectory=/root/grid_pro4_managed
ExecStart=/root/grid_pro4_managed/start.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```
## Activate
```
systemctl daemon-reload
systemctl enable gridbot
systemctl start gridbot
```

## Installation
```bash
git clone https://github.com/Meszi84/grid_pro4_managed.git
cd grid_pro4_managed
pip install -r requirements.txt
