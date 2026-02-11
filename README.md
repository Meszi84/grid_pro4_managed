![CI](https://github.com/Meszi84/grid_pro4_managed/actions/workflows/python-app.yml/badge.svg)

# Grid Pro 4 – Managed Trading Engine (Bybit / CCXT)

⚠️ Advanced / Experimental Trading Engine

Grid Pro 4 is a state-driven trading engine designed for controlled live execution on Bybit Futures via CCXT.

This is not a beginner trading bot.

The system is built around deterministic state machines, strict regime separation (GRID vs TREND), and explicit startup reconciliation to prevent unintended actions after restarts.

---

## Disclaimer

This software is provided for educational purposes only.

Trading cryptocurrencies involves substantial risk.  
Use at your own risk.  
The author assumes no responsibility for financial losses.

---

# Core Design Principles

- One decision per cycle (no order spam)
- Deterministic execution (no hidden loops)
- Explicit startup reconciliation
- GRID mode and TREND mode are strictly separated
- API rate-limit safe (CCXT compatible)
- Structured risk controls
- No blind market orders

---

# Strategy Overview

## GRID Mode (Sideways Regime)

Activated only when ADX confirms range conditions.

- Gradual grid placement (configurable pacing)
- Grid rebuild only on structural change
- No SL interference unless regime flips
- Equity-aware order sizing

## TREND Mode

Activated when ADX detects directional momentum.

- Grid LIMIT orders are cancelled (SL preserved)
- Losing position → ATR-based stop loss (single placement)
- Winning position → ATR-percentage trailing logic
- No immediate re-entry after close
- Position allowed to run naturally

---

# Startup Reconciliation (Critical Safety Feature)

On startup, the engine does not trade immediately.

It validates:

- Existing open positions
- Active stop-loss / conditional orders
- Open grid orders
- Current ADX regime
- Equity state and risk thresholds

Only after a coherent state is established does execution continue.

This prevents:

- Duplicate grid placement
- Accidental SL removal
- Immediate unwanted position closure after restart

---

# Requirements

- Python 3.11
- Bybit USDT Perpetual account
- API key with trading permissions
- Minimum recommended equity: 80 USD

---

# Installation

```bash
git clone https://github.com/Meszi84/grid_pro4_managed.git
cd grid_pro4_managed

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
````

---

# Environment Configuration

Create a `.env` file based on the example below:

## `.env.example`

```env
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here

DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

ENV=production
```

---

# Running the Engine

```bash
cp .env.example .env
python grid_pro4_managed.py
```

---

# Risk Configuration Profiles

Below are example risk presets.
Adjust according to your equity and risk tolerance.

---

## 500 USD+

```json
"risk": {
  "risk_per_order_pct": 0.003,
  "grid_equity_ratio": 0.75
},
"grid": {
  "order_count_cap": 24
}
```

---

## 400–500 USD

```json
"risk": {
  "risk_per_order_pct": 0.0025,
  "grid_equity_ratio": 0.7
},
"grid": {
  "order_count_cap": 20
}
```

---

## 300–400 USD

```json
"risk": {
  "risk_per_order_pct": 0.002,
  "grid_equity_ratio": 0.65
},
"grid": {
  "order_count_cap": 16
}
```

---

## 250 USD

```yaml
risk:
  grid_equity_ratio: 0.6
  risk_per_order_pct: 0.003

grid:
  order_count_cap: 8
```

---

## 150 USD

```yaml
risk:
  grid_equity_ratio: 0.55
  risk_per_order_pct: 0.0015

grid:
  order_count_cap: 8
```

---

## Minimum (80 USD)

```json
"risk": {
  "risk_per_order_pct": 0.0015,
  "grid_equity_ratio": 0.5
},
"grid": {
  "order_count_cap": 6
}
```

---

# Optional Process Management

## start.sh

```bash
#!/bin/bash
source .venv/bin/activate
python grid_pro4_managed.py
```

Make executable:

```bash
chmod +x start.sh
```

---

## restart.sh

```bash
#!/bin/bash
pkill -f grid_pro4_managed.py
sleep 2
./start.sh
```

---

## Systemd Service Example

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

Enable service:

```bash
systemctl daemon-reload
systemctl enable gridbot
systemctl start gridbot
```

---

# Architecture Notes

* State-machine driven decision engine
* Regime-aware execution (GRID vs TREND)
* Explicit reconciliation logic
* No implicit retries
* CCXT rate-limit aware
* Deterministic behavior

---

# License

MIT License

```
MIT License

Copyright (c) 2026 Meszi84

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
