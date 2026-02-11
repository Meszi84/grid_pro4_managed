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
