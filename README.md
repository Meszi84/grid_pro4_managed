# Grid Pro 4 – Managed Trading Engine (Bybit / CCXT)

⚠️ **Advanced / Experimental trading engine**  
This project is **not a beginner bot**. It is a **state-driven trading engine** designed to run safely on live exchanges with **minimal API usage**, **deterministic decisions**, and **clear separation between GRID and TREND regimes**.

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

Python **3.8+** required.

```bash
pip install ccxt pandas numpy
