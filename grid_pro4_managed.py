# =========================================
# VERSION: grid_pro4_managed  v4.2.0      =
# STATUS: LOCKED (DO NOT EDIT ON MAINNET) =
# DATE: 2026-01-29                        =
# =========================================
#!/usr/bin/env python3
import os
import sys
import time
import json
import math
import pandas as pd
from dc.commands.risk import equity
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone, date, timedelta, time as dtime
from typing import Optional, Dict, Any, Tuple, List
import ccxt
import logging

from discord_notifier import send_discord

# =========================
# LOGGING
# =========================

LOG_LEVEL = os.getenv("GRIDBOT_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("cuncibot.grid")
logger.info("STARTED | CUNCIBOT GRID | VERSION v4.0 | grid_pro4_STABLE")

# =========================
# KONFIG
# =========================

CFG: Dict[str, Any] = {
    "exchange": {
        "testnet": False,
        "recv_window": 20000,
        "defaultType": "linear",
        "leverage": 7,
        "marginMode": "isolated",
    },
    "market": {
        "symbol": "SOL/USDT:USDT",
        "timeframe": "30m",
        "ohlcv_limit": 300,
        "poll_seconds": 10,
        "min_order_usdt": 5,
        "max_backoff_sec": 60,
    },
    "grid": {
        "lookback_bars": 48,
        "atr_period": 14,
        "atr_buffer": 0.8,
        "max_grids": 17,
        "min_step_pct": 0.0025,
        "order_count_cap": 28,
        "post_only": False,
        "refill_ratio": 0.5,
        "min_atr_pct": 0.0025,
    },
    "regime": {
        "adx_period": 14,
        "adx_on": 32.0,
        "adx_off": 24.0,
    },
    "risk": {
        "equity_usdt_assumed": 80.0,
        "use_balance_fetch": True,
        "max_net_pos_equity_ratio": 0.5,  # (most nem használod aktívan; logika marad)
        "range_break_buffer_pct": 0.006,  # (most nem használod aktívan; logika marad)
        "daily_dd_pct": 0.05,
        "grid_equity_ratio": 0.85,  # korábban hardcode volt; most CFG-ből
        "sl_streak_max": 3,
        "sl_streak_pause_min": 60,
    },
    "adx": {
        "cooldown_sec": 60,
        "period": 14,
        "threshold": 25,
        "sideways_level": 20,
        "trend_level": 25,
    },
    "trailing": {"distance_pct": 0.01},
    "logging": {
        "jsonl_path": "grid_bot_log.jsonl",
    },
    "api": {
        # Javaslat: env-ből olvasd. Példa:
        # export BYBIT_API_KEY="..."
        # export BYBIT_API_SECRET="..."
        "apiKey": os.getenv("BYBIT_API_KEY", ""),
        "secret": os.getenv("BYBIT_API_SECRET", ""),
    },
}


# =========================
# SEGÉDEK
# =========================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_today() -> date:
    return utc_now().date()


def safe_float(x, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def handle_sl_streak(pnl_pct: float, current_streak: int) -> tuple[int, float]:
    """Visszaadja az új streak számot és a pause időpontját (ha kell)."""
    if pnl_pct >= -0.001:
        return 0, 0.0  # Profitnál nullázunk

    new_streak = current_streak + 1
    pause_ts = 0.0

    if new_streak >= int(CFG["risk"].get("sl_streak_max", 3)):
        pause_min = int(CFG["risk"].get("sl_streak_pause_min", 30))
        pause_ts = time.time() + (pause_min * 60)
        logger.warning(f"SL STREAK LIMIT ELÉRVE ({new_streak}) -> Pause: {pause_min}m")

    return new_streak, pause_ts


def log_jsonl(path: str, obj: Dict[str, Any]) -> None:
    payload = dict(obj)
    payload["ts_utc"] = utc_now().isoformat()
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        # logging nem dőlhet el a bot miatt
        try:
            sys.stderr.write(f"log_jsonl error: {e}\n")
        except Exception:
            pass


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _pos_contracts(p: Optional[Dict[str, Any]]) -> float:
    """Robusztus pozíció-méret kinyerés (ccxt/Bybit mező-eltérések miatt)."""
    if not p:
        return 0.0
    for k in ("contracts", "contractSize", "size", "positionAmt"):
        v = p.get(k)
        fv = safe_float(v, default=None)
        if fv is not None and fv != 0:
            return float(fv)
    info = p.get("info") or {}
    for k in ("size", "positionAmt", "qty"):
        fv = safe_float(info.get(k), default=None)
        if fv is not None and fv != 0:
            return float(fv)
    return 0.0


def get_position(ex: ccxt.Exchange, symbol: str) -> Optional[Dict[str, Any]]:
    """Pozíció lekérdezése adott szimbólumra, tőzsde-független formátumkezeléssel."""

    def normalize(s: str) -> str:
        """Eltávolítja a szükségtelen karaktereket az összehasonlításhoz."""
        return s.replace("/", "").replace(":USDT", "").upper()

    try:
        positions = ex.fetch_positions([symbol])
        target_normalized = normalize(symbol)

        for p in positions:
            # Megpróbáljuk kinyerni a szimbólumot több helyről is
            raw_symbol = p.get("symbol") or p.get("info", {}).get("symbol")

            if not raw_symbol:
                continue

            # Összehasonlítás a tisztított formátumok alapján
            if normalize(raw_symbol) == target_normalized:
                return p

    except Exception:
        # A logger.exception automatikusan hozzáadja a hiba részleteit (traceback)
        logger.exception("Hiba a pozíció lekérdezésekor (%s)", symbol)

    return None


def place_reduce_only_conditional_market(
    ex: ccxt.Exchange,
    symbol: str,
    side: str,
    amount: float,
    trigger_price: float,
) -> Optional[Dict[str, Any]]:
    """Bybit V5 'conditional market' reduce-only order (védelmi SL jelleggel).

    Bybit V5-ben a trigger paraméterekhez a triggerPrice + triggerDirection páros használatos.
    CCXT-ben a stopPrice -> triggerPrice változás is releváns.
    """
    if amount <= 0 or trigger_price <= 0:
        logger.error(
            "Invalid SL params: amount=%s trigger_price=%s",
            amount,
            trigger_price,
        )
        return None
    try:
        # triggerDirection: 1 = rises to triggerPrice, 2 = falls to triggerPrice
        # Long SL: SELL trigger on FALL (2). Short SL: BUY trigger on RISE (1).
        td = 2 if side.lower() == "sell" else 1

        params = {
            "category": "linear",
            "reduceOnly": True,
            "orderFilter": "StopOrder",
            "triggerPrice": float(trigger_price),
            "triggerDirection": int(td),
            # Bybit gyakran igényli a triggerBy mezőt; ha nem kell, ignorálja
            "triggerBy": "MarkPrice",
        }

        return ex.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=float(amount),
            price=None,
            params=params,
        )
    except Exception:
        logger.exception(
            f"Hiba a conditional reduce-only market ordernél ({side} @ {trigger_price})"
        )
        return None


def check_existing_stop_loss(ex, symbol) -> bool:
    """
    Precízen ellenőrzi, hogy van-e aktív Stop Loss (reduce-only kondicionális order).
    """
    try:
        # A 'linear' kategória a Bybit/OKX-nél fontos, más tőzsdéknél figyelmen kívül hagyják
        orders = ex.fetch_open_orders(symbol, params={"category": "linear"}) or []

        for o in orders:
            # 1. SZŰRÉS: Csak a kondicionális (stop/trigger) orderek érdekelnek minket
            # A CCXT egységesített mezői: 'triggerPrice' vagy 'stopPrice'
            trigger_price = o.get("triggerPrice") or o.get("stopPrice")

            # 2. SZŰRÉS: Csak a Reduce-Only (vagy Close-On-Trigger) típus
            # Megnézzük a gyári mezőt, majd az 'info' alatti nyers tőzsdei választ is
            info = o.get("info", {})
            side = o.get("side") or info.get("side") or "".lower()
            is_reduce_only = (
                o.get("reduceOnly") is True
                or str(info.get("reduceOnly")).lower() == "true"
                or info.get("closeOnTrigger") is True
            )

            # Ha mindkét feltétel teljesül, megvan az SL
            if trigger_price and is_reduce_only:
                logger.info(
                    "[CHECK_SL] Talált SL | symbol=%s | side=%s | trigger=%s",
                    symbol,
                    side or "UNKNOWN",
                    trigger_price,
                )
                # Opcionális: ellenőrizheted, hogy az order státusza 'open' vagy 'untriggered'
                return True

    except Exception:
        logger.exception(f"[CHECK_SL] Hiba a lekérés során (%s)", symbol)
        # Hiba esetén False-al térünk vissza, hogy a startup_reconcile
        # inkább tegyen be egy új SL-t, mintsem védelem nélkül hagyja a pozíciót.

    return False


def calc_position_size_atr(
    equity: float,
    risk_pct: float,
    atr: float,
    price: float,
    atr_mult: float = 1.5,
    min_usdt: float = 5.0,
) -> float:
    """
    ATR-alapú pozícióméret:
    - equity: aktuális tőke
    - risk_pct: kockáztatott arány (pl. 0.01 = 1%)
    - atr: aktuális ATR
    - price: aktuális ár
    """

    if atr <= 0 or price <= 0 or equity <= 0:
        return 0.0

    risk_usdt = equity * risk_pct
    stop_dist = atr * atr_mult

    qty = risk_usdt / stop_dist
    notional = qty * price

    if notional < min_usdt:
        qty = min_usdt / price

    return qty


# =========================
# INDICATOROK (Explicit logika)
# =========================


def calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)

    # Kiszámoljuk a három lehetséges távolságot
    tr_raw = high - low
    tr_h_pc = (high - prev_close).abs()
    tr_l_pc = (low - prev_close).abs()

    # Explicit módon választjuk ki a maximumot
    tr = pd.concat([tr_raw, tr_h_pc, tr_l_pc], axis=1).max(axis=1)

    return tr.rolling(period).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # 1. DM (Directional Movement) explicit meghatározása
    diff_high = high.diff()
    diff_low = low.diff() * -1

    # Explicit feltételrendszer np.where használatával (vektorizált döntés)
    plus_dm = np.where((diff_high > diff_low) & (diff_high > 0), diff_high, 0.0)
    minus_dm = np.where((diff_low > diff_high) & (diff_low > 0), diff_low, 0.0)

    # 2. TR (True Range)
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)

    # 3. Wilder-féle simítás (RMA)
    atr_wilder = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di_smooth = pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean()
    minus_di_smooth = pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean()

    # 4. DI értékek
    plus_di = 100 * (plus_di_smooth / atr_wilder)
    minus_di = 100 * (minus_di_smooth / atr_wilder)

    # 5. DX és ADX (Explicit nullával való osztás elleni védelem)
    denom = (plus_di + minus_di).replace(0, 1)
    dx = 100 * (abs(plus_di - minus_di) / denom)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return adx


def check_liquidation_safety(
    ex,
    symbol: str,
    mark: float,
    threshold_pct: float = 0.05,
    pos: Optional[Dict] = None,
) -> bool:
    """Explicit ellenőrzés a likvidációs kockázatra."""
    try:
        p = pos if pos is not None else get_position(ex, symbol)
        if not p:
            return False

        raw_contracts = _pos_contracts(p)
        contracts = float(raw_contracts) if raw_contracts is not None else 0.0
        if contracts == 0:
            return False

        liq_price = safe_float(p.get("liquidationPrice"))
        if liq_price is None or liq_price <= 0:
            return False

        # Itt definiáljuk a 'side'-ot a függvényen belül
        side = str(p.get("side") or p.get("info", {}).get("side", "")).lower()

        if side in ("long", "buy"):
            dist = (mark - liq_price) / mark
        elif side in ("short", "sell"):
            dist = (liq_price - mark) / mark
        else:
            dist = abs(mark - liq_price) / mark

        if dist < 0:
            logger.warning(
                f"VÉSZHELYZET: Az ár átlépte a likvidációs szintet! {symbol}"
            )
            return True

        if dist < threshold_pct:
            # Itt a 'side' változót használjuk, ami már létezik
            log_jsonl(
                CFG["logging"]["jsonl_path"],
                {
                    "event": "LIQUIDATION_RISK",
                    "symbol": symbol,
                    "side": side.upper(),
                    "dist_pct": round(dist * 100, 3),
                },
            )
            send_discord(
                title="VÉSZHELYZET",
                description=(
                    f":warning: **VÉSZHELYZET** :warning:\n"
                    f"Szimbólum: `{symbol}`\n"
                    f"Oldal: `{side.upper()}`\n"
                    f"Távolság likvidációtól: `{round(dist * 100, 3)}%`"
                ),
                color=0xFF0000,
            )

            return True

        return False

    except Exception:
        logger.exception(f"Hiba a likvidáció ellenőrzése során: {symbol}")
        return False


def place_fixed_trend_stop_loss(ex, symbol: str, df):
    """
    Trend esetén ATR alapú STOP LOSS elhelyezése.
    """
    try:
        # ❗ DUPLIKÁCIÓ VÉDELEM
        if check_existing_stop_loss(ex, symbol):
            logger.info("TREND SL már létezik, kihagyva")
            return

        positions = ex.fetch_positions([symbol])

        for p in positions:
            contracts = safe_float(_pos_contracts(p), 0.0)
            if contracts == 0:
                continue

            entry = safe_float(p.get("entryPrice"))
            if not entry or entry <= 0:
                continue

            side_raw = (p.get("side") or "").lower()
            is_long = side_raw in ("long", "buy") or contracts > 0

            atr_series = calculate_atr(df, int(CFG["grid"]["atr_period"]))
            if atr_series is None or len(atr_series) < 2:
                continue

            atr = safe_float(atr_series.iloc[-2])
            if not atr or atr <= 0:
                continue

            atr_mult = 1.5

            if is_long:
                order_side = "sell"
                stop_price = entry - (atr_mult * atr)
                trigger_dir = 2
            else:
                order_side = "buy"
                stop_price = entry + (atr_mult * atr)
                trigger_dir = 1

            amount = abs(contracts)
            mark = safe_float(p.get("markPrice")) or entry

            if is_long and stop_price >= mark:
                stop_price = mark * 0.995
            if not is_long and stop_price <= mark:
                stop_price = mark * 1.005

            stop_price = float(ex.price_to_precision(symbol, stop_price))
            amount = float(ex.amount_to_precision(symbol, amount))

            logger.warning(
                f"TREND ATR SL | {symbol} | side={'LONG' if is_long else 'SHORT'} | stop={stop_price}"
            )

            ex.create_order(
                symbol=symbol,
                type="market",
                side=order_side,
                amount=amount,
                price=None,
                params={
                    "category": "linear",
                    "reduceOnly": True,
                    "orderFilter": "StopOrder",
                    "triggerPrice": stop_price,
                    "triggerDirection": trigger_dir,
                    "triggerBy": "MarkPrice",
                },
            )

            stop_pct = abs(stop_price - entry) / entry

            log_jsonl(
                CFG["logging"]["jsonl_path"],
                {
                    "event": "TREND_STOP_LOSS_PLACED",
                    "symbol": symbol,
                    "side": "LONG" if is_long else "SHORT",
                    # --- KOMPATIBILITÁS ---
                    "entry": round(float(entry), 6),
                    "stop_price": round(float(stop_price), 6),
                    "stop_pct": round(float(stop_pct * 100), 2),
                    "mark": round(float(mark), 6),
                    # --- ÚJ / ATR INFO ---
                    "atr": round(float(atr), 6),
                    "atr_mult": atr_mult,
                    "model": "ATR_TREND_STOP",
                    "ts_utc": utc_now().isoformat(),
                },
            )
            send_discord(
                title="TREND STOP LOSS ELHELYEZVE",
                description=(
                    f":warning: **TREND STOP LOSS ELHELYEZVE** :warning:\n"
                    f"Szimbólum: `{symbol}`\n"
                    f"Oldal: `{'LONG' if is_long else 'SHORT'}`\n"
                    f"Belépési ár: `{round(float(entry), 6)}`\n"
                    f"Stop ár: `{round(float(stop_price), 6)}`\n"
                    f"Stop távolság: `{round(float(stop_pct * 100), 2)}%`\n"
                    f"ATR: `{round(float(atr), 6)}` (x{atr_mult})",
                ),
                color=0xFFA500,
            )

    except Exception:
        logger.exception("Hiba trend stop loss elhelyezésekor")


# =========================
# EXCHANGE WRAPPER (Explicit döntésekkel és API jelzésekkel)
# =========================


def make_exchange(cfg: Dict[str, Any]) -> ccxt.Exchange:
    """Exchange példányosítása és inicializálása."""
    api_key = cfg["api"]["apiKey"]
    secret = cfg["api"]["secret"]

    # Explicit ellenőrzés a kulcsok meglétére
    if not api_key:
        logger.critical("Hiányzó API kulcs (apiKey)!")
        raise RuntimeError("Hiányzó API kulcs. Ellenőrizd a környezeti változókat.")

    if not secret:
        logger.critical("Hiányzó API secret!")
        raise RuntimeError("Hiányzó API secret. Ellenőrizd a környezeti változókat.")

    ex_config = {
        "apiKey": api_key,
        "secret": secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": cfg["exchange"]["defaultType"],
            "recvWindow": cfg["exchange"]["recv_window"],
        },
    }

    ex = ccxt.bybit(ex_config)

    # Explicit döntés a Sandbox/Testnet módról
    if cfg["exchange"]["testnet"] is True:
        logger.info("Sandbox mód (Testnet) aktiválva.")
        ex.set_sandbox_mode(True)
    else:
        logger.info("ÉLŐ (Mainnet) mód aktiválva.")

    # --- API HÍVÁS JELZÉSE ---
    # load_markets: Letölti a tőzsdei metaadatokat (precízió, szimbólumok)
    logger.info("Piacok betöltése az exchange-ről...")
    ex.load_markets()

    return ex


def set_leverage_margin(ex: ccxt.Exchange, symbol: str, lev: int, mode: str) -> None:
    """Tőkeáttét és margó mód explicit beállítása."""

    # --- API HÍVÁS JELZÉSE (Leverage) ---
    try:
        ex.set_leverage(lev, symbol)
        logger.info(f"Tőkeáttét beállítva: {lev}x ({symbol})")
    except Exception as e:
        logger.warning(
            f"Nem sikerült a tőkeáttétet beállítani (lehet, hogy már ennyi): {e}"
        )

    # --- API HÍVÁS JELZÉSE (Margin Mode) ---
    try:
        ex.set_margin_mode(mode, symbol)
        logger.info(f"Margó mód beállítva: {mode} ({symbol})")
    except Exception as e:
        logger.warning(
            f"Nem sikerült a margó módot beállítani (lehet, hogy már ezen van): {e}"
        )


logger = logging.getLogger(__name__)


def fetch_ohlcv_df(
    ex: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    limit: int,
    retries: int = 3,
    delay: int = 30,
    cache: Optional[dict] = None,
) -> Optional[pd.DataFrame]:
    """
    Stabil, hibatűrő OHLCV lekérő.
    - Retry + delay
    - CCXT hibák kezelése
    - Cikluson belüli cache (NEM globális!)
    """

    cache_key = (symbol, timeframe, limit)
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    for attempt in range(1, retries + 1):
        try:
            bars = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

            if not bars:
                logger.warning(
                    "[%s/%s] Üres OHLCV válasz: %s",
                    attempt,
                    retries,
                    symbol,
                )
                return None

            df = pd.DataFrame(
                bars,
                columns=["ts", "open", "high", "low", "close", "volume"],
            )

            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            df.set_index("ts", inplace=True)

            # Csak a numerikus oszlopok!
            num_cols = ["open", "high", "low", "close", "volume"]
            df[num_cols] = df[num_cols].astype(float)

            if cache is not None:
                cache[cache_key] = df

            return df

        except (ccxt.NetworkError, ccxt.RateLimitExceeded) as e:
            wait_time = delay
            logger.warning(
                "[%s/%s] OHLCV hiba (%s): %s → várakozás %ss",
                attempt,
                retries,
                symbol,
                e,
                wait_time,
            )
            if attempt < retries:
                time.sleep(wait_time)
            else:
                logger.error("OHLCV retry limit elérve (%s)", symbol)

        except Exception:
            logger.exception("Kritikus OHLCV hiba (%s)", symbol)
            break

    return None


def get_balance_usdt(ex: ccxt.Exchange) -> float:
    """Aktuális USDT egyenleg lekérdezése."""
    try:
        # --- API HÍVÁS JELZÉSE ---
        bal = ex.fetch_balance()

        # Explicit navigáció a Bybit-specifikus adatszerkezetben
        usdt_data = bal.get("USDT", {})
        total_balance = usdt_data.get("total", 0.0)

        return float(total_balance)
    except Exception:
        logger.exception(f"Hiba az egyenleg lekérdezésekor.")
        return 0.0


def cancel_all_orders(ex: ccxt.Exchange, symbol: str) -> None:
    """Összes nyitott order törlése az adott szimbólumon."""
    try:
        logger.info(f"Összes nyitott order törlése: {symbol}")
        # --- API HÍVÁS JELZÉSE ---
        ex.cancel_all_orders(symbol, params={"category": "linear"})
    except Exception:
        logger.exception(f"Hiba az orderek törlésekor ({symbol}).")


def place_limit(
    ex: ccxt.Exchange,
    symbol: str,
    side: str,
    price: float,
    amount: float,
    post_only: bool,
):
    """Limit order elhelyezése explicit paraméterezéssel."""

    params = {
        "category": "linear",
        "reduceOnly": False,
        "timeInForce": "GTC",
    }

    # Explicit döntés a Post-Only módosítóról
    if post_only is True:
        params["postOnly"] = True
        params["timeInForce"] = "PostOnly"
        logger.debug(f"Post-Only mód aktiválva az orderhez ({side})")
    else:
        logger.debug(f"Normál GTC Limit order ({side})")

    try:
        # --- API HÍVÁS JELZÉSE ---
        order = ex.create_order(
            symbol=symbol,
            type="limit",
            side=side,
            amount=float(amount),
            price=float(price),
            params=params,
        )
        return order
    except Exception:
        logger.exception(f"Hiba a limit order elhelyezésekor ({side} @ {price}).")
        return None


# =========================
# STATE (Explicit állapotkezelés)
# =========================


@dataclass
class GridState:
    active: bool = False
    bottom: Optional[float] = None
    top: Optional[float] = None
    step: Optional[float] = None
    grids: int = 0
    started_ts: float = 0.0

    def reset(self) -> None:
        """Explicit állapot törlés."""
        logger.info("GridState alaphelyzetbe állítása.")
        self.active = False
        self.bottom = None
        self.top = None
        self.step = None
        self.grids = 0

    def is_valid(self) -> bool:
        """
        Explicit validációs folyamat.
        Minden hibás állapotnál külön döntés és logolás születik.
        """
        if self.active is False:
            logger.debug("GridState érvénytelen: az állapot nem aktív.")
            return False

        # Mezõk meglétének ellenőrzése
        if self.bottom is None:
            logger.warning("GridState érvénytelen: hiányzó 'bottom' érték.")
            return False
        if self.top is None:
            logger.warning("GridState érvénytelen: hiányzó 'top' érték.")
            return False
        if self.step is None:
            logger.warning("GridState érvénytelen: hiányzó 'step' érték.")
            return False

        # Logikai összefüggések ellenőrzése
        if self.top <= self.bottom:
            logger.error(
                f"GridState hiba: 'top' ({self.top}) nem nagyobb, mint 'bottom' ({self.bottom})."
            )
            return False

        if self.step <= 0:
            logger.error(f"GridState hiba: 'step' ({self.step}) nem pozitív szám.")
            return False

        if self.grids <= 0:
            logger.error(f"GridState hiba: 'grids' száma ({self.grids}) nem pozitív.")
            return False

        return True


@dataclass
class DayState:
    day_utc: date
    start_equity: float
    min_equity: float = 0.0

    def __post_init__(self) -> None:
        self.min_equity = self.start_equity

    def reset_if_new_day(self, current_equity: float) -> bool:
        """
        Explicit döntés a napváltásról.
        """
        today = utc_today()

        if self.day_utc != today:
            logger.info(f"Új nap észlelve: {today}. Állapot frissítése.")
            self.day_utc = today
            self.start_equity = current_equity
            self.min_equity = current_equity
            return True
        else:
            # Maradunk a jelenlegi napnál
            return False

    def update(self, current_equity: float) -> None:
        """
        Explicit döntés a minimum equity frissítéséről.
        """
        if current_equity < self.min_equity:
            logger.info(
                f"Új napi minimum equity: {current_equity} (előző: {self.min_equity})"
            )
            self.min_equity = current_equity
        else:
            # Nem történt negatív irányú elmozdulás a minimumhoz képest
            pass

    @property
    def drawdown(self) -> float:
        """Kiszámított napi drawdown érték."""
        s_equity = float(self.start_equity)
        m_equity = float(self.min_equity)

        # Explicit különbségképzés
        res = s_equity - m_equity
        return res


# =========================
# GRID LOGIKA (Explicit döntésekkel)
# =========================


def calculate_smart_grid_count(equity: float, cfg: Dict[str, Any]) -> int:
    """Kiszámítja, hány grid fér bele a tőkébe. Explicit korlátokkal."""
    ratio = float(cfg["risk"].get("grid_equity_ratio", 0.7))
    usable_equity = float(equity) * ratio
    min_order_cost = float(cfg["market"]["min_order_usdt"])

    # Explicit döntés a tőke alapú korlátról
    if min_order_cost > 0:
        max_by_equity = int(usable_equity / min_order_cost)
    else:
        max_by_equity = 0

    # Explicit összehasonlítás a konfigurációs korlátokkal
    config_max = int(cfg["grid"]["max_grids"])
    order_cap = int(cfg["grid"].get("order_count_cap", 999))

    # Kiválasztjuk a legkisebb korlátot (Explicit módon)
    result = config_max
    if order_cap < result:
        result = order_cap
    if max_by_equity < result:
        result = max_by_equity

    # Minimum 1 grid biztosítása
    if result < 1:
        return 1
    return result


def adx_allows_grid(df: pd.DataFrame, cfg: Dict[str, Any], last_state: bool) -> bool:
    """Regime detektálás ADX alapján. Explicit állapotátmenetek."""
    r = cfg["regime"]
    adx_series = calculate_adx(df, int(r["adx_period"]))

    # Utolsó lezárt gyertya értékének kinyerése
    val = safe_float(adx_series.iloc[-2], default=None)

    # Explicit döntés: Ha nincs adat, megtartjuk az előző állapotot
    if val is None:
        logger.debug("ADX érték nem elérhető, állapot változatlan.")
        return last_state

    if not math.isfinite(val):
        logger.warning("ADX érték nem véges (NaN/Inf), állapot változatlan.")
        return last_state

    # Explicit küszöbvizsgálat
    if val > float(r["adx_off"]):
        # Túl erős trend, grid kikapcsolása
        return False
    elif val < float(r["adx_on"]):
        # Megfelelő oldalazás, grid engedélyezése
        return True
    else:
        # "Szürke zóna", nincs váltás
        return last_state


def compute_range_and_step(
    df: pd.DataFrame, cfg: Dict[str, Any], grids: int
) -> Optional[Tuple[float, float, float]]:
    """Grid tartomány és lépésköz számítása ATR alapján."""
    g = cfg["grid"]
    working_df = df.copy()

    working_df["atr"] = calculate_atr(working_df, int(g["atr_period"]))
    atr_val = safe_float(working_df["atr"].iloc[-2], default=None)

    # Explicit validáció az ATR-re
    if atr_val is None or atr_val <= 0 or not math.isfinite(atr_val):
        logger.error("ATR számítás sikertelen vagy érvénytelen érték.")
        return None

    lookback = int(g["lookback_bars"])
    window = working_df.iloc[-(lookback + 2) : -2]

    if len(window) < 5:
        logger.warning("Nincs elég adat a window-ban a tartomány számításhoz.")
        return None

    low_val = safe_float(window["low"].min())
    high_val = safe_float(window["high"].max())

    if low_val is None or high_val is None:
        return None

    # Buffer hozzáadása
    buffer_val = float(g["atr_buffer"]) * atr_val
    bottom = float(low_val - buffer_val)
    top = float(high_val + buffer_val)

    if not math.isfinite(bottom) or not math.isfinite(top) or top <= bottom:
        logger.error(f"Érvénytelen tartomány: {bottom} - {top}")
        return None

    mid_price = (top + bottom) / 2.0

    # Explicit lépésköz számítás
    denom = grids - 1
    if denom < 1:
        denom = 1

    raw_step = (top - bottom) / denom
    min_step = mid_price * float(g["min_step_pct"])

    # Explicit kiválasztás a lépésközre
    if raw_step > min_step:
        step = raw_step
    else:
        step = min_step

    # Stabil rebuild: a top értéket a tényleges lépésközhöz igazítjuk
    adjusted_top = bottom + step * (grids - 1)

    return bottom, adjusted_top, step


def _open_order_price_set(ex: ccxt.Exchange, symbol: str) -> Tuple[set, set]:
    """Lekéri a nyitott ordereket és árak szerint szortírozza őket."""
    buy_prices = set()
    sell_prices = set()

    try:
        # --- API HÍVÁS JELZÉSE ---
        # fetch_open_orders: Aktuális függőben lévő ajánlatok lekérése
        oo = ex.fetch_open_orders(symbol, params={"category": "linear"})

        if not oo:
            return buy_prices, sell_prices

        for o in oo:
            side = (o.get("side") or "").lower()
            price_raw = o.get("price")

            if price_raw is None:
                continue

            try:
                p = float(price_raw)
            except (ValueError, TypeError):
                continue

            if side == "buy":
                buy_prices.add(p)
            elif side == "sell":
                sell_prices.add(p)

    except Exception:
        logger.exception(f"Hiba az open order árak lekérésekor ({symbol}).")

    return buy_prices, sell_prices


def ensure_grid_orders(
    ex: ccxt.Exchange,
    symbol: str,
    grid: GridState,
    mark: float,
    equity: float,
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    max_new_orders: Optional[int] = None,
) -> int:
    """Hiányzó grid szintek pótlása.

    Fő célok:
    - ATR csak egyszer számolva / ciklus
    - Open order duplikáció ellenőrzés ár-toleranciával
    - Limitált kihelyezés (max_new_orders) az API terhelés csökkentésére
    """

    if not grid.is_valid():
        return 0

    if mark <= 0 or not math.isfinite(mark):
        return 0

    max_orders_total = int(cfg["grid"].get("order_count_cap", 50))
    min_notional = float(cfg["market"]["min_order_usdt"])
    leverage = float(cfg["exchange"].get("leverage", 1))
    equity_ratio = float(cfg["risk"].get("grid_equity_ratio", 0.8))
    usable_capital = float(equity) * leverage * equity_ratio

    # 1) ATR egyszer (risk sizing-hoz)
    atr_series = calculate_atr(df, int(cfg["grid"]["atr_period"]))
    if atr_series is None or len(atr_series) < 2:
        logger.warning("ATR nem elérhető → grid skip")
        return 0
    atr = safe_float(atr_series.iloc[-2])
    if atr is None or atr <= 0 or not math.isfinite(atr):
        logger.warning("ATR érvénytelen → grid skip")
        return 0

    # 2) Ár-létra generálás (végtelen ciklus elleni fék)
    prices: List[float] = []
    current_p = float(grid.bottom)
    limit_p = float(grid.top)
    step_val = float(grid.step)

    safety_count = 0
    while current_p <= limit_p + 1e-9 and safety_count < (max_orders_total + 50):
        try:
            formatted_p = float(ex.price_to_precision(symbol, current_p))
        except Exception:
            formatted_p = float(current_p)
        prices.append(formatted_p)
        current_p += step_val
        safety_count += 1

    unique_prices = sorted(set(prices))
    if not unique_prices:
        return 0

    # 3) Buy/Sell zónák (mark körül hagyjunk üres sávot)
    buys = [px for px in unique_prices if px < mark * 0.999]
    sells = [px for px in unique_prices if px > mark * 1.001]

    # Üres oldal esetén is legyen legalább 1 szint (szél)
    if not buys:
        buys = [unique_prices[0]]
    if not sells:
        sells = [unique_prices[-1]]

    planned = buys + sells
    if len(planned) > max_orders_total:
        planned = planned[:max_orders_total]

    if not planned:
        return 0

    # 4) Open order árak
    buy_open, sell_open = _open_order_price_set(ex, symbol)

    # 5) Per-order risk alapú méretezés (egységes)
    risk_pct = float(cfg["risk"].get("risk_per_order_pct", 0.003))
    risk_usdt = float(equity) * risk_pct

    market_info = ex.market(symbol)
    min_amt = safe_float(((market_info.get("limits") or {}).get("amount") or {}).get("min"), 0.0) or 0.0

    def _is_dupe(price: float, open_set: set) -> bool:
        # relatív tolerancia (Bybit precízió miatt)
        for op in open_set:
            if abs(price - op) / max(op, 1e-9) < 1e-6:
                return True
        return False

    placed = 0
    budget_left = max_new_orders if (max_new_orders is not None and max_new_orders > 0) else None

    # 6) Kihelyezés: előbb BUY, majd SELL (szándékosan determinisztikus)
    for side, plist, open_set in (("buy", buys, buy_open), ("sell", sells, sell_open)):
        for price in plist:
            if budget_left is not None and placed >= budget_left:
                return placed

            if _is_dupe(price, open_set):
                continue

            try:
                # Risk sizing (egyszerű, konzervatív): qty = risk_usdt / (atr * price)
                qty = risk_usdt / (atr * price)
                amt = float(ex.amount_to_precision(symbol, qty))
                if min_amt and amt < float(min_amt):
                    continue

                notional = amt * price
                if notional < min_notional:
                    continue

                logger.info(
                    "Grid order: %s %s %s @ %.6f | atr=%.6f | notional=%.2f",
                    side.upper(), amt, symbol, price, atr, notional
                )

                place_limit(ex, symbol, side, price, amt, bool(cfg["grid"]["post_only"]))

                placed += 1
                open_set.add(price)

            except Exception:
                logger.exception("Hiba a grid order elhelyezésekor (%s @ %s).", side, price)

    return placed

# =========================
# FŐ LOOP ÉS KERESKEDÉSI FUNKCIÓK
# =========================


def partial_derisk(
    ex: ccxt.Exchange,
    symbol: str,
    mark: float,
    pnl_pct_trigger: float = 0.03,
    close_ratio: float = 0.5,
    pos: Optional[Dict] = None,
) -> bool:
    """Részleges profitkivétel explicit döntési logikával."""
    # --- API HÍVÁS JELZÉSE (Ha pos nincs megadva) ---
    if pos is None:
        p = get_position(ex, symbol)
    else:
        p = pos

    if not p:
        return False

    try:
        contracts = float(_pos_contracts(p))
        entry = float(p.get("entryPrice", 0))
    except (TypeError, ValueError):
        return False

    if contracts == 0:
        return False

    # Irány meghatározása explicit módon
    if contracts > 0:
        side_val = 1
        order_side = "sell"
    else:
        side_val = -1
        order_side = "buy"

    current_pnl = (mark - entry) / entry * side_val

    # Explicit döntés a profitkivételről
    if current_pnl >= pnl_pct_trigger:
        amount_to_close = abs(contracts) * close_ratio

        logger.info(
            f"💰 Részleges profit realizálás (PnL: {current_pnl:.2%}) | Mennyiség: {amount_to_close}"
        )

        # --- API HÍVÁS JELZÉSE ---
        ex.create_market_order(
            symbol=symbol,
            side=order_side,
            amount=amount_to_close,
            params={"reduceOnly": True},
        )
        return True

    return False


# def get_position(ex: ccxt.Exchange, symbol: str) -> Optional[Dict]:
#    """Lekéri az aktuális nyitott pozíciót. API hívást tartalmaz."""
#    try:
# --- API HÍVÁS JELZÉSE ---
#        positions = ex.fetch_positions([symbol])

#        for p in positions:
#            if p.get("symbol") == symbol:
#                return p
#        return None
#    except Exception as e:
#        logger.error(f"Hiba a pozíció lekérésekor ({symbol}): {e}")
#       return None


def close_position_market(ex: ccxt.Exchange, symbol: str, pos: Optional[Dict] = None):
    """Pozíció azonnali zárása piaci áron. Explicit irány meghatározás."""
    try:
        p = pos if pos is not None else get_position(ex, symbol)
        if not p:
            return None

        contracts = float(_pos_contracts(p))
        if abs(contracts) < 0.0001:
            return None
        # Explicit irányválasztás
        side = "sell" if contracts > 0 else "buy"
        amount = abs(contracts)
        # --- API HÍVÁS JELZÉSE ---
        return ex.create_market_order(
            symbol=symbol,
            side=side,
            amount=amount,
            params={"reduceOnly": True, "category": "linear"},
        )

    except Exception as e:
        if "current position is zero" in str(e):
            logger.info(f"A pozíció már korábban lezárult: {symbol}")
            return None
        logger.error(f"Hiba a pozíció piaci zárásakor: {e}")
        return None


def sanity_check_cfg(config_to_check: dict) -> None:
    """
    Startup sanity check:
    - hiányzó CFG kulcsok
    - rossz típusok
    - kritikus trading paraméterek
    """

    required = {
        # ADX
        ("adx", "period"): int,
        ("adx", "sideways_level"): (int, float),
        ("adx", "trend_level"): (int, float),
        ("adx", "cooldown_sec"): int,
        # GRID
        ("grid", "atr_period"): int,
        ("grid", "min_step_pct"): (int, float),
        ("grid", "atr_buffer"): (int, float),
        # RISK
        ("risk", "daily_dd_pct"): (int, float),
        ("risk", "grid_equity_ratio"): (int, float),
        # MARKET
        ("market", "poll_seconds"): (int, float),
    }

    for path, expected_type in required.items():
        cur = CFG
        for key in path:
            if key not in cur:
                raise RuntimeError(
                    f"CFG sanity check failed: missing key {'.'.join(path)}"
                )
            cur = cur[key]

        if not isinstance(cur, expected_type):
            raise RuntimeError(
                f"CFG sanity check failed: {'.'.join(path)} "
                f"has invalid type {type(cur).__name__}, expected {expected_type}"
            )

    # extra log – induláskor látod, hogy OK
    logger.info("CFG sanity check OK")


def run() -> None:
    """Main loop – döntés-orientált engine, alacsonyabb API terheléssel.

    Elvek:
    - Startup reconcile: először felméri az állapotot (pozíció / SL / open orders / regime), utána dönt.
    - Throttle: ticker/position/open_orders nem minden körben.
    - Trend mód: aktív pozíciót véd (veszteség -> SL, nyereség -> trailing), nem zár azonnal.
    - Grid mód: nem „rántja fel” az összes ordert egyszerre; limitált refill.
    """
    sanity_check_cfg(CFG)
    log_jsonl(CFG["logging"]["jsonl_path"], {"event": "CFG_OK"})

    ex = make_exchange(CFG)
    symbol = CFG["market"]["symbol"]
    tf = CFG["market"]["timeframe"]
    poll_sec = float(CFG["market"]["poll_seconds"])
    max_backoff = int(CFG["market"].get("max_backoff_sec", 60))

    set_leverage_margin(
        ex, symbol, int(CFG["exchange"]["leverage"]), str(CFG["exchange"]["marginMode"])
    )

    # --- State ---
    grid = GridState()
    equity = float(get_balance_usdt(ex) or 0.0)
    if equity <= 10:
        equity = float(CFG["risk"]["equity_usdt_assumed"])
    initial_daily_equity = float(equity)
    day = DayState(utc_today(), float(equity))

    # SL streak pause
    sl_streak = 0
    sl_pause_until_ts = 0.0

    # Regime state (str: "GRID" / "TREND")
    last_mode = "GRID"

    # Trend trailing state (csak trend módban releváns)
    trend_trailing_active = False
    trend_peak = None  # float

    # API throttles
    open_orders_every_sec = float(CFG["grid"].get("open_orders_every_sec", max(10.0, poll_sec)))
    pos_every_sec = float(CFG["market"].get("position_every_sec", max(5.0, poll_sec)))
    ticker_every_sec = float(CFG["market"].get("ticker_every_sec", poll_sec))

    last_open_orders_ts = 0.0
    last_pos_ts = 0.0
    last_ticker_ts = 0.0

    cached_open_orders: List[Dict[str, Any]] = []
    cached_pos: Optional[Dict[str, Any]] = None
    cached_mark: Optional[float] = None

    send_discord(
        title="▶️ Bot elindult",
        description=f"Symbol: {symbol}\nEquity: {equity:.2f} USDT",
        color=0x3498DB,
    )
    log_jsonl(
        CFG["logging"]["jsonl_path"],
        {
            "event": "START_SESSION",
            "symbol": symbol,
            "tf": tf,
            "start_equity": float(equity),
        },
    )

    def _extract_entry_price(pos: Optional[Dict[str, Any]]) -> Optional[float]:
        if not pos:
            return None
        for k in ("entryPrice", "avgPrice"):
            v = safe_float(pos.get(k))
            if v and v > 0:
                return v
        info = pos.get("info") or {}
        for k in ("positionAvgPrice", "avgEntryPrice", "entryPrice", "avgPrice"):
            v = safe_float(info.get(k))
            if v and v > 0:
                return v
        return None

    def _mode_from_adx(df: pd.DataFrame, prev: str) -> str:
        """ADX hysteresis: GRID <-> TREND"""
        adx = calculate_adx(df, int(CFG["adx"].get("period", 14)))
        if adx is None or len(adx) < 3:
            return prev
        v = safe_float(adx.iloc[-2])
        if v is None or not math.isfinite(v):
            return prev

        sideways = float(CFG["adx"].get("sideways_level", 20))
        trend = float(CFG["adx"].get("trend_level", 25))

        if prev == "GRID":
            return "TREND" if v >= trend else "GRID"
        else:
            return "GRID" if v <= sideways else "TREND"

    def _cancel_grid_orders(open_orders: List[Dict[str, Any]]) -> int:
        """Csak a nem-reduceOnly limit ordereket törli (grid)."""
        cancelled = 0
        for o in open_orders or []:
            try:
                info = o.get("info") or {}
                is_reduce = (
                    o.get("reduceOnly") is True
                    or str(info.get("reduceOnly")).lower() == "true"
                    or info.get("closeOnTrigger") is True
                )
                # stop/trigger order (SL) -> NE töröljük
                trigger_price = o.get("triggerPrice") or o.get("stopPrice") or info.get("triggerPrice")
                if is_reduce and trigger_price:
                    continue

                typ = (o.get("type") or info.get("orderType") or "").lower()
                if typ and typ != "limit":
                    continue

                oid = o.get("id") or info.get("orderId")
                if not oid:
                    continue

                ex.cancel_order(oid, symbol, params={"category": "linear"})
                cancelled += 1
            except Exception:
                logger.exception("Nem sikerült grid order törlés (id=%s)", o.get("id"))
        return cancelled

    def _fetch_mark(force: bool = False) -> Optional[float]:
        nonlocal last_ticker_ts, cached_mark
        now = time.time()
        if (not force) and cached_mark is not None and (now - last_ticker_ts) < ticker_every_sec:
            return cached_mark
        try:
            t = ex.fetch_ticker(symbol) or {}
            m = safe_float(t.get("mark", t.get("last")))
            if m and m > 0 and math.isfinite(m):
                cached_mark = float(m)
                last_ticker_ts = now
                return cached_mark
        except Exception:
            logger.debug("Ticker fetch failed", exc_info=True)
        return cached_mark

    def _fetch_pos(force: bool = False) -> Optional[Dict[str, Any]]:
        nonlocal last_pos_ts, cached_pos
        now = time.time()
        if (not force) and (now - last_pos_ts) < pos_every_sec:
            return cached_pos
        cached_pos = get_position(ex, symbol)
        last_pos_ts = now
        return cached_pos

    def _fetch_open_orders(force: bool = False) -> List[Dict[str, Any]]:
        nonlocal last_open_orders_ts, cached_open_orders
        now = time.time()
        if (not force) and (now - last_open_orders_ts) < open_orders_every_sec:
            return cached_open_orders
        try:
            cached_open_orders = ex.fetch_open_orders(symbol, params={"category": "linear"}) or []
        except Exception:
            logger.exception("fetch_open_orders failed (%s)", symbol)
            cached_open_orders = []
        last_open_orders_ts = now
        return cached_open_orders

    def _startup_reconcile(df: pd.DataFrame, mark: float) -> None:
        """Induláskor: felmér + helyreállít minimálisan."""
        nonlocal last_mode, trend_trailing_active, trend_peak

        pos = _fetch_pos(force=True)
        open_orders = _fetch_open_orders(force=True)

        has_pos = bool(pos and abs(_pos_contracts(pos)) > 1e-9)
        has_sl = check_existing_stop_loss(ex, symbol) if has_pos else False

        last_mode = _mode_from_adx(df, last_mode)

        log_jsonl(
            CFG["logging"]["jsonl_path"],
            {
                "event": "STARTUP_RECONCILE",
                "symbol": symbol,
                "mode": last_mode,
                "has_pos": has_pos,
                "has_sl": has_sl,
                "open_orders": len(open_orders),
                "mark": float(mark),
            },
        )

        send_discord(
            title="🧭 Startup reconcile",
            description=(
                f"Mode: {last_mode}\n"
                f"Pos: {'YES' if has_pos else 'NO'} | SL: {'YES' if has_sl else 'NO'}\n"
                f"Open orders: {len(open_orders)}\n"
                f"Mark: {mark:.4f}"
            ),
            color=0x95A5A6,
        )

        # Ha van pozíció és trend mód, inicializáljuk a trailing állapotot
        if has_pos and last_mode == "TREND":
            trend_trailing_active = False
            trend_peak = float(mark)

        # Ha nincs pozíció és van rengeteg open order -> feltételezhető grid maradék
        # Nem törlünk automatikusan mindent; majd módváltáskor döntünk.

    # --- FIRST MARKET SNAPSHOT (mielőtt bármit csinálna) ---
    backoff = poll_sec
    while True:
        df0 = fetch_ohlcv_df(ex, symbol, tf, int(CFG["market"]["ohlcv_limit"]), cache={})
        if df0 is None or len(df0) < 30:
            logger.warning("Startup: OHLCV insufficient, retry...")
            time.sleep(min(backoff, max_backoff))
            backoff = min(backoff * 1.5, max_backoff)
            continue
        mk0 = _fetch_mark(force=True)
        if mk0 is None:
            mk0 = safe_float(df0.iloc[-2]["close"]) or safe_float(df0.iloc[-1]["close"])
        if mk0 is None:
            time.sleep(min(backoff, max_backoff))
            backoff = min(backoff * 1.5, max_backoff)
            continue
        _startup_reconcile(df0, float(mk0))
        break

    # -----------------
    # MAIN LOOP
    # -----------------
    err_count = 0
    while True:
        try:
            now_ts = time.time()

            # SL pause
            if sl_pause_until_ts > 0 and now_ts < sl_pause_until_ts:
                time.sleep(min(poll_sec, 10.0))
                continue

            # Equity / day
            eq_now = get_balance_usdt(ex)
            if eq_now and eq_now > 10:
                equity = float(eq_now)
            day.update(float(equity))
            if day.reset_if_new_day(float(equity)):
                initial_daily_equity = float(equity)

            # Daily DD stop
            dd_limit_usdt = initial_daily_equity * float(CFG["risk"]["daily_dd_pct"])
            if day.drawdown >= dd_limit_usdt:
                logger.critical("DAILY_DD_STOP | dd=%.2f limit=%.2f", day.drawdown, dd_limit_usdt)
                log_jsonl(
                    CFG["logging"]["jsonl_path"],
                    {"event": "DAILY_DD_STOP", "equity": float(equity), "drawdown": float(day.drawdown)},
                )
                send_discord(
                    title="🚨 DAILY DD STOP",
                    description=f"{symbol}\nDrawdown elérve\nEquity: {equity:.2f} USDT",
                    color=0xFF0000,
                )
                # vész: mindent törlünk, pozíciót zárjuk (reduceOnly market)
                cancel_all_orders(ex, symbol)
                pos = _fetch_pos(force=True)
                if pos and abs(_pos_contracts(pos)) > 1e-9:
                    entry = _extract_entry_price(pos) or safe_float(pos.get("entryPrice")) or 0.0
                    mark = _fetch_mark(force=True) or entry
                    qty = float(_pos_contracts(pos))
                    pnl_pct = 0.0
                    if entry and entry > 0:
                        pnl_pct = ((mark - entry) / entry) if qty > 0 else ((entry - mark) / entry)
                    close_position_market(ex, symbol, pos)
                    sl_streak, sl_pause_until_ts = handle_sl_streak(float(pnl_pct), sl_streak)
                grid.reset()
                time.sleep(max(30.0, poll_sec))
                continue

            # Market data (OHLCV cache per cycle)
            df = fetch_ohlcv_df(ex, symbol, tf, int(CFG["market"]["ohlcv_limit"]), cache={})
            if df is None or len(df) < 30:
                time.sleep(poll_sec)
                continue

            mark = _fetch_mark(force=False)
            if mark is None:
                mark = safe_float(df.iloc[-2]["close"]) or safe_float(df.iloc[-1]["close"])
            if mark is None or mark <= 0:
                time.sleep(poll_sec)
                continue
            mark = float(mark)

            # Position & open orders (throttled)
            pos = _fetch_pos(force=False)
            open_orders = _fetch_open_orders(force=False)

            # Liquidation safety (csak ha van pozíció)
            if pos and abs(_pos_contracts(pos)) > 1e-9:
                if check_liquidation_safety(ex, symbol, mark, threshold_pct=0.05, pos=pos):
                    time.sleep(max(60.0, poll_sec))
                    continue

            # Regime / mode (csak döntés – nincs side effect)
            mode = _mode_from_adx(df, last_mode)
            if mode != last_mode:
                logger.warning("REGIME CHANGE | %s -> %s", last_mode, mode)
                log_jsonl(CFG["logging"]["jsonl_path"], {"event": "REGIME_CHANGE", "from": last_mode, "to": mode, "mark": mark})
                send_discord(
                    title="🔁 Regime váltás",
                    description=f"{last_mode} → {mode}\nMark: {mark:.4f}",
                    color=0xF1C40F,
                )

                # GRID -> TREND: grid orderek törlése, SL marad
                if last_mode == "GRID" and mode == "TREND":
                    cancelled = _cancel_grid_orders(open_orders)
                    if cancelled:
                        send_discord(
                            title="🧹 Grid orderek törölve",
                            description=f"Trend mód miatt törölve: {cancelled} db",
                            color=0x95A5A6,
                        )

                    # trend ctx init
                    trend_trailing_active = False
                    trend_peak = float(mark)

                # TREND -> GRID: trailing ctx reset
                if last_mode == "TREND" and mode == "GRID":
                    trend_trailing_active = False
                    trend_peak = None

                last_mode = mode

            has_pos = bool(pos and abs(_pos_contracts(pos)) > 1e-9)

            # -----------------
            # TREND MODE
            # -----------------
            if mode == "TREND":
                if not has_pos:
                    # trendben nincs pozíció: csak várunk, nem építünk gridet
                    time.sleep(poll_sec)
                    continue

                qty = float(_pos_contracts(pos))
                is_long = qty > 0
                entry = _extract_entry_price(pos)
                if entry is None or entry <= 0:
                    time.sleep(poll_sec)
                    continue

                pnl_pct = ((mark - entry) / entry) if is_long else ((entry - mark) / entry)

                # 1) Veszteség: legyen SL (csak ha nincs)
                if pnl_pct <= 0:
                    if not check_existing_stop_loss(ex, symbol):
                        atr_series = calculate_atr(df, int(CFG["grid"]["atr_period"]))
                        atr = safe_float(atr_series.iloc[-2]) if (atr_series is not None and len(atr_series) >= 2) else None

                        if atr and atr > 0:
                            raw_sl = (entry - (1.2 * atr)) if is_long else (entry + (1.2 * atr))
                        else:
                            raw_sl = entry * (0.97 if is_long else 1.03)

                        if is_long and raw_sl >= mark:
                            raw_sl = mark * 0.995
                        if (not is_long) and raw_sl <= mark:
                            raw_sl = mark * 1.005

                        sl_price = float(ex.price_to_precision(symbol, raw_sl))
                        sl_amount = float(ex.amount_to_precision(symbol, abs(qty)))
                        sl_side = "sell" if is_long else "buy"

                        place_reduce_only_conditional_market(ex, symbol, sl_side, sl_amount, sl_price)

                        log_jsonl(
                            CFG["logging"]["jsonl_path"],
                            {"event": "TREND_SL_PLACED", "symbol": symbol, "side": "LONG" if is_long else "SHORT", "entry": entry, "mark": mark, "sl": sl_price, "pnl_pct": pnl_pct * 100},
                        )
                        send_discord(
                            title="🛑 Trend SL kihelyezve",
                            description=f"{symbol}\nSide: {'LONG' if is_long else 'SHORT'}\nSL: {sl_price:.4f}",
                            color=0xE74C3C,
                        )

                    # veszteségben trailing OFF
                    trend_trailing_active = False
                    trend_peak = float(mark)
                    time.sleep(poll_sec)
                    continue

                # 2) Nyerő: trailing (ATR %-ból)
                atr_series = calculate_atr(df, int(CFG["grid"]["atr_period"]))
                if atr_series is None or len(atr_series) < 2:
                    time.sleep(poll_sec)
                    continue
                atr = safe_float(atr_series.iloc[-2])
                if atr is None or atr <= 0:
                    time.sleep(poll_sec)
                    continue

                trail_dist_pct = 1.8 * (atr / mark)
                trail_dist_pct = max(0.002, min(0.10, trail_dist_pct))

                if not trend_trailing_active or trend_peak is None:
                    trend_trailing_active = True
                    trend_peak = float(mark)
                    log_jsonl(CFG["logging"]["jsonl_path"], {"event": "TREND_TRAILING_START", "symbol": symbol, "distance_pct": trail_dist_pct})
                    send_discord(
                        title="🎯 Trailing stop aktív",
                        description=f"{symbol}\nDist: {trail_dist_pct:.2%}",
                        color=0x9B59B6,
                    )

                if is_long:
                    if mark > float(trend_peak):
                        trend_peak = float(mark)
                    stop_price = float(trend_peak) * (1.0 - trail_dist_pct)
                    triggered = mark <= stop_price
                else:
                    if mark < float(trend_peak):
                        trend_peak = float(mark)
                    stop_price = float(trend_peak) * (1.0 + trail_dist_pct)
                    triggered = mark >= stop_price

                if triggered:
                    logger.warning("TREND TRAILING TRIGGER | mark=%.6f stop=%.6f", mark, stop_price)
                    close_position_market(ex, symbol, pos)
                    sl_streak, sl_pause_until_ts = handle_sl_streak(float(pnl_pct), sl_streak)
                    trend_trailing_active = False
                    trend_peak = None

                    log_jsonl(CFG["logging"]["jsonl_path"], {"event": "TREND_TRAILING_EXIT", "symbol": symbol, "mark": mark, "stop": stop_price, "pnl_pct": pnl_pct * 100})
                    send_discord(
                        title="🏁 Trailing exit",
                        description=f"{symbol}\nExit @ {mark:.4f}\nPnL: {pnl_pct*100:.2f}%",
                        color=0x2ECC71,
                    )

                time.sleep(poll_sec)
                continue

            # -----------------
            # GRID MODE
            # -----------------
            # GRID-ben: ha van pozíció, nem trend-menedzsment, csak opcionális de-risk
            if has_pos:
                try:
                    partial_derisk(ex, symbol, mark, pnl_pct_trigger=float(CFG["risk"].get("partial_take_profit_pct", 0.03)))
                except Exception:
                    logger.debug("partial_derisk error", exc_info=True)

            # Grid paraméterek
            equity_health = float(equity) / max(initial_daily_equity, 1e-9)
            base_ratio = float(CFG["risk"]["grid_equity_ratio"])
            current_grid_ratio = (max(0.5, base_ratio * equity_health) if equity_health < 0.9 else base_ratio)

            adj_equity = float(equity) * (current_grid_ratio / max(base_ratio, 1e-9))
            grids_count = calculate_smart_grid_count(adj_equity, CFG)

            # Range+step
            rs = compute_range_and_step(df, CFG, grids_count)
            if rs is None:
                time.sleep(poll_sec)
                continue
            bottom, top, step = rs

            # GridState init/refresh (nem rebuild minden körben)
            if (not grid.active) or (not grid.is_valid()):
                grid.active = True
                grid.bottom = float(bottom)
                grid.top = float(top)
                grid.step = float(step)
                grid.grids = int(grids_count)
                grid.started_ts = time.time()
                log_jsonl(CFG["logging"]["jsonl_path"], {"event": "GRID_INIT", "symbol": symbol, "bottom": bottom, "top": top, "step": step, "grids": grids_count, "mark": mark})
                send_discord(
                    title="🧱 Grid inicializálva",
                    description=f"{symbol}\nRange: {bottom:.4f} - {top:.4f}\nStep: {step:.6f}\nGrids: {grids_count}",
                    color=0x34495E,
                )

            # Refill / ensure (limitált kihelyezés)
            max_place_per_cycle = int(CFG["grid"].get("max_place_per_cycle", 6))
            placed = ensure_grid_orders(ex, symbol, grid, mark, adj_equity, df, CFG, max_new_orders=max_place_per_cycle)
            if placed:
                log_jsonl(CFG["logging"]["jsonl_path"], {"event": "GRID_ORDERS_PLACED", "symbol": symbol, "placed": placed, "mark": mark})
            time.sleep(poll_sec)
            err_count = 0

        except KeyboardInterrupt:
            send_discord(
                title="🛑 Bot leállítva",
                description=f"{symbol}\nFelhasználói leállítás",
                color=0x7F8C8D,
            )
            try:
                cancel_all_orders(ex, symbol)
            except Exception:
                pass
            sys.exit(0)

        except Exception as e:
            err_count += 1
            logger.exception("Kritikus hiba a főciklusban")

            if err_count > 10:
                send_discord(
                    title="💀 KRITIKUS VÉSZLEÁLLÁS",
                    description=f"{symbol}: túl sok hiba ({err_count}).\n{str(e)}",
                    color=0x000000,
                )
                sys.exit(1)

            time.sleep(min(poll_sec * err_count, float(max_backoff)))


if __name__ == "__main__":
    run()
