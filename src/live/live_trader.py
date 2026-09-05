"""
src/live/live_trader.py
================================================================
The live trading engine (Phase 5). Run:  python -m src.live.live_trader

Loop design (two cadences):
- MANAGE every LIVE_MANAGE_EVERY_SECONDS (60s): reconcile vs broker, then
  run the exit stack on every open position against the live tick.
- ENTRY once per bar close (hourly, LIVE_ENTRY_DELAY_SECONDS after the
  hour): fresh bars -> indicators -> brain verdict -> the SAME gate chain
  the backtester runs -> portfolio risk engine -> size -> send.

Safety architecture:
- DRY_RUN (config, default ON): no order ever reaches the broker. Entries
  fill virtually at the live tick, get NEGATIVE ticket ids, and are managed
  by the exact same exit stack. Everything is journaled (flagged DRY_RUN).
- Real positions carry broker-side SL/TP from the first second - the exit
  stack's ratchets/trails only ever IMPROVE that bracket.
- The broker is the only truth: reconcile runs every cycle; positions that
  closed while we were down are journaled from deal history.
- Account guards: daily loss limit, daily profit target (+breakeven lock),
  drawdown circuit breaker - all account-wide, all in state.

Gate-chain fidelity: the entry gates mirror src/backtester/engine.py
exactly (session, spread, conviction, agreement, quality, ADX, tape
confirm, no-chase, tp-worth-spread). ONE deliberate addition, marked
below: the config-documented CRYPTO_MOMENTUM_GATE, which the backtest
engine never implemented (live is therefore strictly more conservative).
Sentiment is a Phase 6 hook - it is 0.0 here, so the sentiment vetoes
(±0.60) never engage, exactly as in the backtests.
"""
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from config.settings import config
from src.backtester.engine import (_quality_floor, _risk_pct, _session_ok,
                                   prepare_frame)
from src.ingestion.indicator_calculator import calculate_all_indicators
from src.live.journal import journal_open
from src.live.position_manager import (CLOSE_ALL, CLOSE_VOLUME, SET_SL,
                                       ManagedPosition, evaluate)
from src.live.reconciler import estimate_pnl, finalize_close, reconcile
from src.live.risk_engine import (RiskEngine, currency_exposure,
                                  entries_allowed, position_risk_usd,
                                  refresh_daily_guards)
from src.live.state import iso, load_state, now_utc, parse_iso, save_state
from src.memory.meta_learner import Brain
from src.mt5_client.connector import MT5Connector
from src.utils.logger import setup_logger
from src.utils.telegram import send_telegram

logger = setup_logger("LiveTrader", "logs/live.log")

MIN_BARS_FOR_FEATURES = 220       # sma200 + Wilder warm-up, else dropna empties
ENTRY_RETRY_WINDOW_MINUTES = 10   # give a slow broker this long to publish the bar


def _floor_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


class LiveTrader:
    def __init__(self):
        self.connector = MT5Connector()
        self.state = load_state(config.LIVE_STATE_PATH)
        self.universe: dict[str, dict] = {}
        self._brains: dict[str, Brain] = {}
        self._frame_cache: dict[str, tuple] = {}   # symbol -> (d, ts_idx), per entry cycle
        self._retry_symbols: set[str] = set()
        self._maintenance_proc: Optional[subprocess.Popen] = None
        self._stop = False

    # ------------------------------------------------------------------
    # startup
    # ------------------------------------------------------------------
    def setup(self) -> bool:
        if not self.connector.connect():
            logger.error("MT5 connection failed - cannot start")
            return False
        account = self.connector.account()
        if not account:
            logger.error("no account info - cannot start")
            return False
        if not account.get("trade_allowed", True):
            # Data reads still work, but NOTHING can trade. Loud, not fatal:
            # DRY_RUN remains fully usable for plumbing validation.
            send_telegram("MT5 reports Algo Trading DISABLED in the terminal. "
                          "Enable it (toolbar button green) before real orders.",
                          "critical")
        self.universe = self.connector.discover_universe()
        if not self.universe:
            logger.error("no pool symbols resolve at this broker")
            return False
        reconcile(self.state, self.connector, atr_for=self._atr_for,
                  dry_run=config.DRY_RUN)
        save_state(config.LIVE_STATE_PATH, self.state)
        mode = "DRY_RUN (virtual fills, zero broker orders)" if config.DRY_RUN \
            else "REAL ORDERS on this account"
        send_telegram(
            f"Nexus V4 live engine started\n"
            f"mode: {mode}\n"
            f"account {account.get('login')} @ {account.get('server')} | "
            f"equity {account.get('equity', 0):.0f} {account.get('currency')}\n"
            f"universe: {len(self.universe)} symbols | "
            f"open positions tracked: {len(self.state['positions'])}", "info")
        logger.info(f"setup complete: {len(self.universe)} symbols, "
                    f"DRY_RUN={config.DRY_RUN}")
        return True

    # ------------------------------------------------------------------
    # features + brain
    # ------------------------------------------------------------------
    def _brain(self, asset_class: str) -> Brain:
        if asset_class not in self._brains:
            self._brains[asset_class] = Brain(asset_class)
        return self._brains[asset_class]

    def _fresh_frame(self, symbol: str, now: datetime) -> Optional[tuple]:
        """Latest CLOSED bar feature frame for one symbol, prepared exactly
        like the backtester's (same indicator function, same prepare_frame,
        same broker point size). Cached per entry cycle."""
        if symbol in self._frame_cache:
            return self._frame_cache[symbol]
        out = None
        try:
            bars = self.connector.get_bars(symbol, config.LIVE_BARS_WINDOW)
            if bars is not None and len(bars) >= MIN_BARS_FOR_FEATURES:
                hour_start = _floor_hour(now)
                # drop the still-forming bar (its timestamp == current hour)
                if len(bars) and bars["timestamp"].iloc[-1] >= hour_start:
                    bars = bars.iloc[:-1]
                specs = self.connector.symbol_specs(symbol)
                point = specs["point"] if specs else None
                feats = calculate_all_indicators(bars, point=point)
                if not feats.empty:
                    out = prepare_frame(feats)
        except Exception as exc:
            logger.error(f"{symbol}: feature pipeline failed: {exc}")
        self._frame_cache[symbol] = out
        return out

    def _atr_for(self, symbol: str) -> float:
        frame = self._fresh_frame(symbol, now_utc())
        if not frame:
            return 0.0
        d, _ = frame
        atr = float(d["atr_14"].iloc[-1])
        return atr if atr == atr else 0.0   # NaN guard

    # ------------------------------------------------------------------
    # entry cycle (once per closed bar)
    # ------------------------------------------------------------------
    def _entry_cycle(self, now: datetime, hour_start: datetime,
                     only_symbols: set[str] = None) -> None:
        self._frame_cache = {}
        target_bar = pd.Timestamp(hour_start - timedelta(hours=1))
        not_ready: set[str] = set()

        account = self.connector.account()
        if not account:
            logger.warning("entry cycle: no account reading, skipping")
            return
        allowed, why = entries_allowed(self.state)
        if not allowed:
            logger.info(f"entry cycle: blocked account-wide ({why})")

        # open book as risk-engine dicts
        open_infos = []
        open_symbols = set()
        for pos_d in self.state["positions"].values():
            pos = ManagedPosition.from_dict(pos_d)
            open_symbols.add(pos.symbol)
            specs = self.connector.symbol_specs(pos.symbol)
            risk = position_risk_usd(pos.volume, abs(pos.entry_price - pos.sl)
                                     if pos.sl > 0 else config.STOP_ATR_MULT * pos.atr,
                                     specs or {})
            open_infos.append({"symbol": pos.symbol, "side": pos.side,
                               "volume": pos.volume, "sl": pos.sl,
                               "entry_price": pos.entry_price, "atr": pos.atr,
                               "risk_usd": risk})
        risk_engine = RiskEngine(account, open_infos)

        # ---- 1. re-judge OPEN positions (flip exits) --------------------
        for pos_d in list(self.state["positions"].values()):
            pos = ManagedPosition.from_dict(pos_d)
            if only_symbols and pos.symbol not in only_symbols:
                continue
            frame = self._fresh_frame(pos.symbol, now)
            if not frame:
                continue
            d, ts_idx = frame
            if ts_idx[-1] != target_bar:
                continue                      # judged on a stale bar = wrong
            try:
                verdict = self._brain(pos.asset_class).predict(d.iloc[-1], asof=now)
            except Exception as exc:
                logger.error(f"{pos.symbol}: brain failed ({exc})")
                continue
            if verdict is None:
                continue
            price = self.connector.get_latest_price(pos.symbol, pos.side)
            if price:
                self._apply_actions(pos, price,
                                    evaluate(pos, price, now, flip_verdict=verdict),
                                    now)

        # ---- 2. fresh entries -------------------------------------------
        if not allowed:
            self._retry_symbols = set()
            return
        for symbol, info in self.universe.items():
            if only_symbols and symbol not in only_symbols:
                continue
            if symbol in open_symbols:
                continue                      # one open position per symbol
            cls = info["asset_class"]
            frame = self._fresh_frame(symbol, now)
            if not frame:
                continue
            d, ts_idx = frame
            if ts_idx[-1] != target_bar:
                not_ready.add(symbol)         # broker hasn't published it yet
                continue
            cooldown = parse_iso(self.state["no_entry_until"].get(symbol))
            if cooldown and now < cooldown:
                continue
            if not _session_ok(ts_idx[-1], symbol):
                continue
            self._evaluate_entry(symbol, cls, d, now, account, risk_engine)
        self._retry_symbols = not_ready

    def _evaluate_entry(self, symbol: str, cls: str, d: pd.DataFrame,
                        now: datetime, account: dict,
                        risk_engine: RiskEngine) -> None:
        """The gate chain, mirroring engine.py bar-for-bar. Any skip is a
        silent log line; only full passes reach sizing."""
        i = len(d) - 1
        row = d.iloc[i]
        atr = float(d["atr_14"].iat[i])
        if not (atr == atr and atr > 0):
            return
        spread_price = float(d["spread_price"].iat[i])
        med20 = float(d["spread_med20"].iat[i])
        if config.SPREAD_FILTER_ENABLED and med20 == med20 and med20 > 0 \
                and spread_price > config.SPREAD_MAX_MULT_OF_MEDIAN * med20:
            return

        try:
            verdict = self._brain(cls).predict(row, asof=now)
        except Exception as exc:
            logger.error(f"{symbol}: brain failed ({exc})")
            return
        if verdict is None:
            return
        prob = verdict["prob"]
        long_sig = prob > config.BUY_THRESHOLD + config.ENTRY_CONVICTION_MARGIN
        short_sig = prob < config.SELL_THRESHOLD - config.ENTRY_CONVICTION_MARGIN
        if not (long_sig or short_sig):
            return
        if verdict["agreement"] < config.MIN_NEIGHBOR_AGREEMENT:
            return
        quality = verdict["quality"]
        if quality < _quality_floor(cls):
            return
        adx = float(d["adx_14"].iat[i])
        # (NaN < x is False, so a missing ADX passes - same as the engine)
        if config.ENTRY_ADX_MIN > 0 and adx < config.ENTRY_ADX_MIN \
                and quality < config.QUALITY_STRONG:
            return

        o, c = float(d["open"].iat[i]), float(d["close"].iat[i])
        if config.ENTRY_BAR_CONFIRM_ENABLED:
            body = c - o
            if long_sig and body < -config.ENTRY_BAR_CONFIRM_TOLERANCE_ATR * atr:
                return
            if short_sig and body > config.ENTRY_BAR_CONFIRM_TOLERANCE_ATR * atr:
                return
        if config.ENTRY_VWAP_CONFIRM_ENABLED:
            dv = float(d["dist_vwap"].iat[i])
            if dv == dv:
                vwap_atr = dv * c / atr
                if long_sig and vwap_atr < -config.ENTRY_VWAP_CONFIRM_TOLERANCE_ATR:
                    return
                if short_sig and vwap_atr > config.ENTRY_VWAP_CONFIRM_TOLERANCE_ATR:
                    return
        if config.ENTRY_NO_CHASE_ENABLED \
                and float(d["bar_range"].iat[i]) > config.ENTRY_NO_CHASE_MAX_RANGE_ATR * atr:
            return
        tp_dist = config.STOP_ATR_MULT * config.REWARD_RISK_RATIO * atr
        if config.MIN_TP_TO_SPREAD_MULT > 0 \
                and tp_dist < config.MIN_TP_TO_SPREAD_MULT * spread_price:
            return
        # LIVE-ONLY addition (config-documented, absent from the backtest
        # engine): crypto LONGs need price above sma200 AND positive 12h momentum.
        if config.CRYPTO_MOMENTUM_GATE and cls == "crypto" and long_sig:
            ds200 = float(d["dist_sma200"].iat[i])
            r12 = float(d["ret_12"].iat[i])
            if not (ds200 > 0 and r12 > 0):
                return

        self._execute_entry(symbol, cls, "LONG" if long_sig else "SHORT",
                            quality, verdict, atr, float(d["spread_pct"].iat[i]),
                            str(d["regime_label"].iat[i]), now, account,
                            risk_engine)

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------
    def _execute_entry(self, symbol: str, cls: str, side: str, quality: float,
                       verdict: dict, atr: float, spread_pct: float,
                       regime: str, now: datetime, account: dict,
                       risk_engine: RiskEngine) -> None:
        s = 1 if side == "LONG" else -1
        equity = float(account.get("equity") or 0.0)
        slice_cap = min(config.CAPITAL_PER_SYMBOL, equity * config.SLICE_PCT_OF_EQUITY)
        risk_usd = slice_cap * _risk_pct(quality)
        stop_dist = config.STOP_ATR_MULT * atr

        tick = self.connector.get_tick(symbol)
        if tick is None:
            return
        entry_ref = float(tick.ask if side == "LONG" else tick.bid)
        sl = entry_ref - s * stop_dist
        tp = entry_ref + s * stop_dist * config.REWARD_RISK_RATIO

        lots = self.connector.volume_for_risk(symbol, risk_usd, stop_dist)
        specs = self.connector.symbol_specs(symbol)
        if specs and lots > 0:
            contract = specs.get("contract_size") or 0.0
            if contract > 0 and entry_ref > 0:
                cap = min(slice_cap * config.NOTIONAL_CAP_PCT,
                          config.NOTIONAL_CAP_ABS)
                max_lots = cap / (contract * entry_ref)
                if lots > max_lots:
                    lots = MT5Connector._round_volume(specs, max_lots)
        if lots <= 0:
            logger.info(f"{symbol}: sizing produced 0 lots (risk ${risk_usd:.0f})")
            return

        ok, why = risk_engine.check_entry(symbol, side, lots, sl, entry_ref,
                                          specs or {}, atr)
        if not ok:
            logger.info(f"{symbol}: risk engine veto ({why})")
            return

        if config.DRY_RUN:
            ticket = int(self.state["virt_ticket_seq"])
            self.state["virt_ticket_seq"] = ticket - 1
            fill_price = entry_ref
        else:
            res = self.connector.place_market_order(
                symbol, side, lots, sl, tp, comment=f"q{quality:.2f}")
            if not res:
                send_telegram(f"{symbol} {side}: order REJECTED by broker "
                              f"(see mt5.log)", "warning")
                return
            ticket = int(res["ticket"])
            fill_price = float(res["price"])

        pos = ManagedPosition(
            ticket=ticket, symbol=symbol, asset_class=cls, side=side,
            entry_price=fill_price, entry_time=iso(now),
            initial_volume=lots, volume=lots, atr=atr, sl=sl, tp=tp,
            peak_price=fill_price, trough_price=fill_price,
            quality=quality, prob=float(verdict["prob"]),
            agreement=float(verdict["agreement"]),
            memory_n=int(verdict.get("n_kept", 0)), regime=regime,
            spread_pct_at_entry=spread_pct,
            risk_usd=position_risk_usd(lots, stop_dist, specs or {}),
            dry_run=config.DRY_RUN)
        self.state["positions"][str(ticket)] = pos.to_dict()
        save_state(config.LIVE_STATE_PATH, self.state)

        journal_open(magic=self.connector.magic_for(symbol), symbol=symbol,
                     asset_class=cls, side=side, volume_lots=lots,
                     entry_time=now, entry_price=fill_price, sl=sl, tp=tp,
                     quality=quality, eff_quality=quality, regime=regime,
                     sentiment=0.0, memory_n=pos.memory_n,
                     spread_pct=spread_pct, atr=atr, ticket=ticket,
                     dry_run=config.DRY_RUN)
        send_telegram(
            f"{symbol} {side} {lots} lots @ {fill_price}\n"
            f"SL {round(sl, specs['digits'] if specs else 5)} | "
            f"TP {round(tp, specs['digits'] if specs else 5)} | "
            f"q={quality:.2f} p={verdict['prob']:.3f} n={pos.memory_n}"
            + (" | DRY_RUN" if config.DRY_RUN else ""), "entry")

        # the book changed: later entries this cycle must see it
        risk_engine.open_positions.append(
            {"symbol": symbol, "side": side, "volume": lots, "sl": sl,
             "entry_price": fill_price, "atr": atr, "risk_usd": pos.risk_usd})
        risk_engine.total_open_risk += pos.risk_usd
        risk_engine.exposure = currency_exposure(risk_engine.open_positions)

    # ------------------------------------------------------------------
    # management cycle (every LIVE_MANAGE_EVERY_SECONDS)
    # ------------------------------------------------------------------
    def _manage(self, now: datetime) -> None:
        reconcile(self.state, self.connector, atr_for=self._atr_for,
                  dry_run=config.DRY_RUN)

        broker_by_ticket = {}
        if not config.DRY_RUN:
            try:
                broker_by_ticket = {int(p["ticket"]): p
                                    for p in self.connector.positions(ours_only=True)}
            except Exception:
                pass

        for ticket_s, pos_d in list(self.state["positions"].items()):
            pos = ManagedPosition.from_dict(pos_d)
            bp = broker_by_ticket.get(pos.ticket)
            if bp:      # resync with broker truth (volume/sl/tp may move)
                pos.volume = float(bp["volume"])
                pos.sl = float(bp["sl"] or 0.0)
                pos.tp = float(bp["tp"] or 0.0)
            price = self.connector.get_latest_price(pos.symbol, pos.side)
            if price is None or price <= 0:
                continue
            if pos.sign > 0:
                pos.peak_price = max(pos.peak_price, price)
                pos.trough_price = min(pos.trough_price, price)
            else:
                pos.peak_price = min(pos.peak_price, price)
                pos.trough_price = max(pos.trough_price, price)
            self._apply_actions(pos, price, evaluate(pos, price, now), now)

    def _apply_actions(self, pos: ManagedPosition, price: float,
                       actions: list, now: datetime) -> None:
        for kind, payload, reason in actions:
            if kind == SET_SL:
                new_sl = float(payload)
                if pos.dry_run:
                    pos.sl = new_sl
                elif self.connector.modify_sltp(pos.ticket, pos.symbol,
                                                new_sl, pos.tp):
                    pos.sl = new_sl
                else:
                    continue                     # retry next cycle
                if reason in ("ratchet", "flip_tighten"):
                    send_telegram(f"{pos.symbol}: stop {reason} -> {new_sl}",
                                  "info")
                if reason == "flip_tighten":
                    pos.flip_tightened = True

            elif kind == CLOSE_VOLUME:
                lots = float(payload)
                specs = self.connector.symbol_specs(pos.symbol)
                if specs:
                    lots = MT5Connector._round_volume(specs, lots)
                if lots <= 0 or lots >= pos.volume:
                    continue
                est = estimate_pnl(pos, price, specs, lots)
                if pos.dry_run:
                    pos.realized_pnl += est
                    pos.volume = round(pos.volume - lots, 8)
                elif self.connector.close_position(pos.symbol, lots=lots,
                                                   ticket=pos.ticket,
                                                   comment=reason):
                    pos.volume = round(pos.volume - lots, 8)
                else:
                    continue
                if reason == "scale_out_1":
                    pos.scaled_1 = True
                elif reason == "scale_out_2":
                    pos.scaled_2 = True
                elif reason == "time_partial":
                    pos.time_partialed = True
                send_telegram(f"{pos.symbol}: {reason} {lots} lots "
                              f"(est {est:+.2f})", "partial")

            elif kind == CLOSE_ALL:
                if pos.dry_run:
                    specs = self.connector.symbol_specs(pos.symbol)
                    pnl = pos.realized_pnl + estimate_pnl(pos, price, specs)
                    exit_price, exit_time = price, now
                else:
                    if not self.connector.close_position(
                            pos.symbol, ticket=pos.ticket, comment=reason):
                        continue                 # retry next cycle
                    summary = None
                    for _ in range(2):
                        try:
                            summary = self.connector.closed_position_summary(
                                pos.ticket)
                        except Exception:
                            summary = None
                        if summary:
                            break
                        time.sleep(1)
                    if summary:
                        exit_price = summary["exit_price"]
                        exit_time = summary["exit_time"]
                        pnl = summary["profit"]
                    else:
                        specs = self.connector.symbol_specs(pos.symbol)
                        exit_price, exit_time = price, now
                        pnl = estimate_pnl(pos, price, specs)
                finalize_close(self.state, pos, exit_price=exit_price,
                               exit_time=exit_time, reason=reason, pnl=pnl)
                return
        # position is gone
        self.state["positions"][str(pos.ticket)] = pos.to_dict()

    # ------------------------------------------------------------------
    # schedulers: heartbeat / EOD / maintenance
    # ------------------------------------------------------------------
    def _schedulers(self, now: datetime, account: dict) -> None:
        equity = float(account.get("equity") or 0.0)
        day = self.state["day"]

        last_hb = parse_iso(self.state.get("last_heartbeat"))
        if last_hb is None or (now - last_hb).total_seconds() >= config.HEARTBEAT_SECONDS:
            self.state["last_heartbeat"] = iso(now)
            send_telegram(
                f"equity {equity:.2f} | open {len(self.state['positions'])} | "
                f"day pnl {float(day.get('realized_pnl') or 0):+.2f} | "
                f"closed today {int(day.get('closed_count') or 0)}"
                + (" | DRY_RUN" if config.DRY_RUN else ""), "heartbeat")

        if config.TELEGRAM_EOD_REPORT and now.hour == 21 \
                and self.state.get("last_eod_date") != now.date().isoformat():
            self.state["last_eod_date"] = now.date().isoformat()
            lines = [f"EOD report {now.date().isoformat()}",
                     f"equity {equity:.2f} | day realized "
                     f"{float(day.get('realized_pnl') or 0):+.2f} | "
                     f"closed {int(day.get('closed_count') or 0)}"]
            for pos_d in self.state["positions"].values():
                pos = ManagedPosition.from_dict(pos_d)
                lines.append(f"  open: {pos.symbol} {pos.side} {pos.volume} "
                             f"@ {pos.entry_price}")
            send_telegram("\n".join(lines), "report")

        if config.DAILY_MAINTENANCE_ENABLED \
                and now.hour == config.DAILY_MAINTENANCE_UTC_HOUR \
                and self.state.get("last_maintenance_date") != now.date().isoformat():
            self.state["last_maintenance_date"] = now.date().isoformat()
            if self._maintenance_proc is None or self._maintenance_proc.poll() is not None:
                logger.info("starting daily maintenance (feature pump)")
                try:
                    self._maintenance_proc = subprocess.Popen(
                        [sys.executable, "-m", "src.ingestion.run_pump"])
                except Exception as exc:
                    send_telegram(f"maintenance failed to start: {exc}", "warning")
        if self._maintenance_proc is not None:
            rc = self._maintenance_proc.poll()
            if rc is not None:
                if rc != 0:
                    send_telegram(f"daily maintenance exited with code {rc} "
                                  f"- check logs/ingestion.log", "warning")
                else:
                    logger.info("daily maintenance completed OK")
                self._maintenance_proc = None

    def _lock_breakeven(self, now: datetime) -> None:
        """Daily profit target hit: pull every stop to at least entry."""
        for pos_d in list(self.state["positions"].values()):
            pos = ManagedPosition.from_dict(pos_d)
            be = pos.entry_price
            improves = (be > pos.sl) if pos.sign > 0 else (pos.sl == 0 or be < pos.sl)
            if improves:
                self._apply_actions(pos, be, [(SET_SL, be, "breakeven_lock")], now)
        send_telegram("daily profit target hit - stops locked to breakeven, "
                      "no new entries today", "target")

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run(self, once: bool = False) -> int:
        if not self.setup():
            return 1
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "_stop", True))
        try:
            signal.signal(signal.SIGTERM, lambda *_: setattr(self, "_stop", True))
        except (ValueError, OSError):
            pass                                # not in main thread / platform

        while not self._stop:
            now = now_utc()
            try:
                account = self.connector.account() or {}
                events = refresh_daily_guards(self.state, account, now)
                for ev in events:
                    if ev == "daily_loss_limit":
                        send_telegram("DAILY LOSS LIMIT hit - no new entries "
                                      "until tomorrow (UTC)", "critical")
                    elif ev == "daily_profit_target":
                        self._lock_breakeven(now)
                    elif ev == "drawdown_breaker":
                        send_telegram(f"DRAWDOWN BREAKER: equity fell "
                                      f"{config.MAX_DRAWDOWN_PCT:.0%} from peak - "
                                      f"entries halted until tomorrow", "critical")

                self._manage(now)

                hour_start = _floor_hour(now)
                due = now >= hour_start + timedelta(seconds=config.LIVE_ENTRY_DELAY_SECONDS)
                last = parse_iso(self.state.get("last_entry_hour"))
                if due and (last is None or last < hour_start):
                    self._entry_cycle(now, hour_start)
                    self.state["last_entry_hour"] = iso(hour_start)
                elif self._retry_symbols and due and \
                        now < hour_start + timedelta(minutes=ENTRY_RETRY_WINDOW_MINUTES):
                    self._entry_cycle(now, hour_start,
                                      only_symbols=self._retry_symbols)

                self._schedulers(now, account)
                save_state(config.LIVE_STATE_PATH, self.state)
            except Exception as exc:
                logger.exception(f"cycle error: {exc}")
                send_telegram(f"cycle error (engine alive): {exc}", "critical")
            if once:
                break
            time.sleep(config.LIVE_MANAGE_EVERY_SECONDS)

        save_state(config.LIVE_STATE_PATH, self.state)
        self.connector.shutdown()
        logger.info("live engine stopped cleanly")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Nexus V4 live engine (Phase 5)")
    ap.add_argument("--once", action="store_true",
                    help="run a single manage+entry cycle and exit (testing)")
    args = ap.parse_args()
    return LiveTrader().run(once=args.once)


if __name__ == "__main__":
    sys.exit(main())
