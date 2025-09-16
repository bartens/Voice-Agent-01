"""Audio Test Script

Funktionen:
 - Wählt Mikrofon- und Lautsprecher-Gerät anhand von `config.py` (Name oder ID)
 - Ermittelt geeignete Sample-Rate und Kanalanzahl mit Fallbacks
 - Nimmt kurz Audio auf, zeigt Pegel/Statistik und spielt es wieder ab
 - Liefert klare Fehlermeldungen mit Hinweisen zur Behebung
"""

import sys
import argparse
from typing import List

import sounddevice as sd
import numpy as np
from config import MIC_DEVICE_NAME, SPEAKER_DEVICE_NAME

"""Hinweis: Geräte werden jetzt anhand der Namen / IDs aus config.py gesucht.
Falls kein Treffer: Fallback auf Default-Gerät mit WARN-Hinweis."""

DURATION_SEC = 5.0
PREFERRED_SAMPLE_RATES = [48000, 44100, 32000, 16000]



def _resolve_device(spec, kind: str) -> int:
	devices = sd.query_devices()
	# direkte ID
	if isinstance(spec, int):
		if 0 <= spec < len(devices):
			return spec
		raise ValueError(f"Geräte-ID {spec} existiert nicht.")
	# Versuch als int-String
	if isinstance(spec, str) and spec.isdigit():
		idx = int(spec)
		if 0 <= idx < len(devices):
			return idx
	# Name / Substring Suche
	if isinstance(spec, str):
		lowered = spec.lower()
		candidates = [i for i, d in enumerate(devices) if lowered in d['name'].lower() and d[f'max_{kind}_channels'] > 0]
		if candidates:
			return candidates[0]
	# Default fallback
	default_in, default_out = sd.default.device
	return default_in if kind == 'input' else default_out


def choose_sample_rate(device_id: int) -> int:
	dev = sd.query_devices(device_id)
	default_sr = int(dev.get('default_samplerate') or 0)
	if default_sr:
		return default_sr
	# fallback sequence
	for sr in PREFERRED_SAMPLE_RATES:
		try:
			sd.check_input_settings(device=device_id, samplerate=sr)
			return sr
		except Exception:  # pragma: no cover
			continue
	# Letzter Ausweg
	return 16000


def choose_channels(device_id: int) -> int:
	dev = sd.query_devices(device_id)
	max_in = dev['max_input_channels']
	if max_in >= 1:
		# Bevorzuge Mono (geringere Latenz / Bandbreite)
		return 1
	raise RuntimeError(f"Gerät {device_id} bietet keine Input-Kanäle")


def record_audio(device_id: int, samplerate: int, channels: int, duration: float, use_wasapi: bool = True) -> np.ndarray:
	frames = int(samplerate * duration)
	print(f"Starte Aufnahme: {duration:.1f}s @ {samplerate} Hz, ch={channels}, device={device_id} (WASAPI={use_wasapi})")
	extra = None
	if use_wasapi and hasattr(sd, "WasapiSettings"):
		try:
			extra = sd.WasapiSettings(exclusive=False, loopback=False)
		except Exception:
			extra = None
	try:
		data = sd.rec(frames, samplerate=samplerate, channels=channels, dtype='int16', device=device_id, extra_settings=extra)
		sd.wait()
	except Exception as e:
		if use_wasapi:
			print("WASAPI Versuch fehlgeschlagen, versuche ohne extra_settings ...")
			return record_audio(device_id, samplerate, channels, duration, use_wasapi=False)
		raise RuntimeError(f"Aufnahme fehlgeschlagen: {e}") from e
	return data


def brute_force_open(device_id: int, duration: float) -> None:
	"""Versucht verschiedene (samplerate, channels, hostapi) Kombinationen für das Eingabegerät.
	Liefert detaillierte Fehlermeldungen zur Eingrenzung.
	"""
	print("\n=== Brute-Force Gerätekombinationen testen ===")
	dev = sd.query_devices(device_id)
	hostapis = sd.query_hostapis()
	# Liste möglicher Samplerates: default, bevorzugte, plus 22050
	candidate_rates = []
	if dev.get('default_samplerate'):
		candidate_rates.append(int(dev['default_samplerate']))
	for r in [48000, 44100, 32000, 22050, 16000]:
		if r not in candidate_rates:
			candidate_rates.append(r)
	candidate_channels = [1, 2]
	tried = 0
	successes = []
	failures = []
	for ha_index, ha in enumerate(hostapis):
		for sr in candidate_rates:
			for ch in candidate_channels:
				try:
					# Direkt RawInputStream (um Overhead zu vermeiden)
					stream = sd.InputStream(device=device_id, samplerate=sr, channels=ch, dtype='int16', blocksize=0, latency='low')
					stream.start()
					data, _ = stream.read(int(sr * min(0.25, duration)))
					stream.stop(); stream.close()
					peak = int(np.max(np.abs(data))) if data.size else 0
					print(f"  OK hostapi={ha_index}({ha['name']}), sr={sr}, ch={ch}, peak={peak}")
					successes.append((ha_index, sr, ch))
					# Erste erfolgreiche Kombination reicht für Rückkehr
					return
				except Exception as e:  # pragma: no cover
					msg = str(e)
					# typische Fehler extrahieren
					tag = "INVALID_CH" if "Invalid number of channels" in msg else "+HOST" if "host error" in msg.lower() else "ERR"
					failures.append((ha_index, sr, ch, tag))
					tried += 1
					if tried % 10 == 0:
						print(f"  ... {tried} Kombinationen getestet ...")
	if not successes:
		print("Keine funktionierende Kombination gefunden.")
		# Kurzer Report gruppiert nach Tag
		invalid_channels = [f for f in failures if f[3] == 'INVALID_CH']
		host_errors = [f for f in failures if f[3] == '+HOST']
		print(f"  INVALID_CH Fälle: {len(invalid_channels)}; HOST Errors: {len(host_errors)}; Gesamt: {len(failures)}")
		hints = (
			"Hinweise:\n"
			" - Prüfe Windows Datenschutz (Einstellungen > Datenschutz & Sicherheit > Mikrofon).\n"
			" - Deaktiviere 'Exklusiven Modus' im Geräteeigenschaften-Dialog (Aufnahmegerät > Eigenschaften > Erweitert).\n"
			" - Schalte temporär Sound-Verbesserungen aus.\n"
			" - Falls ein Audio-Manager (Realtek, Nahimic) läuft: beenden.\n"
			" - Teste mit anderem physischen Mikrofon oder USB-Adapter.\n"
			" - Führe 'mmsys.cpl' aus und setze das Gerät als Standardkommunikationsgerät.\n"
		)
		print(hints)



def analyze_audio(data: np.ndarray) -> None:
	if data.ndim > 1:
		mono = data.mean(axis=1)
	else:
		mono = data
	rms = np.sqrt(np.mean(mono.astype(np.float32) ** 2))
	peak = np.max(np.abs(mono))
	if peak == 0:
		dbfs = -120.0
	else:
		dbfs = 20 * np.log10(peak / 32767.0)
	print(f"Analyse: Samples={len(mono)}, RMS={rms:.1f}, Peak={peak}, Peak dBFS={dbfs:.1f}")


def playback_audio(data: np.ndarray, device_id: int, samplerate: int) -> None:
	print(f"Spiele Aufnahme ab über Gerät {device_id} @ {samplerate} Hz ...")
	try:
		sd.play(data, samplerate=samplerate, device=device_id)
		sd.wait()
	except Exception as e:
		raise RuntimeError(f"Wiedergabe fehlgeschlagen: {e}") from e


def diagnose_inputs(short: float = 0.5) -> None:
	print("\n=== Diagnose aller Eingabegeräte ===")
	devices = sd.query_devices()
	failures: List[str] = []
	for idx, dev in enumerate(devices):
		if dev['max_input_channels'] <= 0:
			continue
		print(f"\n>> Test Device {idx}: {dev['name']}")
		# Test verschiedene Samplerates & channels=1
		test_rates = [int(dev.get('default_samplerate') or 0)] + PREFERRED_SAMPLE_RATES
		seen = set()
		ok_any = False
		for sr in test_rates:
			if sr <= 0 or sr in seen:
				continue
			seen.add(sr)
			try:
				snippet = record_audio(idx, sr, 1, short)
				print(f"  OK @ {sr}Hz - Peak {np.max(np.abs(snippet))}")
				ok_any = True
				break
			except Exception as e:  # pragma: no cover
				print(f"  FAIL @ {sr}Hz: {e}")
		if not ok_any:
			failures.append(f"Device {idx} {dev['name']}")
	if failures:
		print("\nNicht nutzbare Geräte:")
		for f in failures:
			print(" -", f)
	print("=== Diagnose Ende ===\n")


def _pick_device(requested: str | int, kind: str) -> int:
	devices = sd.query_devices()
	if isinstance(requested, int):
		if 0 <= requested < len(devices):
			if devices[requested][f'max_{kind}_channels'] > 0:
				return requested
			raise RuntimeError(f"Gerät-ID {requested} hat keine {kind}-Kanäle")
		raise RuntimeError(f"Gerät-ID {requested} existiert nicht")
	# string -> try exact (case-insensitive) then substring
	req_lower = str(requested).lower()
	exact = [i for i,d in enumerate(devices) if d['name'].lower() == req_lower and d[f'max_{kind}_channels']>0]
	if exact:
		return exact[0]
	subs = [i for i,d in enumerate(devices) if req_lower in d['name'].lower() and d[f'max_{kind}_channels']>0]
	if subs:
		return subs[0]
	# fallback default
	def_in, def_out = sd.default.device
	fallback = def_in if kind=='input' else def_out
	print(f"[WARN] Kein Treffer für '{requested}' ({kind}); benutze Default {fallback} -> {devices[fallback]['name']}")
	return fallback


def main():
	parser = argparse.ArgumentParser(description="Audio Test & Diagnose")
	parser.add_argument("--diagnose", action="store_true", help="Alle Eingabegeräte testen")
	parser.add_argument("--seconds", type=float, default=DURATION_SEC, help="Aufnahmedauer")
	parser.add_argument("--mono", action="store_true", help="Erzwinge 1 Kanal")
	args = parser.parse_args()

	print("=== Audio Test Start ===")
	if args.diagnose:
		diagnose_inputs()
	print(f"Konfiguration: MIC_DEVICE_NAME='{MIC_DEVICE_NAME}' SPEAKER_DEVICE_NAME='{SPEAKER_DEVICE_NAME}'")
	mic_id = _pick_device(MIC_DEVICE_NAME, 'input')
	spk_id = _pick_device(SPEAKER_DEVICE_NAME, 'output')
	mic_info = sd.query_devices(mic_id)
	spk_info = sd.query_devices(spk_id)
	print(f"Gewähltes Mikrofon (config): {mic_id} -> {mic_info['name']}")
	print(f"Gewählter Lautsprecher (config): {spk_id} -> {spk_info['name']}")
	samplerate = choose_sample_rate(mic_id)
	channels = 1 if args.mono else choose_channels(mic_id)
	print(f"Verwende Samplerate={samplerate}, Channels={channels}")
	try:
		data = record_audio(mic_id, samplerate, channels, args.seconds)
	except Exception as e:
		print("Primäre Aufnahme fehlgeschlagen:", e)
		brute_force_open(mic_id, args.seconds)
		raise
	analyze_audio(data)
	playback_audio(data, spk_id, samplerate)
	print("=== Audio Test Ende ===")


if __name__ == "__main__":  # pragma: no cover
	try:
		main()
	except Exception as exc:
		print("FEHLER:", exc, file=sys.stderr)
		print("Hinweise:")
		print(" - Prüfe, ob ein anderes Programm das Mikrofon blockiert.")
		print(" - Teste ein anderes Gerät oder reduziere Samplerate (config anpassen).")
		print(" - Stelle sicher, dass Treiber aktuell sind.")
		sys.exit(1)
