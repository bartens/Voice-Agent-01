"""Diagnose-Skript für Google Calendar Event-Erstellung.

Funktionen:
 1. Pre-Auth testen
 2. Kalender-ID auflösen
 3. Kleines Test-Event in ~10 Minuten für 15 Minuten anlegen
 4. Direkt danach per events.get verifizieren
 5. Kürzlich kommende Events (nächste 2 Stunden) listen

Nutzung (PowerShell):
    $env:CALENDAR_TRACE='1'
    python event_diagnose.py

Falls Fehler auftreten, Ausgabe kopieren und zur Analyse senden.
"""
from __future__ import annotations
from datetime import datetime, timedelta
from calendar_tools import pre_auth, create_calendar_event, list_calendar_events


def main():
    print('[DIA] Starte Diagnose...')
    if not pre_auth():
        print('[DIA][FAIL] pre_auth fehlgeschlagen – OAuth / client_secret.json prüfen.')
        return
    print('[DIA] OAuth / Token OK.')
    now = datetime.now().replace(microsecond=0)
    start = now + timedelta(minutes=10)
    end = start + timedelta(minutes=15)
    print(f"[DIA] Versuche Test-Insert: {start.isoformat()} -> {end.isoformat()}")
    try:
        res = create_calendar_event(
            summary='Diag Test',
            start_iso=start.isoformat(),
            end_iso=end.isoformat(),
            timezone='Europe/Berlin',
            description='Diagnose Insert'
        )
        print('[DIA][OK] Insert Response:', {k: res.get(k) for k in ['eventId','htmlLink','summary','calendarId','verified']})
    except Exception as e:
        print('[DIA][FAIL] Insert Fehler:', e)
        return
    # Listing nächster 2 Stunden
    try:
        list_res = list_calendar_events(
            start_iso=now.isoformat(),
            end_iso=(now + timedelta(hours=2)).isoformat(),
            max_results=10
        )
        print(f"[DIA] Events im Zeitfenster: count={list_res.get('count')}")
        for ev in list_res.get('events', []):
            print('   -', ev.get('summary'), ev.get('start'), '->', ev.get('end'))
    except Exception as e:
        print('[DIA][WARN] Listing Fehler:', e)


if __name__ == '__main__':
    main()
