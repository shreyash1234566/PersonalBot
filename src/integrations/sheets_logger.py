"""
Google Sheets Logger
====================
Append-only chat log for visibility.
"""

import json
from datetime import datetime

from google.oauth2 import service_account
from googleapiclient.discovery import build

from src.config import (
    SHEETS_SPREADSHEET_ID,
    SHEETS_TAB_NAME,
    SHEETS_SERVICE_ACCOUNT_JSON,
    ENABLE_SHEETS_LOG,
)


class SheetsLogger:
    """Append chat events to a Google Sheet if enabled."""

    def __init__(self):
        self._service = None

    @property
    def enabled(self) -> bool:
        return ENABLE_SHEETS_LOG and bool(SHEETS_SPREADSHEET_ID)

    @property
    def service(self):
        if self._service is None:
            if not SHEETS_SERVICE_ACCOUNT_JSON:
                raise RuntimeError("SHEETS_SERVICE_ACCOUNT_JSON not set")
            info = json.loads(SHEETS_SERVICE_ACCOUNT_JSON)
            creds = service_account.Credentials.from_service_account_info(
                info,
                scopes=["https://www.googleapis.com/auth/spreadsheets"],
            )
            self._service = build("sheets", "v4", credentials=creds)
        return self._service

    def append_message(
        self,
        conversation_id: str,
        partner_name: str,
        role: str,
        content: str,
        provider: str = "",
        timestamp: str = None,
    ):
        if not self.enabled:
            return

        ts = timestamp or datetime.utcnow().isoformat()
        values = [[ts, conversation_id, partner_name, role, content, provider]]
        body = {"values": values}
        sheet_range = f"{SHEETS_TAB_NAME}!A1"

        self.service.spreadsheets().values().append(
            spreadsheetId=SHEETS_SPREADSHEET_ID,
            range=sheet_range,
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body=body,
        ).execute()
