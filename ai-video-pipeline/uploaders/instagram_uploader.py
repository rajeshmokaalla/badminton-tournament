"""Upload Reels to Instagram via Meta Graph API."""

import logging
import time
from pathlib import Path

import requests

import config
from utils.helpers import timed

logger = logging.getLogger(__name__)

_BASE = f"https://graph.facebook.com/{config.META_API_VERSION}"
_MAX_POLL = 30          # maximum status-check iterations
_POLL_INTERVAL = 10     # seconds between status checks


class InstagramUploader:
    def __init__(
        self,
        access_token: str = config.INSTAGRAM_ACCESS_TOKEN,
        account_id: str = config.INSTAGRAM_ACCOUNT_ID,
    ):
        if not access_token:
            raise ValueError("INSTAGRAM_ACCESS_TOKEN is required")
        if not account_id:
            raise ValueError("INSTAGRAM_ACCOUNT_ID is required")
        self.access_token = access_token
        self.account_id = account_id

    # ── Public API ────────────────────────────────────────────────────────────

    @timed("instagram_upload")
    def upload_reel(
        self,
        video_url: str,          # publicly accessible HTTPS URL to the video
        caption: str = "",
        share_to_feed: bool = True,
    ) -> str:
        """
        Publish a Reel; returns the published media ID.

        Parameters
        ----------
        video_url : str
            A publicly accessible URL (Meta servers must be able to fetch it).
            Use a CDN / ngrok / S3 pre-signed URL for local files.
        caption : str
            Caption text (hashtags are included here).
        share_to_feed : bool
            Whether to also share to the main feed.
        """
        logger.info("Creating Instagram Reel container …")
        container_id = self._create_container(video_url, caption, share_to_feed)

        logger.info("Waiting for container to be ready (id=%s) …", container_id)
        self._wait_for_ready(container_id)

        logger.info("Publishing Reel …")
        media_id = self._publish(container_id)
        logger.info("Instagram Reel published: media_id=%s", media_id)
        return media_id

    # ── Internal ──────────────────────────────────────────────────────────────

    def _create_container(self, video_url: str, caption: str, share_to_feed: bool) -> str:
        url = f"{_BASE}/{self.account_id}/media"
        params = {
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "share_to_feed": str(share_to_feed).lower(),
            "access_token": self.access_token,
        }
        resp = requests.post(url, data=params, timeout=60)
        self._check(resp)
        return resp.json()["id"]

    def _wait_for_ready(self, container_id: str) -> None:
        url = f"{_BASE}/{container_id}"
        params = {
            "fields": "status_code,status",
            "access_token": self.access_token,
        }
        for attempt in range(_MAX_POLL):
            resp = requests.get(url, params=params, timeout=30)
            self._check(resp)
            data = resp.json()
            status = data.get("status_code", "")
            logger.debug("Container status: %s", status)
            if status == "FINISHED":
                return
            if status == "ERROR":
                raise RuntimeError(f"Instagram container error: {data.get('status')}")
            time.sleep(_POLL_INTERVAL)
        raise TimeoutError("Instagram container did not become ready in time")

    def _publish(self, container_id: str) -> str:
        url = f"{_BASE}/{self.account_id}/media_publish"
        params = {
            "creation_id": container_id,
            "access_token": self.access_token,
        }
        resp = requests.post(url, data=params, timeout=60)
        self._check(resp)
        return resp.json()["id"]

    @staticmethod
    def _check(resp: requests.Response) -> None:
        if not resp.ok:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise RuntimeError(f"Meta API error {resp.status_code}: {detail}")
