"""Upload videos to YouTube Shorts via YouTube Data API v3."""

import logging
import os
from pathlib import Path

import config
from utils.helpers import timed

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

# YouTube Shorts are detected automatically when video is ≤60s and 9:16.
# Appending #Shorts to the title also helps discovery.


class YouTubeUploader:
    def __init__(
        self,
        client_secrets: str = config.YOUTUBE_CLIENT_SECRETS,
        token_file: str = config.YOUTUBE_TOKEN_FILE,
        privacy: str = config.YOUTUBE_PRIVACY,
    ):
        self.client_secrets = client_secrets
        self.token_file = token_file
        self.privacy = privacy
        self._service = None

    # ── Public API ────────────────────────────────────────────────────────────

    @timed("youtube_upload")
    def upload(
        self,
        video_path: Path,
        title: str,
        description: str = "",
        tags: list[str] | None = None,
        category_id: str = config.YOUTUBE_CATEGORY_ID,
    ) -> str:
        """Upload *video_path* as a YouTube Short; returns the video ID."""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        tags = tags or []
        shorts_title = f"{title} #Shorts"[:100]   # YouTube title limit

        body = {
            "snippet": {
                "title": shorts_title,
                "description": f"{description}\n\n#Shorts",
                "tags": tags + ["Shorts"],
                "categoryId": category_id,
            },
            "status": {"privacyStatus": self.privacy},
        }

        try:
            service = self._get_service()
            from googleapiclient.http import MediaFileUpload  # type: ignore

            media = MediaFileUpload(
                str(video_path),
                mimetype="video/mp4",
                resumable=True,
                chunksize=1024 * 1024 * 5,  # 5 MB chunks
            )
            request = service.videos().insert(
                part="snippet,status",
                body=body,
                media_body=media,
            )
            response = self._resumable_upload(request)
            video_id = response.get("id", "unknown")
            url = f"https://youtube.com/shorts/{video_id}"
            logger.info("YouTube upload complete: %s", url)
            return video_id

        except Exception as exc:
            logger.error("YouTube upload failed: %s", exc)
            raise

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_service(self):
        if self._service is not None:
            return self._service

        from google.oauth2.credentials import Credentials  # type: ignore
        from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore
        from google.auth.transport.requests import Request  # type: ignore
        from googleapiclient.discovery import build  # type: ignore

        creds = None
        token_path = Path(self.token_file)
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), _SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.client_secrets, _SCOPES
                )
                creds = flow.run_local_server(port=0)
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(creds.to_json())

        self._service = build("youtube", "v3", credentials=creds)
        return self._service

    @staticmethod
    def _resumable_upload(request):
        """Execute a resumable upload with retry on transient errors."""
        import httplib2
        from googleapiclient.errors import HttpError  # type: ignore

        response = None
        error = None
        retry = 0
        while response is None:
            try:
                _, response = request.next_chunk()
                if response is not None:
                    return response
            except HttpError as e:
                if e.resp.status in [500, 502, 503, 504]:
                    error = e
                else:
                    raise
            except (IOError, httplib2.HttpLib2Error) as e:
                error = e

            retry += 1
            if retry > 10:
                raise error
            import time
            time.sleep(2 ** retry)
        return response
