"""Branding UI components."""

import base64
import os

from khmer_rag.config import PROJECT_ROOT

CHATBOT_NAME = "ជំនួយការធានារ៉ាប់រងកម្ពុជា"

CHATBOT_SUBTITLE = "Cambodia Insurance Assistant"

WELCOME_GREETING = "សូមគោរពជម្រាបសួរ"

WELCOME_SUBLINE = "តើមានអ្វីខ្ញុំអាចជួយដល់លោក លោកស្រីដែរឬទេ?"

WELCOME_HINT = "សូមសួរអំពីច្បាប់ អនុក្រឹត្ត សេចក្តីប្រកាស និងសេចក្តីណែនាំ។"

IRC_LOGO_PATH = os.path.join(PROJECT_ROOT, "assets", "logo", "logo.png")

FAVICON_PATH = os.path.join(PROJECT_ROOT, "assets", "logo", "favicon.png")


def image_data_uri(path, mime_type):
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


IRC_LOGO_DATA_URI = image_data_uri(IRC_LOGO_PATH, "image/png")

IRC_LOGO_HTML = f'<img class="kc-logo" src="{IRC_LOGO_DATA_URI}" alt="IRC logo" />'
