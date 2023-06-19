import logging
import mimetypes
import os
import time

import openai
from approaches.chatreadretrieveread import ChatReadRetrieveReadApproach
from approaches.readdecomposeask import ReadDecomposeAsk
from approaches.readretrieveread import ReadRetrieveReadApproach
from approaches.retrievethenread import RetrieveThenReadApproach
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient
from flask import Flask, current_app, jsonify, request

# Replace these with your own values, either in environment variables or directly here
AZURE_STORAGE_ACCOUNT = os.environ.get("AZURE_STORAGE_ACCOUNT") or "mystorageaccount"
AZURE_STORAGE_CONTAINER = os.environ.get("AZURE_STORAGE_CONTAINER") or "content"
AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE") or "gptkb"
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX") or "gptkbindex"
AZURE_OPENAI_SERVICE = os.environ.get("AZURE_OPENAI_SERVICE") or "myopenai"
AZURE_OPENAI_GPT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_GPT_DEPLOYMENT") or "davinci"
AZURE_OPENAI_CHATGPT_DEPLOYMENT = (
    os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT") or "chat"
)

KB_FIELDS_CONTENT = os.environ.get("KB_FIELDS_CONTENT") or "content"
KB_FIELDS_CATEGORY = os.environ.get("KB_FIELDS_CATEGORY") or "category"
KB_FIELDS_SOURCEPAGE = os.environ.get("KB_FIELDS_SOURCEPAGE") or "sourcepage"

CONFIG_OPENAI_TOKEN = "openai_token"
CONFIG_CREDENTIAL = "azure_credential"
CONFIG_ASK_APPROACHES = "ask_approaches"
CONFIG_CHAT_APPROACHES = "chat_approaches"


def create_app():
    # Use the current user identity to authenticate with Azure OpenAI, Cognitive Search and Blob Storage (no secrets needed,
    # just use 'az login' locally, and managed identity when deployed on Azure). If you need to use keys, use separate AzureKeyCredential instances with the
    # keys for each service
    # If you encounter a blocking error during a DefaultAzureCredential resolution, you can exclude the problematic credential by using a parameter (ex. exclude_shared_token_cache_credential=True)
    azure_credential = DefaultAzureCredential()

    # Used by the OpenAI SDK
    openai.api_type = "azure"
    openai.api_base = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com"
    openai.api_version = "2022-12-01"

    # Comment these two lines out if using keys, set your API key in the OPENAI_API_KEY environment variable instead
    openai.api_type = "azure_ad"
    openai_token = azure_credential.get_token(
        "https://cognitiveservices.azure.com/.default"
    )
    openai.api_key = openai_token.token

    # Set up clients for Cognitive Search and Storage
    search_client = SearchClient(
        endpoint=f"https://{AZURE_SEARCH_SERVICE}.search.windows.net",
        index_name=AZURE_SEARCH_INDEX,
        credential=azure_credential,
    )
    blob_client = BlobServiceClient(
        account_url=f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net",
        credential=azure_credential,
    )
    blob_container = blob_client.get_container_client(AZURE_STORAGE_CONTAINER)

    app = Flask(__name__)
    app.config[CONFIG_OPENAI_TOKEN] = openai_token
    app.config[CONFIG_CREDENTIAL] = azure_credential

    # Various approaches to integrate GPT and external knowledge, most applications will use a single one of these patterns
    # or some derivative, here we include several for exploration purposes
    app.config[CONFIG_ASK_APPROACHES] = {
        "rtr": RetrieveThenReadApproach(
            search_client,
            AZURE_OPENAI_GPT_DEPLOYMENT,
            KB_FIELDS_SOURCEPAGE,
            KB_FIELDS_CONTENT,
        ),
        "rrr": ReadRetrieveReadApproach(
            search_client,
            AZURE_OPENAI_GPT_DEPLOYMENT,
            KB_FIELDS_SOURCEPAGE,
            KB_FIELDS_CONTENT,
        ),
        "rda": ReadDecomposeAsk(
            search_client,
            AZURE_OPENAI_GPT_DEPLOYMENT,
            KB_FIELDS_SOURCEPAGE,
            KB_FIELDS_CONTENT,
        ),
    }

    app.config[CONFIG_CHAT_APPROACHES] = {
        "rrr": ChatReadRetrieveReadApproach(
            search_client,
            AZURE_OPENAI_CHATGPT_DEPLOYMENT,
            AZURE_OPENAI_GPT_DEPLOYMENT,
            KB_FIELDS_SOURCEPAGE,
            KB_FIELDS_CONTENT,
        )
    }

    @app.route("/", defaults={"path": "index.html"})
    @app.route("/<path:path>")
    def static_file(path):
        return app.send_static_file(path)

    # Serve content files from blob storage from within the app to keep the example self-contained.
    # *** NOTE *** this assumes that the content files are public, or at least that all users of the app
    # can access all the files. This is also slow and memory hungry.
    @app.route("/content/<path>")
    def content_file(path):
        blob = blob_container.get_blob_client(path).download_blob()
        mime_type = blob.properties["content_settings"]["content_type"]
        if mime_type == "application/octet-stream":
            mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        return (
            blob.readall(),
            200,
            {
                "Content-Type": mime_type,
                "Content-Disposition": f"inline; filename={path}",
            },
        )

    @app.route("/ask", methods=["POST"])
    def ask():
        ensure_openai_token()
        approach = request.json["approach"]
        try:
            impl = app.config[CONFIG_ASK_APPROACHES].get(approach)
            if not impl:
                return jsonify({"error": "unknown approach"}), 400
            r = impl.run(request.json["question"], request.json.get("overrides") or {})
            return jsonify(r)
        except Exception as e:
            logging.exception("Exception in /ask")
            return jsonify({"error": str(e)}), 500

    @app.route("/chat", methods=["POST"])
    def chat():
        ensure_openai_token()
        approach = request.json["approach"]
        try:
            impl = app.config[CONFIG_CHAT_APPROACHES].get(approach)
            if not impl:
                return jsonify({"error": "unknown approach"}), 400
            r = impl.run(request.json["history"], request.json.get("overrides") or {})
            return jsonify(r)
        except Exception as e:
            logging.exception("Exception in /chat")
            return jsonify({"error": str(e)}), 500

    return app


def ensure_openai_token():
    openai_token = current_app.config[CONFIG_OPENAI_TOKEN]
    if openai_token.expires_on < int(time.time()) - 60:
        openai_token = current_app.config[CONFIG_CREDENTIAL].get_token(
            "https://cognitiveservices.azure.com/.default"
        )
        current_app.config[CONFIG_OPENAI_TOKEN] = openai_token
        openai.api_key = openai_token.token


if __name__ == "__main__":
    app = create_app()
    app.run()
