#!/bin/bash
cd "$(dirname "$0")"
# This bypasses the email prompt and runs your app
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py --server.headless true