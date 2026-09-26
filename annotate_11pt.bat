@echo off
cd /d "%~dp0"
python -m streamlit run tools/annotate_11pt.py --server.headless true --server.port 8504
