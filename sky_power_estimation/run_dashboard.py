#!/usr/bin/env python3
"""
Launch the PV Power Estimation Dashboard.

Usage:
    python run_dashboard.py
    
Or directly with streamlit:
    streamlit run dashboard/app.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Streamlit dashboard."""
    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
    
    print("🌤️ Starting PV Power Estimation Dashboard...")
    print(f"📁 Dashboard path: {dashboard_path}")
    print()
    print("🌐 The dashboard will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the server.")
    print()
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.headless=true",
            "--browser.gatherUsageStats=false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped.")


if __name__ == "__main__":
    main()
