import sys
from pathlib import Path

# Add the dashboard directory to sys.path so relative imports (if any) work
dashboard_path = Path(__file__).parent / "dashboard"
sys.path.append(str(dashboard_path))

# Import and run the dashboard app
with open(dashboard_path / "app.py", "r") as f:
    code = f.read()
    exec(code)
