# AI Doctor — Disease Prediction and Recommendations

Lightweight Flask app that predicts diseases from symptoms using a pre-trained model and provides precautions, medications, and diet/workout suggestions.

## Quick start

1. Create a virtual environment and activate it:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add environment variables in a `.env` file (optional):

```
SECRET_KEY=your-secret
OPENROUTER_API_KEY=your-openrouter-key
PORT=10000
```

4. Place the trained model at `model/svc.pkl` and datasets under `dataset/` (these are required for predictions).

5. Run the app:

```bash
python main.py
```

Or for production with Gunicorn + eventlet:

```bash
gunicorn -k eventlet -w 1 main:app
```

6. Open http://127.0.0.1:10000

## Notes

- If model or datasets are missing, the app will log warnings and the prediction endpoint will return a friendly message.
- The app uses a local SQLite DB `users.db` in the project root to store users and predictions.

## Demo data

To quickly populate the app with a demo user and sample predictions run:

```bash
python scripts/seed_demo.py
```

This creates a demo user `demo@local` with password `demo` and inserts sample prediction history so the dashboard displays meaningful charts.

## Health check

There is a simple health endpoint at `/health` which returns JSON about model and dataset availability.

## Development

- Templates are in the `templates/` folder and static files are in `static/`.
- To reset the database, delete `users.db` and restart the app.

## License

This project is provided as-is.

## Refactor notes (for presentation)

- Project reorganized into `app/` package with `db.py`, `model.py`, and `routes.py` for clarity.
- Use `python scripts/train_model.py` to create a demo model and `model/svc_meta.json` with metadata.
- Configure secrets in `.env` (see `.env.example`). Debug is disabled by default; use only for development.


