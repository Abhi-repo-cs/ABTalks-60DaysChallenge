# Monitoring Screenshot Checklist

Capture these screenshots after running the application:

1. Terminal showing application startup.
2. Browser/Postman showing `GET /health` → HTTP 200.
3. Browser/Postman showing a successful `POST /predict`.
4. Browser/Postman showing invalid input → HTTP 422.
5. Browser/Postman showing `GET /metrics`.
6. Terminal/editor showing `logs/production.log` with INFO/WARNING events.
7. Test terminal showing `pytest` passing.

These screenshots demonstrate both functional behavior and operational monitoring.
