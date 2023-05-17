python3 /app/ml_service.py register-connector
python3 /app/ml_service.py reset-output-db
python3 /app/ml_service.py prepare-ml-service
celery multi start worker1 \
    --app=ml_celery \
    --loglevel=INFO \
    --concurrency=4 -n worker1@%h \
    --pidfile=/var/run/celery/%n.pid \
    --logfile=/var/log/celery/%n%I.log
python3 /app/ml_service.py run-ml-service &
streamlit run /app/app.py --server.port=8501 --server.address=0.0.0.0 --browser.gatherUsageStats=false --server.allowRunOnSave 1 --server.runOnSave 1
