import multiprocessing

bind = "0.0.0.0:8000"
workers = 4
preload_app = False
wsgi_app = "api.main:app"
worker_class = "uvicorn.workers.UvicornWorker"
loglevel = "info"
accesslog = "-"
errorlog = "-"
timeout = 120
graceful_timeout = 30
keepalive = 5


