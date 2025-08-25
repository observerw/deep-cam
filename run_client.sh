pip install -r requirements-client.txt
python run_client.py \
    --camera 0 \
    --port 8000 \
    --ssh-host root@connect.westc.gpuhub.com \
    --ssh-port 27857 \
    --push-port 8000:8000 \
    --pull-port 8001:8001