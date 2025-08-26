pip install -r requirements-server.txt
python run_server.py \
    --input-tcp tcp://localhost:8000 \
    --output-tcp tcp://localhost:8001 \
    --swapper-model models/inswapper_128.onnx \
    --enhancer-model models/GFPGANv1.4.pth \
    --source-image source.jpg 