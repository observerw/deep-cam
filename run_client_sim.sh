#!/bin/bash

# Deep-Cam 测试客户端脚本
# 用于启动测试用的 TCP 流，模拟摄像头输入

echo "Deep-Cam 测试客户端启动中..."

# 安装依赖
echo "安装客户端依赖..."
pip install -r requirements-client.txt

# 检查是否存在测试视频文件
TEST_VIDEO="data/test_video.mp4"

if [ ! -f "$TEST_VIDEO" ]; then
    echo "未找到测试视频文件，正在创建测试视频..."
    
    # 创建 data 目录（如果不存在）
    mkdir -p data
    
    # 使用 FFmpeg 创建一个简单的测试视频
    # 创建一个 10 秒的彩色条纹测试视频（将会循环播放）
    ffmpeg -f lavfi -i testsrc=duration=10:size=640x480:rate=30 \
           -c:v libx264 -preset fast -pix_fmt yuv420p \
           "$TEST_VIDEO" -y
    
    if [ $? -eq 0 ]; then
        echo "✓ 测试视频创建成功: $TEST_VIDEO"
    else
        echo "✗ 测试视频创建失败，请检查 FFmpeg 是否已安装"
        echo "安装 FFmpeg: sudo apt install ffmpeg (Ubuntu/Debian) 或 brew install ffmpeg (macOS)"
        exit 1
    fi
else
    echo "✓ 使用现有测试视频: $TEST_VIDEO"
fi

# 启动测试客户端，使用 FFmpeg 直接实现循环播放
echo "启动测试 TCP 流（无限循环播放）..."
echo "提示：按 Ctrl+C 停止测试流"
echo "TCP 流地址: tcp://0.0.0.0:8000"
echo ""

# 使用 FFmpeg 直接创建循环播放的 TCP 流
ffmpeg -stream_loop -1 -re -i "$TEST_VIDEO" \
       -c:v libx264 -preset ultrafast -tune zerolatency \
       -b:v 2000k -maxrate 2000k -bufsize 4000k \
       -g 30 -profile:v baseline -pix_fmt yuv420p \
       -f mpegts tcp://0.0.0.0:8000?listen=1

echo "测试客户端已停止"