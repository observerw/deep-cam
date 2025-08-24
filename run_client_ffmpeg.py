#!/usr/bin/env python3

import argparse
import subprocess
import sys
import threading
import time
from typing import Optional, Tuple

import cv2

from deep_cam.utils.camera import get_available_cameras


def select_camera(camera_index: Optional[int] = None) -> int:
    """选择摄像头

    Args:
        camera_index: 指定的摄像头索引，如果为None则使用第一个可用摄像头

    Returns:
        选中的摄像头索引
    """
    camera_indices, camera_names = get_available_cameras()

    if not camera_indices:
        print("错误: 未找到可用的摄像头")
        sys.exit(1)

    print("\n可用摄像头:")
    for i, (idx, name) in enumerate(zip(camera_indices, camera_names)):
        print(f"{i + 1}. {name} (索引: {idx})")

    # 如果指定了摄像头索引，验证并使用
    if camera_index is not None:
        if camera_index in camera_indices:
            selected_name = camera_names[camera_indices.index(camera_index)]
            print(f"已选择: {selected_name} (索引: {camera_index})")
            return camera_index
        else:
            print(f"错误: 摄像头索引 {camera_index} 不可用")
            print(f"可用索引: {camera_indices}")
            sys.exit(1)

    # 如果没有指定，使用第一个可用摄像头
    selected_camera = camera_indices[0]
    selected_name = camera_names[0]
    print(f"自动选择: {selected_name} (索引: {selected_camera})")
    return selected_camera


def check_ffmpeg() -> bool:
    """检查FFmpeg是否可用"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def parse_port_mapping(port_mapping: str) -> Tuple[int, int]:
    """解析端口映射字符串，格式如 '8080:8080'

    Args:
        port_mapping: 端口映射字符串，格式为 'local_port:remote_port'

    Returns:
        Tuple[int, int]: (本地端口, 远程端口)

    Raises:
        ValueError: 端口映射格式错误
    """
    try:
        local_port_str, remote_port_str = port_mapping.split(":")
        local_port = int(local_port_str.strip())
        remote_port = int(remote_port_str.strip())

        if not (1024 <= local_port <= 65535) or not (1024 <= remote_port <= 65535):
            raise ValueError("端口号应在 1024-65535 范围内")

        return local_port, remote_port
    except ValueError as e:
        if "端口号应在" in str(e):
            raise e
        raise ValueError(
            "端口映射格式错误，应为 'local_port:remote_port'，如 '8080:8080'"
        )


def start_tcp_viewer(tcp_url: str, window_name: str = "TCP Stream Viewer") -> None:
    """使用OpenCV读取TCP流并在窗口中显示

    Args:
        tcp_url: TCP流地址
        window_name: 窗口名称
    """
    print(f"正在连接TCP流: {tcp_url}")

    # 创建VideoCapture对象
    cap = cv2.VideoCapture(tcp_url)

    if not cap.isOpened():
        print(f"错误: 无法连接到TCP流 {tcp_url}")
        return

    print("成功连接到TCP流，按 'q' 键退出查看器")

    try:
        while True:
            # 读取帧
            ret, frame = cap.read()

            if not ret:
                print("警告: 无法读取帧，可能是流中断")
                break

            # 显示帧
            cv2.imshow(window_name, frame)

            # 检查按键，按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("用户退出TCP查看器")
                break

    except KeyboardInterrupt:
        print("\n用户中断，正在关闭TCP查看器...")
    except Exception as e:
        print(f"TCP查看器出错: {e}")
    finally:
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
        print("TCP查看器已关闭")


def setup_ssh_tunnel(
    ssh_host: str,
    ssh_port: int,
    push_port_mapping: Optional[str] = None,
    pull_port_mapping: Optional[str] = None,
) -> Optional[subprocess.Popen]:
    """设置SSH端口转发隧道

    Args:
        ssh_host: SSH主机地址
        ssh_port: SSH端口
        push_port_mapping: 推流端口映射，格式如 '8080:8080'
        pull_port_mapping: 拉流端口映射，格式如 '8081:8081'

    Returns:
        SSH进程对象，如果不需要SSH隧道则返回None
    """
    if not push_port_mapping and not pull_port_mapping:
        return None

    ssh_cmd = ["ssh", "-N", "-p", str(ssh_port)]

    # 添加推流端口转发 (本地端口转发到远程)
    if push_port_mapping:
        local_port, remote_port = parse_port_mapping(push_port_mapping)
        ssh_cmd.extend(["-R", f"{local_port}:localhost:{remote_port}"])
        print(f"设置推流端口转发: 本地:{local_port} -> 远程:{remote_port}")

    # 添加拉流端口转发 (远程端口转发到本地)
    if pull_port_mapping:
        local_port, remote_port = parse_port_mapping(pull_port_mapping)
        ssh_cmd.extend(["-R", f"{remote_port}:localhost:{local_port}"])
        print(f"设置拉流端口转发: 远程:{remote_port} -> 本地:{local_port}")

    ssh_cmd.append(ssh_host)

    print(f"启动SSH隧道: {' '.join(ssh_cmd)}")

    try:
        ssh_process = subprocess.Popen(
            ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # 等待一下确保SSH连接建立
        time.sleep(2)

        # 检查SSH进程是否还在运行
        if ssh_process.poll() is not None:
            stdout, stderr = ssh_process.communicate()
            print(f"SSH隧道启动失败: {stderr}")
            return None

        print("SSH隧道已建立")
        return ssh_process

    except Exception as e:
        print(f"启动SSH隧道时出错: {e}")
        return None


def start_ffmpeg_stream(
    camera_index: int,
    tcp_port: int = 8554,
    resolution: str = "640x480",
    fps: int = 30,
    ssh_process: Optional[subprocess.Popen] = None,
) -> None:
    """使用FFmpeg启动TCP服务器和视频流

    Args:
        camera_index: 摄像头索引
        tcp_port: TCP端口号
        resolution: 视频分辨率
        fps: 帧率
        ssh_process: SSH隧道进程，用于清理资源
    """
    width, height = resolution.split("x")

    # 构建FFmpeg命令 - 使用TCP输出
    if sys.platform == "darwin":  # macOS
        ffmpeg_cmd = [
            "ffmpeg",
            "-f",
            "avfoundation",
            "-video_device_index",
            str(camera_index),
            "-framerate",
            str(fps),
            "-video_size",
            resolution,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",  # 最快编码速度
            "-tune",
            "zerolatency",  # 零延迟调优
            "-b:v",
            "2000k",  # 2Mbps码率
            "-maxrate",
            "2000k",
            "-bufsize",
            "4000k",
            "-g",
            "30",  # 关键帧间隔
            "-profile:v",
            "baseline",  # 基线配置文件，兼容性好
            "-pix_fmt",
            "yuv420p",
            "-f",
            "mpegts",
            f"tcp://127.0.0.1:{tcp_port}?listen=1",
        ]
    else:  # Linux
        ffmpeg_cmd = [
            "ffmpeg",
            "-f",
            "v4l2",
            "-i",
            f"/dev/video{camera_index}",
            "-framerate",
            str(fps),
            "-video_size",
            resolution,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-b:v",
            "2000k",
            "-maxrate",
            "2000k",
            "-bufsize",
            "4000k",
            "-g",
            "30",
            "-profile:v",
            "baseline",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "mpegts",
            f"tcp://127.0.0.1:{tcp_port}?listen=1",
        ]

    print("\n启动FFmpeg视频流...")
    print(f"摄像头索引: {camera_index}")
    print(f"分辨率: {resolution}")
    print(f"帧率: {fps} fps")
    print(f"TCP地址: tcp://127.0.0.1:{tcp_port}")

    if ssh_process:
        print("\n✓ SSH隧道已建立，视频流将通过SSH转发")

    print(f"\n执行命令: {' '.join(ffmpeg_cmd)}")
    print("\n按 Ctrl+C 停止流...\n")

    try:
        print("提示: 可以使用以下命令查看TCP流:")
        print(f"ffplay tcp://127.0.0.1:{tcp_port}")
        print(f"或者: vlc tcp://127.0.0.1:{tcp_port}")

        if ssh_process:
            print("\n注意: 如果配置了SSH端口转发，请使用相应的转发端口访问视频流")

        print("\n正在启动FFmpeg...")

        # 启动FFmpeg进程
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            check=False,  # 不自动抛出异常，手动处理返回码
            timeout=300,  # 5分钟超时
        )

        # 检查执行结果
        if result.returncode != 0:
            print(f"\nFFmpeg执行失败，返回码: {result.returncode}")
            if result.stderr:
                print("\n错误详情:")
                print(result.stderr)

            # 提供常见问题的解决建议
            if (
                "Permission denied" in result.stderr
                or "Operation not permitted" in result.stderr
            ):
                print("\n可能的解决方案:")
                print("1. 检查摄像头权限设置 (系统偏好设置 > 安全性与隐私 > 摄像头)")
                print("2. 确保没有其他应用正在使用摄像头")
            elif (
                "Input/output error" in result.stderr
                or "Could not open" in result.stderr
            ):
                print("\n可能的解决方案:")
                print("1. 尝试不同的摄像头索引")
                print("2. 检查摄像头是否正常工作")
                print("3. 尝试降低分辨率或帧率")
            elif "Connection refused" in result.stderr:
                print("\n可能的解决方案:")
                print("1. 检查RTSP端口是否被占用")
                print("2. 尝试使用不同的端口号")
        else:
            print("\n视频流传输完成")
            if result.stdout:
                print(f"输出信息: {result.stdout}")

    except subprocess.TimeoutExpired:
        print("\nFFmpeg运行超时 (5分钟)，可能是正常的长时间流传输")
        print("如果需要继续运行，请重新启动脚本")
    except KeyboardInterrupt:
        print("\n用户中断，正在停止视频流...")
        print("视频流已停止")
    except Exception as e:
        print(f"\n未预期的错误: {e}")
        print("请检查FFmpeg和MediaMTX是否正确安装和配置")
    finally:
        # 清理SSH隧道
        if ssh_process and ssh_process.poll() is None:
            print("正在关闭SSH隧道...")
            ssh_process.terminate()
            try:
                ssh_process.wait(timeout=5)
                print("SSH隧道已关闭")
            except subprocess.TimeoutExpired:
                print("强制关闭SSH隧道")
                ssh_process.kill()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Deep-Cam FFmpeg TCP 视频流客户端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  %(prog)s --camera 0 --port 8554 --resolution 1280x720 --fps 30
  %(prog)s --camera 1 --ssh-host example.com --push-port 8080:8080 --pull-port 8081:8081
  %(prog)s --list-cameras  # 列出可用摄像头""",
    )

    # 摄像头相关参数
    parser.add_argument(
        "--camera",
        "-c",
        type=int,
        help="摄像头索引 (如果不指定，将自动选择第一个可用摄像头)",
    )
    parser.add_argument(
        "--list-cameras", action="store_true", help="列出所有可用摄像头并退出"
    )

    # 视频流参数
    parser.add_argument(
        "--port", "-p", type=int, default=8554, help="TCP端口号 (默认: 8554)"
    )
    parser.add_argument(
        "--resolution",
        "-r",
        default="640x480",
        choices=["640x480", "1280x720", "1920x1080"],
        help="视频分辨率 (默认: 640x480)",
    )
    parser.add_argument("--fps", "-f", type=int, default=30, help="帧率 (默认: 30)")

    # SSH隧道参数
    parser.add_argument("--ssh-host", help="SSH主机地址")
    parser.add_argument("--ssh-port", type=int, default=22, help="SSH端口 (默认: 22)")
    parser.add_argument(
        "--push-port", help="推流端口映射，格式: local:remote (如: 8080:8080)"
    )
    parser.add_argument(
        "--pull-port", help="拉流端口映射，格式: local:remote (如: 8081:8081)"
    )

    # TCP查看器
    parser.add_argument(
        "--viewer", action="store_true", help="启动TCP流查看器 (需要配置拉流端口映射)"
    )

    return parser.parse_args()


def validate_args(args):
    """验证命令行参数"""
    # 验证TCP端口
    if not (1024 <= args.port <= 65535):
        print(f"错误: TCP端口 {args.port} 应在 1024-65535 范围内")
        sys.exit(1)

    # 验证帧率
    if not (1 <= args.fps <= 60):
        print(f"错误: 帧率 {args.fps} 应在 1-60 范围内")
        sys.exit(1)

    # 验证SSH端口
    if not (1 <= args.ssh_port <= 65535):
        print(f"错误: SSH端口 {args.ssh_port} 应在 1-65535 范围内")
        sys.exit(1)

    # 验证端口映射格式
    if args.push_port:
        try:
            parse_port_mapping(args.push_port)
        except ValueError as e:
            print(f"错误: 推流端口映射格式错误 - {e}")
            sys.exit(1)

    if args.pull_port:
        try:
            parse_port_mapping(args.pull_port)
        except ValueError as e:
            print(f"错误: 拉流端口映射格式错误 - {e}")
            sys.exit(1)

    # 验证TCP查看器配置
    if args.viewer and not args.pull_port:
        print("错误: 启动TCP查看器需要配置拉流端口映射 (--pull-port)")
        sys.exit(1)


def list_cameras():
    """列出所有可用摄像头"""
    camera_indices, camera_names = get_available_cameras()

    if not camera_indices:
        print("未找到可用的摄像头")
        return

    print("可用摄像头:")
    for idx, name in zip(camera_indices, camera_names):
        print(f"  索引 {idx}: {name}")


def main():
    """主函数"""
    args = parse_args()

    # 如果只是列出摄像头，执行后退出
    if args.list_cameras:
        list_cameras()
        return

    # 验证参数
    validate_args(args)

    print("Deep-Cam FFmpeg TCP 视频流客户端")
    print("=" * 34)

    # 检查FFmpeg
    if not check_ffmpeg():
        print("错误: 未找到FFmpeg，请先安装FFmpeg")
        print("macOS: brew install ffmpeg")
        print("Ubuntu/Debian: sudo apt install ffmpeg")
        sys.exit(1)

    print("✓ FFmpeg 已安装")

    print("✓ 使用FFmpeg TCP输出")

    # 选择摄像头
    camera_index = select_camera(args.camera)

    # 显示配置信息
    print("\n配置信息:")
    print(f"  摄像头索引: {camera_index}")
    print(f"  TCP端口: {args.port}")
    print(f"  分辨率: {args.resolution}")
    print(f"  帧率: {args.fps} fps")

    if args.ssh_host:
        print(f"  SSH主机: {args.ssh_host}:{args.ssh_port}")
        if args.push_port:
            print(f"  推流端口映射: {args.push_port}")
        if args.pull_port:
            print(f"  拉流端口映射: {args.pull_port}")

    if args.viewer:
        print("  TCP查看器: 启用")

    # 建立SSH隧道 (如果需要)
    ssh_process = None
    if args.ssh_host and (args.push_port or args.pull_port):
        print("\n正在建立SSH隧道...")
        ssh_process = setup_ssh_tunnel(
            args.ssh_host, args.ssh_port, args.push_port, args.pull_port
        )
        if not ssh_process:
            print("SSH隧道建立失败，继续使用本地连接")

    # 启动TCP查看器 (如果需要)
    viewer_thread = None
    if args.viewer and args.pull_port:
        local_port, _ = parse_port_mapping(args.pull_port)
        tcp_url = f"tcp://127.0.0.1:{local_port}"
        print(f"\n将在单独窗口中显示TCP流: {tcp_url}")

        # 在单独线程中启动TCP查看器
        viewer_thread = threading.Thread(
            target=start_tcp_viewer,
            args=(tcp_url, "Deep-Cam FFmpeg TCP Viewer"),
            daemon=True,
        )
        viewer_thread.start()

    # 启动视频流
    try:
        start_ffmpeg_stream(
            camera_index,
            args.port,
            args.resolution,
            args.fps,
            ssh_process,
        )
    finally:
        # 如果TCP查看器在运行，等待其结束
        if viewer_thread and viewer_thread.is_alive():
            print("\n等待TCP查看器关闭...")
            # 注意：由于OpenCV窗口在主线程中，这里不需要特殊处理


if __name__ == "__main__":
    main()
