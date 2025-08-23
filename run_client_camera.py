#!/usr/bin/env python3

import subprocess
import sys
import threading
import time
from typing import Optional, Tuple

import cv2

from deep_cam.utils.camera import get_available_cameras


def select_camera() -> int:
    camera_indices, camera_names = get_available_cameras()

    if not camera_indices:
        print("错误: 未找到可用的摄像头")
        sys.exit(1)

    print("\n可用摄像头:")
    for i, (idx, name) in enumerate(zip(camera_indices, camera_names)):
        print(f"{i + 1}. {name} (索引: {idx})")

    while True:
        try:
            choice = input(f"\n请选择摄像头 (1-{len(camera_indices)}): ")
            choice_idx = int(choice) - 1

            if 0 <= choice_idx < len(camera_indices):
                selected_camera = camera_indices[choice_idx]
                selected_name = camera_names[choice_idx]
                print(f"已选择: {selected_name} (索引: {selected_camera})")
                return selected_camera
            else:
                print(f"请输入 1 到 {len(camera_indices)} 之间的数字")
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n已取消")
            sys.exit(0)


def check_gstreamer() -> bool:
    """检查GStreamer是否可用"""
    try:
        result = subprocess.run(
            ["gst-launch-1.0", "--version"], capture_output=True, text=True, timeout=5
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


def start_rtsp_viewer(rtsp_url: str, window_name: str = "RTSP Stream Viewer") -> None:
    """使用OpenCV读取RTSP流并在窗口中显示

    Args:
        rtsp_url: RTSP流地址
        window_name: 窗口名称
    """
    print(f"正在连接RTSP流: {rtsp_url}")

    # 创建VideoCapture对象
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print(f"错误: 无法连接到RTSP流 {rtsp_url}")
        return

    print("成功连接到RTSP流，按 'q' 键退出查看器")

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
                print("用户退出RTSP查看器")
                break

    except KeyboardInterrupt:
        print("\n用户中断，正在关闭RTSP查看器...")
    except Exception as e:
        print(f"RTSP查看器出错: {e}")
    finally:
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
        print("RTSP查看器已关闭")


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
        ssh_cmd.extend(["-L", f"{local_port}:localhost:{remote_port}"])
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


def start_tcp_stream(
    camera_index: int,
    tcp_port: int = 5000,
    resolution: str = "640x480",
    fps: int = 30,
    ssh_process: Optional[subprocess.Popen] = None,
) -> None:
    """启动TCP视频流

    Args:
        camera_index: 摄像头索引
        tcp_port: TCP端口号
        resolution: 视频分辨率
        fps: 帧率
        ssh_process: SSH隧道进程，用于清理资源
    """
    # 构建GStreamer命令 - 低延迟模式
    if sys.platform == "darwin":  # macOS
        gst_cmd = [
            "gst-launch-1.0",
            "avfvideosrc",
            f"device-index={camera_index}",
            "!",
            f"video/x-raw,width={resolution.split('x')[0]},height={resolution.split('x')[1]},framerate={fps}/1",
            "!",
            "videoconvert",
            "!",
            "x264enc",
            "tune=zerolatency",  # 零延迟调优
            "speed-preset=ultrafast",  # 最快编码速度
            "bitrate=2000",  # 2Mbps码率
            "key-int-max=30",  # 关键帧间隔
            "!",
            "video/x-h264,profile=baseline",  # 基线配置文件，兼容性好
            "!",
            "h264parse",
            "!",
            "mpegtsmux",
            "!",
            "tcpserversink",
            "host=127.0.0.1",
            f"port={tcp_port}",
            "sync=false",  # 禁用同步以减少延迟
        ]
    else:  # Linux
        gst_cmd = [
            "gst-launch-1.0",
            "v4l2src",
            f"device=/dev/video{camera_index}",
            "!",
            f"video/x-raw,width={resolution.split('x')[0]},height={resolution.split('x')[1]},framerate={fps}/1",
            "!",
            "videoconvert",
            "!",
            "x264enc",
            "tune=zerolatency",
            "speed-preset=ultrafast",
            "bitrate=2000",
            "key-int-max=30",
            "!",
            "video/x-h264,profile=baseline",
            "!",
            "h264parse",
            "!",
            "mpegtsmux",
            "!",
            "tcpserversink",
            "host=127.0.0.1",
            f"port={tcp_port}",
            "sync=false",
        ]

    print("\n启动TCP视频流...")
    print(f"摄像头索引: {camera_index}")
    print(f"分辨率: {resolution}")
    print(f"帧率: {fps} fps")
    print(f"TCP地址: tcp://127.0.0.1:{tcp_port}")

    if ssh_process:
        print("\n✓ SSH隧道已建立，视频流将通过SSH转发")

    print(f"\n执行命令: {' '.join(gst_cmd)}")
    print("\n按 Ctrl+C 停止流...\n")

    try:
        print("提示: 可以使用以下命令查看TCP流:")
        print(
            f"gst-launch-1.0 tcpclientsrc host=127.0.0.1 port={tcp_port} ! decodebin ! videoconvert ! autovideosink"
        )
        print(f"或者: vlc tcp://127.0.0.1:{tcp_port}")

        if ssh_process:
            print("\n注意: 如果配置了SSH端口转发，请使用相应的转发端口访问视频流")

        print("\n正在启动GStreamer...")

        # 启动GStreamer进程
        result = subprocess.run(
            gst_cmd,
            capture_output=True,
            text=True,
            check=False,  # 不自动抛出异常，手动处理返回码
            timeout=300,  # 5分钟超时
        )

        # 检查执行结果
        if result.returncode != 0:
            print(f"\nGStreamer执行失败，返回码: {result.returncode}")
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
        else:
            print("\n视频流传输完成")
            if result.stdout:
                print(f"输出信息: {result.stdout}")

    except subprocess.TimeoutExpired:
        print("\nGStreamer运行超时 (5分钟)，可能是正常的长时间流传输")
        print("如果需要继续运行，请重新启动脚本")
    except KeyboardInterrupt:
        print("\n用户中断，正在停止视频流...")
        print("视频流已停止")
    except Exception as e:
        print(f"\n未预期的错误: {e}")
        print("请检查GStreamer是否正确安装和配置")
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


def main():
    """主函数"""
    print("Deep-Cam TCP 视频流客户端")
    print("=" * 30)

    # 检查GStreamer
    if not check_gstreamer():
        print("错误: 未找到GStreamer，请先安装GStreamer")
        print(
            "macOS: brew install gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly"
        )
        print(
            "Ubuntu/Debian: sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly"
        )
        sys.exit(1)

    print("✓ GStreamer 已安装")

    # 选择摄像头
    camera_index = select_camera()

    # 获取用户配置
    print("\n配置视频流参数:")

    # SSH配置
    print("\nSSH端口转发配置 (可选):")
    ssh_host = input("SSH主机地址 (留空跳过SSH): ").strip()
    ssh_port = 22
    push_port_mapping = None
    pull_port_mapping = None
    ssh_process = None

    if ssh_host:
        # SSH端口
        while True:
            try:
                ssh_port_input = input("SSH端口 (默认: 22): ").strip()
                ssh_port = int(ssh_port_input) if ssh_port_input else 22
                if 1 <= ssh_port <= 65535:
                    break
                else:
                    print("端口号应在 1-65535 范围内")
            except ValueError:
                print("请输入有效的端口号")

        # 推流端口映射
        while True:
            push_input = input(
                "推流端口映射 (格式: local:remote, 如 8080:8080, 留空跳过): "
            ).strip()
            if not push_input:
                break
            try:
                parse_port_mapping(push_input)  # 验证格式
                push_port_mapping = push_input
                break
            except ValueError as e:
                print(f"错误: {e}")

        # 拉流端口映射
        while True:
            pull_input = input(
                "拉流端口映射 (格式: local:remote, 如 8081:8081, 留空跳过): "
            ).strip()
            if not pull_input:
                break
            try:
                parse_port_mapping(pull_input)  # 验证格式
                pull_port_mapping = pull_input
                break
            except ValueError as e:
                print(f"错误: {e}")

    # TCP端口
    while True:
        try:
            port_input = input("\nTCP端口 (默认: 5000): ").strip()
            tcp_port = int(port_input) if port_input else 5000
            if 1024 <= tcp_port <= 65535:
                break
            else:
                print("端口号应在 1024-65535 范围内")
        except ValueError:
            print("请输入有效的端口号")

    # 分辨率
    resolution_options = ["640x480", "1280x720", "1920x1080"]
    print("\n可用分辨率:")
    for i, res in enumerate(resolution_options, 1):
        print(f"{i}. {res}")

    while True:
        try:
            res_input = input("选择分辨率 (默认: 1): ").strip()
            res_choice = int(res_input) if res_input else 1
            if 1 <= res_choice <= len(resolution_options):
                resolution = resolution_options[res_choice - 1]
                break
            else:
                print(f"请输入 1 到 {len(resolution_options)} 之间的数字")
        except ValueError:
            print("请输入有效的数字")

    # 帧率
    while True:
        try:
            fps_input = input("帧率 (默认: 30): ").strip()
            fps = int(fps_input) if fps_input else 30
            if 1 <= fps <= 60:
                break
            else:
                print("帧率应在 1-60 范围内")
        except ValueError:
            print("请输入有效的帧率")

    # 建立SSH隧道 (如果需要)
    if ssh_host and (push_port_mapping or pull_port_mapping):
        print("\n正在建立SSH隧道...")
        ssh_process = setup_ssh_tunnel(
            ssh_host, ssh_port, push_port_mapping, pull_port_mapping
        )
        if not ssh_process:
            print("SSH隧道建立失败，继续使用本地连接")

    # 询问是否启动RTSP查看器
    start_viewer = False
    if pull_port_mapping:
        while True:
            viewer_input = (
                input("\n是否启动RTSP流查看器? (y/n, 默认: n): ").strip().lower()
            )
            if viewer_input in ["", "n", "no"]:
                break
            elif viewer_input in ["y", "yes"]:
                start_viewer = True
                break
            else:
                print("请输入 y 或 n")

    # 如果需要启动查看器，在单独线程中运行
    viewer_thread = None
    if start_viewer and pull_port_mapping:
        local_port, _ = parse_port_mapping(pull_port_mapping)
        rtsp_url = f"rtsp://127.0.0.1:{local_port}/stream"
        print(f"\n将在单独窗口中显示RTSP流: {rtsp_url}")

        # 在单独线程中启动RTSP查看器
        viewer_thread = threading.Thread(
            target=start_rtsp_viewer,
            args=(rtsp_url, "Deep-Cam RTSP Viewer"),
            daemon=True,
        )
        viewer_thread.start()

    # 启动视频流
    try:
        start_tcp_stream(camera_index, tcp_port, resolution, fps, ssh_process)
    finally:
        # 如果RTSP查看器在运行，等待其结束
        if viewer_thread and viewer_thread.is_alive():
            print("\n等待RTSP查看器关闭...")
            # 注意：由于OpenCV窗口在主线程中，这里不需要特殊处理


if __name__ == "__main__":
    main()
