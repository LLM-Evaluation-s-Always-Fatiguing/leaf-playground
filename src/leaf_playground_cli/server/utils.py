import socket


def get_local_ip() -> str:
    try:
        # Create a temporary socket
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # Connect to a public DNS server (Google's)
            s.connect(("8.8.8.8", 80))
            # Get the local IP address from the socket
            local_ip = s.getsockname()[0]
            return local_ip
    except Exception:
        return "127.0.0.1"


__all__ = ["get_local_ip"]
