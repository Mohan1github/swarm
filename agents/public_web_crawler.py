from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urljoin, urlparse


def public_addresses(value: object) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    if not isinstance(value, str) or len(value) > 2048:
        return []
    try:
        parsed = urlparse(value)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in (None, 443)
            or parsed.hostname.casefold() == "localhost"
        ):
            return []
        try:
            addresses = [ipaddress.ip_address(parsed.hostname)]
        except ValueError:
            addresses = [
                ipaddress.ip_address(result[4][0])
                for result in socket.getaddrinfo(
                    parsed.hostname,
                    443,
                    type=socket.SOCK_STREAM,
                )
            ]
        if not addresses or any(not address.is_global for address in addresses):
            return []
        return list(dict.fromkeys(addresses))
    except (OSError, ValueError):
        return []


def is_public_web_url(value: object) -> bool:
    return bool(public_addresses(value))


def fetch_public_url(
    url: str,
    max_bytes: int,
    timeout: float = 6,
    max_redirects: int = 3,
) -> tuple[str, str, bytes]:
    try:
        import httpx
    except ImportError as error:
        raise RuntimeError("Public web crawling requires the httpx package.") from error

    current_url = url
    headers = {"User-Agent": "SwarmPublicWebCrawler/1.0 (+public source discovery)"}
    with httpx.Client(timeout=timeout, follow_redirects=False, headers=headers) as client:
        for _ in range(max_redirects):
            addresses = public_addresses(current_url)
            if not addresses:
                raise ValueError("Crawler refused a non-public or unsupported URL.")
            parsed = urlparse(current_url)
            hostname = parsed.hostname.encode("idna").decode("ascii")
            address = str(addresses[0])
            if ":" in address:
                address = f"[{address}]"
            pinned_url = f"https://{address}{parsed.path or '/'}"
            if parsed.query:
                pinned_url += f"?{parsed.query}"
            request = client.build_request(
                "GET",
                pinned_url,
                headers={"Host": hostname},
                extensions={"sni_hostname": hostname.encode("ascii")},
            )
            response = client.send(request, stream=True, follow_redirects=False)
            try:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise RuntimeError(
                            "Crawler received a redirect without a destination."
                        )
                    current_url = urljoin(current_url, location)
                    continue
                response.raise_for_status()
                body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) > max_bytes:
                        raise RuntimeError("Crawler response exceeded the size limit.")
                return (
                    str(response.headers.get("content-type", "")).casefold(),
                    response.encoding or "utf-8",
                    bytes(body),
                )
            finally:
                response.close()
    raise RuntimeError("Crawler exceeded the redirect limit.")
