"""Read one v3 request body in adapter tests."""

from dataclasses import dataclass

from hpke_http import Request, Response, Server, StreamResponseRight


@dataclass(slots=True)
class BufferedStreamRequest:
    request: Request
    response_right: StreamResponseRight

    def protect_response(self, response: Response) -> bytes:
        return self.response_right.protect_response(response)


def open_stream_request(server: Server, raw: bytes, psk: bytes) -> BufferedStreamRequest:
    first_length = server.stream_start_length(raw)
    assert first_length is not None
    opened_stream = server.preparse_stream(raw[:first_length]).authenticate(psk).admit(accepted=True)
    parts: list[bytes] = []
    offset = first_length
    while offset < len(raw):
        used, record = opened_stream.feed(raw, offset)
        assert used > 0
        offset += used
        if record is not None and record[0] == "data":
            parts.append(record[1])
    response_right = opened_stream.finish_eof()
    request = Request(
        opened_stream.head.method,
        opened_stream.head.authority,
        opened_stream.head.path,
        opened_stream.head.headers,
        b"".join(parts),
    )
    return BufferedStreamRequest(request, response_right)
