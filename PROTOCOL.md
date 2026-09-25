# hpke-http protocol

This file specifies the hpke-http/3 request and response wire format, the
HHKD v2 key record, and their HTTPS GET/POST exchange. It describes the format
used by the Rust, Python, and TypeScript
[v4.0.0 packages](https://github.com/dualeai/hpke-http/releases/tag/v4.0.0).
This file was added after that release tag; its bytes were checked against
the tag.
Package versions, the hpke-http/3 wire version, the HHKD v2 record version,
and the private binding ABI 8 are separate values.

The Rust engine owns the protected bytes. The Python and TypeScript bindings
use that engine. Their HTTP adapters add runtime policy, such as timeouts,
request body handling, and browser CORS. This file marks those choices where
they affect an exchange.

Use [RFC 9180](https://www.rfc-editor.org/rfc/rfc9180.html) for HPKE,
[RFC 9000, Section 16](https://www.rfc-editor.org/rfc/rfc9000.html#section-16)
for QUIC integers, [RFC 8878](https://www.rfc-editor.org/rfc/rfc8878.html)
for zstd frames, and [RFC 9110](https://www.rfc-editor.org/rfc/rfc9110.html)
for HTTP terms. The field vectors below use QUIC integer lengths, but the
whole protected message is **not** an
[RFC 9292 Binary HTTP message](https://www.rfc-editor.org/rfc/rfc9292.html).
It has no Binary HTTP framing indicator, scheme vector, or trailers.
It is also **not** an
[RFC 9458 Oblivious HTTP message](https://www.rfc-editor.org/rfc/rfc9458.html).

## Exchange at a glance

The configured outer endpoint is one full HTTPS URL with no user
information, query, or fragment, for example
https://api.example.test/protected. The logical target can be a different
path on the same origin. A gateway can use a different logical origin only
when the client and server set that origin explicitly. A pinned public key
skips the GET.

~~~text
CLIENT
  key = pinned key, or GET and check one HHKD v2 key record
  make public request header with a fresh issue time
  (enc, request_context) = HPKE SetupPSKS(public_key, info, psk, psk_id)
  response_secret = HPKE Export(request_context, response_label, 32)
  start = header || enc || seal(START)
  check that the key lease still permits POST START
  POST start || seal(each DATA) || seal(END)
  end the outer POST body
  check the outer 200 response and open its protected START
  for a finite reply, hold DATA until END and true outer body EOF
  for SSE, release each checked complete block; still require END and EOF

SERVER, STAGED UPLOAD
  read a bounded public header, enc, and complete START frame
  find the private key by key_id; find the PSK by the untrusted psk_id
  open and check START, then check issue time
  atomically reserve replay_id until its exclusive deadline
  check the time again and admit the reserved request
  check the logical target policy
  check each DATA; keep clear bytes away from the app
  check END, logical length, and true outer request body EOF
  only then call the app
  use the one response right to seal START, DATA, and END
  end the outer reply body
~~~

For a one-shot call, the host first collects the whole outer body through
true EOF. The API then checks that END ends that supplied envelope before it
asks the host for replay admission. The staged upload API
admits replay after START and checks DATA, END, and EOF after that. Both keep
clear request data from the app until replay, the full request, and the target
check pass.

## HTTPS binding

GET and POST use the **same** configured endpoint URL. A discovery client
sends no logical authorization, cookies, or PSK ID in GET. The first-party
adapters do not follow redirects for either call. Both calls need HTTPS.
A protected POST is one attempt;
clients do not resend it after a transport failure. A lost reply leaves the
app result unknown. Outer HTTP status is separate from the protected logical
status.

The examples use HTTP/1.1 syntax only to show fields. The protocol does not
require a specific HTTP version. Angle brackets mark binary body bytes.
HTTP transfer framing and any outer Content-Length describe the outer body,
not the clear logical body.

~~~http
GET /protected HTTP/1.1
Host: api.example.test
Accept: application/octet-stream
Cache-Control: no-store

HTTP/1.1 200 OK
Content-Type: application/octet-stream
Cache-Control: no-store

<one HHKD v2 record>

POST /protected HTTP/1.1
Host: api.example.test
Content-Type: message/hpke-http-request
Accept: message/hpke-http-response
Cache-Control: no-store

<one HHRQ v3 request through END, then outer body EOF>

HTTP/1.1 200 OK
Content-Type: message/hpke-http-response
Cache-Control: no-store

<one HHRP v3 response through END, then outer body EOF>
~~~

The successful GET body has exactly one HHKD record. A successful POST has
outer status 200 and the response media type above; the protected START holds
the logical status and fields. Send outer bodies without content coding.
Receivers accept an absent outer Content-Encoding or identity. An outer fault
can use an ordinary HTTP status and body; do not parse it as a protected
reply. The Python adapter accepts media type parameters on a protected
reply, while the current Fetch adapter requires the exact response media type
with no parameters. Send the exact value shown above for both. Use one
Content-Type field on the POST and on each successful GET or POST reply.

The source starts the discovery lease clock **before** GET. It checks the
lease again before it yields POST START: elapsed time must be less than
use_for_s. If it cannot refresh a spent key
without losing a one-use body, it fails before POST START. The service must
allow for the time from that final check through delivery and parsing of POST
START. The explicit key source may keep a checked key for its lease even
though the HTTP response says Cache-Control: no-store.

The first-party clients check a logical HTTPS origin before they read its
body or send GET. They fold DNS case, use IDNA names, normalize IPv6 and
default port 443, and keep other ports distinct. A gateway needs an
explicit client target origin and server expected authority.

## HHKD v2 key record

HHKD v2 has its own version byte. It does not change hpke-http/3 request or
response bytes. Read the GET body to its real EOF and reject extra bytes.

| Field | Size | Value |
| --- | ---: | --- |
| Magic | 4 bytes | ASCII HHKD |
| Version | 1 byte | 0x02 |
| Key ID length | 1 byte | 1..255 |
| Key ID | Key ID length | Opaque public ID |
| Recipient public key | 32 bytes | X25519 public key |
| use_for_s | 4 bytes | Positive unsigned seconds, big endian |

The exact record is 43..297 bytes. A client checks outer status 200,
Content-Type application/octet-stream, absent or identity Content-Encoding,
record size, version, ID length, 32-byte key, positive lease, and EOF.
The record has no signature of its own; the client trusts the HTTPS origin.
There is no HHKD v1 fallback. A client and host that use different HHKD
versions need separate endpoints or a coordinated switch.

For a planned recipient key change from A to B, set **all** workers to
advertise A and accept B; then advertise B and accept A. Remove A only after
its last lease, the POST START delivery bound, and a worker clock margin
end. Do not reuse a key ID while keys overlap. The host can advertise one
key and accept other keys during this change.

## Byte notation and lengths

All fixed-size integers below are unsigned and big endian. The symbol ||
means byte concatenation. ASCII labels are exact byte strings; 0x00 means
one NUL byte. A frame's ciphertext_len counts ciphertext **and** the
16-byte tag, but excludes its own four-byte length. A record number n starts
at zero for START and rises by one for each DATA and END.

~~~text
vec(x)   = quic_varint(len(x)) || x
fields   = vec(name_1) || vec(value_1) || ... || vec(name_N) || vec(value_N)
frame(P) = ciphertext_len:u32be || AEAD_seal(P, aad_for_this_frame)
~~~

The first two bits of a QUIC integer select its 1, 2, 4, or 8 byte width.
The remaining bits hold a 6, 14, 30, or 62 bit value in network byte order.
Use the **shortest** width. Reject a longer encoding of the same value, a
short vector, an odd field vector, and bytes after a complete field section.
The field section has no pair count. The request START has one length around
its field section; the response START uses the remaining record bytes for
its field section.

## Protected request

The request has exactly one START, zero or more DATA records, one END, and
then true outer POST body EOF. This form also applies to an empty body.

| Order | Field | Size | Value |
| ---: | --- | ---: | --- |
| 1 | Magic | 4 bytes | ASCII HHRQ |
| 2 | Version | 1 byte | 0x03 |
| 3 | key_id_len | 1 byte | 1..255 |
| 4 | psk_id_len | 1 byte | 1..255 |
| 5 | KEM ID | 2 bytes | 0x0020: DHKEM(X25519, HKDF-SHA256) |
| 6 | KDF ID | 2 bytes | 0x0001: HKDF-SHA256 |
| 7 | AEAD ID | 2 bytes | 0x0003: ChaCha20-Poly1305 |
| 8 | issued_at_unix_s | 8 bytes | Unix seconds |
| 9 | key_id | key_id_len | Public recipient key ID |
| 10 | psk_id | psk_id_len | Public PSK lookup ID |
| 11 | enc | 32 bytes | HPKE encapsulated key |
| 12 | records | To EOF | Four-byte length and ciphertext per record |

Fields 1..10 are the public_header. Its fixed prefix is 21 bytes; its
full size is 23..531 bytes. The PSK ID in this header is an **untrusted
lookup hint** until the START tag passes. Neither ID is secret, but an
ID can identify a tenant or client to an observer.

| Record | Clear bytes before encryption | Count |
| --- | --- | --- |
| START | 0x01 || vec(method) || vec(authority) || vec(path) || vec(fields) | Exactly one, first |
| DATA | 0x02 || coding:u8 || coded_body | Zero or more |
| END | 0x03 | Exactly one, last |

The scheme is HTTPS and has no vector. Each DATA decodes to a nonempty clear
part of at most 64 KiB. A DATA record count cannot exceed 1,048,576. The
sum of clear DATA bytes is bounded separately. Request START can carry an
empty field section. END has no body. An unknown kind, bad order, missing
END, short frame, or any bytes after END fails the request.

### Request HPKE steps

Use RFC 9180 PSK mode with the IDs above. The client and server share the
PSK and its public PSK ID. A PSK has at least 32 bytes and must differ from
its ID; provision at least 32 bytes of **entropy**. An X25519 recipient public
key and enc each have 32 bytes.

~~~text
info = ASCII("message/hpke-http request") || 0x00
       || ASCII("v3") || 0x00 || public_header
(enc, ctx) = HPKE.SetupPSKS(recipient_public_key, info, psk, psk_id)

for n, clear_record in START, each DATA, END:
    ciphertext_len = len(clear_record) + 16
    aad = ASCII("hpke-http/3 request record") || 0x00
          || public_header || u64be(n) || u32be(ciphertext_len)
    ciphertext = ctx.Seal(aad, clear_record)  # RFC 9180 sequence advances
    append u32be(ciphertext_len) || ciphertext
~~~

Do not make a new HPKE context for each record. Use the one context and its
RFC 9180 nonce sequence. The server uses SetupPSKR with the same inputs and
opens records in order with the same AAD.
Make a new encapsulation for each protected attempt. Do not resend the same
envelope after a failed or lost outer POST.

### Request time and replay

Let t be issued_at_unix_s and now be trusted Unix seconds at START
authentication. Accept only if t <= now + 30 and now < t + 330, with checked
integer arithmetic. The exclusive replay deadline is t + 330. A replay ID
is:

~~~text
SHA-256(ASCII("hpke-http/replay") || 0x00 || ASCII("v3") || 0x00
        || public_header || enc)
~~~

The host atomically reserves this 32-byte ID only once, across all workers
that can accept the same credentials, until the exclusive deadline. The
admission result must match this ID and deadline. Check now < deadline again
when applying it. A denied or uncertain reserve result fails closed. Do not
give clear request bytes to the app until replay admission, all DATA checks,
END, Content-Length if present, outer EOF, and logical target policy pass.

## Protected response

One admitted request grants one response right. The response uses a new
32-byte random server_nonce and a key bound to the request HPKE context.
The response writer consumes that right; it cannot make a second protected
response with the same material.

~~~text
response_secret = HPKE.Export(
    request_context,
    ASCII("message/hpke-http response") || 0x00 || ASCII("v3"),
    32
)
salt = enc || server_nonce
prk = HKDF-SHA256.Extract(salt, response_secret)
response_key = HKDF-SHA256.Expand(prk, ASCII("hpke-http/3 response key"), 32)
base_nonce = HKDF-SHA256.Expand(prk, ASCII("hpke-http/3 response nonce"), 12)

response_prefix = ASCII("HHRP") || 0x03 || server_nonce
for n, clear_record in START, each DATA, END:
    ciphertext_len = len(clear_record) + 16
    nonce = base_nonce
    nonce[4..12] ^= u64be(n)    # XOR the last eight bytes
    aad = ASCII("hpke-http/3 response record") || 0x00
          || response_prefix || u64be(n) || u32be(ciphertext_len)
    ciphertext = ChaCha20-Poly1305.Seal(response_key, nonce, aad, clear_record)
    append u32be(ciphertext_len) || ciphertext
~~~

The response_prefix has 37 bytes. The wire is that prefix followed by
length-prefixed records through END and true outer response body EOF.

| Record | Clear bytes before encryption | Count |
| --- | --- | --- |
| START | 0x01 || status:u16be || fields | Exactly one, first |
| DATA | 0x02 || coding:u8 || coded_body | Finite: zero or one; SSE: zero or more |
| END | 0x03 | Exactly one, last |

The logical status is 200..599. The fields after status are ordered
name/value vectors up to the end of START; they have no enclosing length or
pair count. A duplicate Content-Type fails. The receiver checks the START
tag before it uses status or fields.

A finite response has an empty body with no DATA, or one DATA with a
nonempty body. Hold finite DATA until END and **real outer EOF** pass. For
SSE, one DATA holds one complete checked block; it can reach the caller as
soon as its own tag, coding, and block checks pass. The stream as a whole
still fails if END or EOF is missing. No DATA may follow END.

SSE mode applies when the one Content-Type value, before any semicolon
parameter and after trimming ASCII space or tab, equals text/event-stream
without case. It requires logical
status 200, a request method other than HEAD, and no logical Content-Length
or Content-Encoding. Each decoded DATA block must be nonempty, use LF line
ends with no CR, and end with one empty line at its end. A block can consist
of one LF byte. The engine checks block shape; the caller parses UTF-8,
fields, comments, and dispatch under the
[SSE rules](https://html.spec.whatwg.org/multipage/server-sent-events.html).
There is no total SSE stream body cap, but each block has a body limit.

## DATA coding and logical HTTP checks

Each DATA uses coding 0 for raw clear bytes or 1 for **one** zstd frame.
The coded bytes and the coding byte are inside the protected record. For
coding 1, decode exactly one RFC 8878 frame; reject a bad checksum, extra
coded bytes, an empty result, or a decoded result over the part limit.
Check a DATA tag before using its coding byte or decoding its body.
A receiver accepts a valid zstd frame even when it does not save bytes.
The current Rust writer tries zstd only for clear parts of at least 64
bytes and sends it only if it is shorter. That is a writer choice, not a
receiver rule. This coding does not set outer HTTP Content-Encoding.
The current decoder caps the zstd window at the larger of 1 MiB and the
clear part limit. It checks a declared clear size, when present, against
that part limit.

The engine checks these logical message rules:

- The request method is exactly GET, POST, PUT, PATCH, DELETE, HEAD, or
  OPTIONS in uppercase ASCII.
- Authority is nonempty graphic ASCII. It is an RFC 3986 reg-name with an
  optional decimal port, a valid numeric IPv4 address, or a bracketed IPv6
  or IPvFuture literal with an optional decimal port. A four-part host made
  only of digits and dots must parse as IPv4. Percent escapes in a reg-name
  have two hex digits. Port digits may have leading zeros; their value
  must fit 0..65535. User information is not allowed.
- Path is exactly * only for OPTIONS, or starts with /. Its raw bytes are
  graphic ASCII with valid percent escapes, no #, and no backslash. The
  part before ? must decode as UTF-8 and must have no control byte,
  backslash, encoded slash, or . or .. segment. The query still needs
  valid percent escapes but has no decoded UTF-8 check.
- Each field name is a nonempty lowercase HTTP token: ASCII a..z, 0..9,
  or one of ! # $ % & ' * + - . ^ _ ~ or the backtick or vertical bar.
  Each value uses only tab or ASCII 0x20..0x7e, with no leading or trailing
  space or tab. Ordered duplicate fields stay separate.
- The engine rejects these names in either message: connection, expect,
  host, keep-alive, proxy-authenticate, proxy-authentication-info,
  proxy-authorization, proxy-connection, te, trailer, transfer-encoding,
  and upgrade.
- A request Content-Length, if present, is one nonempty ASCII decimal
  u64 value equal to the sum of **clear** DATA bytes. A response
  Content-Length, if present, is one nonempty decimal field. It equals
  the clear finite body size, except for HEAD and status 304, where it
  is metadata. Status 204 forbids it; 205 permits only zero.
  HEAD and status 204, 205, or 304 have no response body.

The Rust wire engine treats logical Content-Encoding as an ordinary allowed
field. The Python and Fetch HTTP adapters accept only absent or identity
logical Content-Encoding, since they do not add a second logical body
decoder. The adapters also drop transport-only fields before protection.
This adapter policy does not change DATA coding. A direct Rust HTTP host
must apply its own logical content-coding policy.

### Current engine limits

These are v4.0.0 engine limits, not new fields in the format. MiB, KiB, and
GiB use powers of 1024. A service can set lower limits in both peers.

| Limit | Default | Hard maximum | Counts |
| --- | ---: | ---: | --- |
| Complete in-memory request, finite reply, or one SSE block | 8 MiB | 64 MiB | Clear body bytes |
| Streamed request total | 1 GiB | 4 GiB | Sum of clear DATA bytes |
| Request DATA part | 64 KiB | 64 KiB | Clear bytes per record |
| Request DATA records | 1,048,576 | 1,048,576 | Records |
| Header size | 16 KiB | 64 KiB | Sum of name and value bytes per message |
| Header count | 64 | 256 | Pairs per message |
| Target size | 8 KiB | 8 KiB | Authority plus path bytes |
| Public key or PSK ID | 255 bytes | 255 bytes | Each ID; minimum 1 byte |

The engine also bounds START and DATA ciphertext frame storage before it
allocates it. The request START ciphertext bound is
max_header_bytes + max_target_len + 16 * max_header_count + 256 + 16.
Later request ciphertext lengths cannot exceed 64 KiB + 2 + 16.
The response reader uses
4 + max_header_bytes + 16 * max_header_count + 16 for its START bound
and 2 + max_body_len + 16 for later records. The limits on **clear** bytes
do not bound the whole outer body byte-for-byte. An HTTP host must also
bound concurrent uploads, elapsed time, and temporary storage.
All configurable limits can be zero except max_request_bytes, whose
minimum is one byte.

## Failure and test data

Reject a bad tag, unsupported version or suite, invalid field, broken
frame, limit breach, repeated or missing END, or missing real outer EOF.
Do not release a partial finite reply or dispatch a partial request.
Errors must not include PSK, private key, or clear payload bytes. HTTPS
protects the outer exchange. HPKE does not hide the endpoint, public IDs,
payload size, record count, or timing, and this format adds no padding.
The outer URL and endpoint path are absent from HPKE info and record AAD.
The host enforces its route and logical target policy, and shares replay
state across routes that can accept the same credentials.

The frozen [hpke-http/3 wire vectors](rust/hpke-http/tests/vectors/protocol-v3.json)
give request, finite response, SSE, and compressed request bytes, including
AAD and nonce inputs. The separate
[HHKD v2 record vectors](rust/hpke-http/tests/vectors/key-discovery-v2.json)
give exact discovery bytes. The
[Rust corpus test](rust/hpke-http/tests/corpus.rs) and
[Python corpus test](python/tests/test_protocol_corpus.py) check the shared
wire data. Use the tables and pseudocode above as the format; use the
vectors to check an implementation.

For example, the main wire vector has a request issue time of 1800000000,
a 15-byte key ID, and a 9-byte PSK ID. Its protected records have kinds
START, DATA, DATA, END. The two raw DATA records decode to 8 and 6 bytes;
their sum matches the logical Content-Length of 14. Its finite reply has
logical status 201 and START, DATA, END. This gives a short check of the
tables without copying the full hex exchange.
