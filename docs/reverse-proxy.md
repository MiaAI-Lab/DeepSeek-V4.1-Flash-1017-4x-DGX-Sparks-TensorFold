# Serving rank 0 behind a reverse proxy

Rank 0 is the only rank with an HTTP listener, so a deployment that wants TLS, a WAF or a rate limit puts a proxy in front of it and leaves ranks 1-3 on the private links.
This page is what the proxy has to do with connections, and the one guarantee it can rely on.

## The server keeps its connections

Rank 0 keeps a connection after a reply: it may sit idle for 900 seconds, and the next request on it starts a fresh head budget.
So a proxy's upstream pool is the intended way to reach it, and the pool is also the setting that matters most: with nginx's default 60-second `keepalive_timeout`, every gap longer than a minute between an agent's turns closed its connection, and every turn paid for a new handshake.
Patch `0044` drains the request body of a `GET` or a `DELETE`, which the routes never read.
That is the guarantee a pooled connection needs: the next request on it is parsed from its own request line.

Without it, a body-carrying `GET` (what a scanner probe sends, or a proxy that always sends one) left its bytes in the socket, and at the next keep-alive reuse they were glued onto a real request's line: cross-session `400 bad request syntax` replies, with the probe's bytes inside an agent's error page (2026-10-08).
A body whose framing is uncertain closes the connection instead of desyncing it.

## nginx

```nginx
upstream spark-llm {
    keepalive 16;
    keepalive_requests 10000;
    keepalive_timeout 300s;
    server 10.0.0.11:8888;
}

limit_req_zone $binary_remote_addr zone=sparkapi:10m rate=30r/s;

server {
    listen 443 ssl;
    client_max_body_size 64m;

    location ~ ^/(health|v1(/|$)) {
        proxy_pass http://spark-llm;
        proxy_http_version 1.1;
        proxy_set_header Connection "";

        proxy_socket_keepalive on;
        proxy_connect_timeout 5s;
        proxy_next_upstream error timeout;

        gzip off;
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;

        limit_req zone=sparkapi burst=60 nodelay;
        limit_req_status 429;
    }

    location / {
        return 404;
    }
}
```

`keepalive_timeout 300s` and `keepalive_requests 10000` are what make the pool survive a long session: the default timeouts recycle the connection between an agent's turns, and the default request count recycles it mid-session.
`client_max_body_size 64m` covers a full-window agent prompt; nginx's own 1 MiB default answers 413 to one, and the engine's own limit is 32 MiB.

`proxy_connect_timeout 5s` with `proxy_next_upstream error timeout` is there because on the private LAN we measured the first SYN of a fresh connection to rank 0 being dropped silently now and then: with the default 60-second connect timeout that request stalled a full minute before the retry, and with a five-second cap the retry costs seconds.
A pooled connection avoids most of those handshakes in the first place.

`proxy_socket_keepalive on` covers the other half: a pooled connection that dies while idle is detected and replaced instead of being waited on.

Forward only `/health` and `/v1/*`. `/health` is the one route that answers without a key, which is what a guard or a dashboard polls; everything else is answered at the proxy and never reaches rank 0.
Streamed replies need `proxy_buffering off` and `gzip off` in that location, and `proxy_read_timeout` above the longest generation.
If the deployment sets `TF_SSE_KEEPALIVE_S`, the proxy must not buffer the SSE comments it carries.

## Adding a key

`start.sh` passes its extra arguments to rank 0's `serve` after the defaults, so keys are opt-in the same way:

```sh
./start.sh restart --api-key-file /root/spark-key
```

The file must be readable only by its owner. With keys on, `/v1/*` needs the bearer and `/health` still answers without it, and its access line is tagged `key=unauthenticated` unless patch `0045` is applied.

## What to check when it goes wrong

| Symptom | Check |
| --- | --- |
| Cross-session `400` with foreign bytes in the message | Confirm `proxy_http_version 1.1` and `proxy_set_header Connection ""`, and that patch `0044` is in the image. |
| A turn stalls about 60 seconds, then completes | A fresh connection's SYN is being dropped: short `proxy_connect_timeout` with `proxy_next_upstream error timeout`, and pool the upstream. |
| Replies arrive in bursts after the turn ends | Buffering is on: `proxy_buffering off` and `gzip off` for the `/v1/*` location. |
| `413` on a long prompt | `client_max_body_size`, and the engine's own 32 MiB body limit, which no proxy setting raises. |
| `503` with `Retry-After` on a burst of connections | The engine's 128-open-connection cap, idle keep-alive connections included. |
