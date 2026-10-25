#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastapi>=0.115.0",
#     "uvicorn>=0.30.0",
#     "httpx>=0.27.0",
#     "psycopg[binary]>=3.1.0",
# ]
# ///
"""ghloc - Python rewrite of https://github.com/subtle-byte/ghloc.

One file with both programs the Go version ships:

    uv run rewrite.py [-c] [-m MATCH]      # was cmd/ghloc:
                                           #   counts LOC of the current directory,
                                           #   -c prints the table to console,
                                           #   no flags starts a local web UI for it
    uv run rewrite.py serve                # was cmd/server: HTTP API for GitHub repos

Environment variables of the server are kept identical: JSON_LOGS (default
true), DEBUG_TOKEN, MAX_REPO_SIZE_MB (required), MAX_CONCURRENT_WORK
(required), DB_CONN (optional; without it the server runs without cache).

Deliberate deviations from the Go original:
  * /debug/{user}/{repo}/{branch}: Go streams a `pprof` CPU profile. With a
    valid debug token this rewrite returns a `cProfile` text profile instead.
  * The server binds 0.0.0.0:8080 (Go binds the wildcard ":8080").
  * JSON log field order approximates zerolog's, it is not byte-identical.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import contextvars
import gzip
import io
import json
import mimetypes
import os
import re
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
import uuid
import webbrowser
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# imported at top level: FastAPI resolves the (stringified) request annotations
from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import PlainTextResponse, RedirectResponse, Response
from starlette.middleware.cors import CORSMiddleware

SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# tiny zerolog-compatible logger (used by serve mode)
# ---------------------------------------------------------------------------

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
_json_logs = threading.Event()
_stdout_lock = threading.Lock()


def _now() -> str:
    # zerolog: RFC3339Nano, UTC, rounded to microseconds
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def log(level: str, msg: str, **fields: Any) -> None:
    rid = _request_id.get()
    if rid is not None:
        fields = {"requestId": rid, **fields}
    if _json_logs.is_set():
        record = {"level": level, "time": _now(), **fields, "message": msg}
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False, default=str)
    else:
        extra = " ".join(f"{k}={v}" for k, v in fields.items())
        line = " ".join(filter(None, (_now(), level.upper(), msg, extra)))
    with _stdout_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def log_info(msg: str, **fields: Any) -> None:
    log("info", msg, **fields)


def log_warn(msg: str, **fields: Any) -> None:
    log("warn", msg, **fields)


def log_error(msg: str, **fields: Any) -> None:
    log("error", msg, **fields)


# ---------------------------------------------------------------------------
# loc_count: line counting, filtering, statistics tree
# (port of internal/service/loc_count)
# ---------------------------------------------------------------------------

# Bytes that make a file "binary" in the Go version (everything <= ' ' except
# '\t' '\n' '\r' ' '); such a file is counted as exactly one line.
_BINARY_RE = re.compile(rb"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_READ_CHUNK = 1 << 16

# translate table: byte > ' ' -> 1 (non-space present), '\n' -> '\n',
# everything else (' ', '\t', '\r') -> 0. Used for fast line classification.
_TRANS_TABLE = bytes(1 if b > 32 else (10 if b == 10 else 0) for b in range(256))


def _count_loc(reader: Any) -> int:
    """Port of loc_count.fileLOCCounter.Count: counts non-empty lines.

    A line is non-empty if it contains any byte > ' '. The last line without
    a trailing '\n' counts too. A file containing any "binary" byte is counted
    as 1 (that is what the Go version returns).
    """
    loc = 0
    carry_has = False  # does the segment after the last '\n' contain a non-space byte?
    while True:
        chunk = reader.read(_READ_CHUNK)
        if not chunk:
            break
        if _BINARY_RE.search(chunk):
            return 1  # binary file: the Go counter returns exactly here
        pieces = chunk.translate(_TRANS_TABLE).split(b"\n")
        if len(pieces) == 1:
            carry_has = carry_has or b"\x01" in pieces[0]
            continue
        # pieces[0] completes the carried-over segment (one '\n' seen)
        if carry_has or b"\x01" in pieces[0]:
            loc += 1
        for piece in pieces[1:-1]:
            if b"\x01" in piece:
                loc += 1
        carry_has = b"\x01" in pieces[-1]
    if carry_has:
        loc += 1
    return loc


@dataclass(frozen=True)
class LOCForPath:
    path: str
    loc: int


@dataclass
class FileForPath:
    path: str
    opener: Callable[[], Any]  # returns a binary file-like object; may be used once


class FilesLOCCounter:
    """Port of loc_count.FilesLocCounter."""

    def __init__(self) -> None:
        self._locs_for_paths: list[LOCForPath] = []

    def add_file(self, path: str, reader: Any) -> None:
        self._locs_for_paths.append(LOCForPath(path=path, loc=_count_loc(reader)))

    def get_locs_for_paths(self) -> list[LOCForPath]:
        return self._locs_for_paths


def count_files(counter: FilesLOCCounter, files: list[FileForPath]) -> None:
    for file in files:
        with file.opener() as reader:
            counter.add_file(file.path, reader)


def is_filtered(path: str, expr: str) -> bool:
    """Port of loc_count.filtered.

    s is wrapped in ^...$: patterns like "!.go$" / "^src/" work on full paths.
    Comma-separated list; "!" prefix negates a pattern. If any positive pattern
    matches, the path is filtered out; a negative match keeps it. An expression
    starting with "!" inverts the default.
    """
    path = "^" + path + "$"
    expr = expr.strip()
    result = expr.startswith("!")
    for pattern in expr.split(","):
        if pattern.startswith("!"):
            if pattern[1:] in path:
                result = False
        elif pattern in path:
            result = True
    return result


@dataclass
class StatTree:
    """Port of loc_count.StatTree. `children is None` marks a file node."""

    loc: int = 0
    loc_by_langs: dict[str, int] | None = None  # only for dir nodes
    children: dict[str, "StatTree"] | None = None  # dir nodes only

    @property
    def is_dir(self) -> bool:
        return self.children is not None


def _split_path(file_path: str) -> tuple[list[str], str]:
    dirs_str, _, file_name = file_path.rpartition("/")
    dirs = dirs_str.split("/") if dirs_str else []
    return dirs, file_name


def _get_lang_name(file_name: str) -> str:
    i = file_name.rfind(".")  # like Go's path.Ext (works with ".env" etc.)
    return file_name[i:] if i >= 0 else file_name


def _tree_add(tree: StatTree, path: str, loc: int) -> None:
    dirs, file_name = _split_path(path)
    for dir_name in dirs:
        child = tree.children.get(dir_name)
        if child is None:
            child = StatTree(children={})
            tree.children[dir_name] = child
        tree = child
    if file_name:
        tree.children[file_name] = StatTree(loc=loc)


def _count_dirs(dir_node: StatTree) -> None:
    dir_node.loc_by_langs = {}
    for child_name, child in dir_node.children.items():
        if child.is_dir:
            _count_dirs(child)
            for lang_name, loc in child.loc_by_langs.items():
                dir_node.loc_by_langs[lang_name] = dir_node.loc_by_langs.get(lang_name, 0) + loc
                dir_node.loc += loc
        else:
            lang_name = _get_lang_name(child_name)
            dir_node.loc_by_langs[lang_name] = dir_node.loc_by_langs.get(lang_name, 0) + child.loc
            dir_node.loc += child.loc


def build_stat_tree(
    locs_for_paths: list[LOCForPath],
    filter: str | None = None,  # noqa: A002  (Go names it `filter`)
    matcher: str | None = None,
) -> StatTree:
    root = StatTree(children={})
    for item in locs_for_paths:
        if filter is not None and is_filtered(item.path, filter):
            continue
        if matcher is not None and not is_filtered(item.path, matcher):
            continue
        _tree_add(root, item.path, item.loc)
    _count_dirs(root)
    return root


# ---------------------------------------------------------------------------
# rest: SortedStat JSON + responses
# (port of internal/server/rest)
# ---------------------------------------------------------------------------


def sorted_stat_json(tree: StatTree) -> str:
    """Port of rest.SortedStat.MarshalJSON (compact form).

    A file node marshals to just its LOC number; a dir node to
    {"loc":N,"locByLangs":{...},"children":{...}} where locByLangs is sorted
    by value desc / key asc and children by their loc desc / key asc.
    """
    if not tree.children:
        return json.dumps(tree.loc, ensure_ascii=False)
    out = ['{"loc":', json.dumps(tree.loc)]
    if tree.loc_by_langs:
        langs = sorted(tree.loc_by_langs.items(), key=lambda kv: (-kv[1], kv[0]))
        body = ",".join(
            f"{json.dumps(k, ensure_ascii=False)}:{json.dumps(v)}" for k, v in langs
        )
        out.append(f',"locByLangs":{{{body}}}')
    if tree.children:
        children = sorted(tree.children.items(), key=lambda kv: (-kv[1].loc, kv[0]))
        body = ",".join(
            f"{json.dumps(k, ensure_ascii=False)}:{_dir_stat_json(v)}" for k, v in children
        )
        out.append(f',"children":{{{body}}}')
    out.append("}")
    return "".join(out)


def _dir_stat_json(tree: StatTree) -> str:
    return sorted_stat_json(tree)


def to_pretty_json(compact: str) -> str:
    """Same indentation style as Go's json.MarshalIndent(v, "", "  ")."""
    return json.dumps(json.loads(compact), indent=2, ensure_ascii=False)


class BadRequest(Exception):
    """Port of rest.BadRequest."""

    def __init__(self, msg: str) -> None:
        super().__init__(msg)
        self.msg = msg


class RepoTooLarge(BadRequest):
    """Port of github_stat.ErrRepoTooLarge."""

    def __init__(self) -> None:
        super().__init__("Too large repository")


class ErrNoData(Exception):
    """Port of github_stat.ErrNoData (no cache entry)."""


class RepoNotFound(Exception):
    """Port of rest.NotFound."""


def error_response(exc: Exception, pretty: bool) -> Response:
    """Port of the error branch of rest.WriteResponse."""
    if isinstance(exc, BadRequest):
        code, payload = 400, {"error": exc.msg}
    elif isinstance(exc, RepoNotFound):
        code, payload = 404, {"error": "Not found"}
    else:
        log_error("Internal server error", error=str(exc))
        code, payload = 500, {"error": "Internal server error"}
    if pretty:
        body = json.dumps(payload, indent=2, ensure_ascii=False)
    else:
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return Response(content=body, status_code=code, media_type="application/json")


# ---------------------------------------------------------------------------
# local files provider
# (port of internal/infrastructure/local_files_provider)
# ---------------------------------------------------------------------------


def get_files_in_dir(path: str = ".") -> list[FileForPath]:
    """Portable port of filepath.WalkDir: lexical order, no symlink following.

    Only exactly-named top-dirs ".git", ".vscode", ".idea" are skipped (the Go
    code compares the full walked path, matching only at that depth).
    """
    out: list[FileForPath] = []
    skip = {".git", ".vscode", ".idea"}

    def walk(path: str) -> None:
        with os.scandir(path) as it:
            for entry in sorted(it, key=lambda e: e.name):
                walked = os.path.normpath(entry.path)  # like filepath.Join(".", name)
                if entry.is_dir(follow_symlinks=False):
                    if walked in skip:
                        continue
                    walk(entry.path)
                elif entry.is_file(follow_symlinks=False):
                    entry_path = entry.path

                    def opener(p: str = entry_path) -> Any:
                        return open(p, "rb")

                    out.append(FileForPath(path=walked, opener=opener))

    walk(path)
    return out


# ---------------------------------------------------------------------------
# github files provider
# (port of internal/infrastructure/github_files_provider)
# ---------------------------------------------------------------------------


def build_github_url(user: str, repo: str, branch: str) -> str:
    return f"https://github.com/{user}/{repo}/archive/refs/heads/{branch}.zip"


class TempStorage:
    File = "file"
    Ram = "ram"


class GitHubFilesProvider:
    def __init__(self, max_zip_size_mb: int) -> None:
        self.max_zip_size_bytes = max_zip_size_mb * 1024 * 1024

    async def get_content(
        self, user: str, repo: str, branch: str, temp_storage: str
    ) -> tuple[list[FileForPath], Callable[[], None]]:
        """Downloads the repo zip (limit applies to the zip size) and returns,
        for every zip entry, a FileForPath whose opener opens that entry.
        The returned close() must free the zip source."""
        import httpx

        url = build_github_url(user, repo, branch)
        start = time.perf_counter()
        tmp = None
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
                async with client.stream("GET", url) as resp:
                    if resp.status_code == 404:
                        raise RepoNotFound()
                    if resp.status_code != 200:
                        raise RuntimeError(f"{url} unexpected status code {resp.status_code}")
                    total = 0
                    if temp_storage == TempStorage.File:
                        tmp = tempfile.TemporaryFile()
                        try:
                            async for chunk in resp.aiter_bytes():
                                total += len(chunk)
                                if total > self.max_zip_size_bytes:
                                    raise RepoTooLarge()
                                tmp.write(chunk)
                            tmp.seek(0)
                        except BaseException:
                            tmp.close()
                            tmp = None
                            raise
                        log_info(
                            "Using disk file to temporary store repo archive", fileName=tmp.name
                        )
                        source: Any = tmp
                    else:
                        buf = bytearray()
                        async for chunk in resp.aiter_bytes():
                            total += len(chunk)
                            if total > self.max_zip_size_bytes:
                                raise RepoTooLarge()
                            buf += chunk
                        log_info("Using RAM to temporary store repo archive")
                        source = io.BytesIO(bytes(buf))
        except (RepoTooLarge, RepoNotFound):
            raise  # both must reach the caller unwrapped (like rest.NotFound / BadRequest)
        except Exception as err:
            # fits Go's "%v %v %v" (non-200) and "do request: %w" wrapping
            if isinstance(err, RuntimeError) and "unexpected status code" in str(err):
                raise
            raise RuntimeError(f"do request: {err}") from err

        e = time.perf_counter() - start
        log_info("Downloaded repo zip archive", durationSec=e, url=url, sizeBytes=total)

        def _cleanup(zf: zipfile.ZipFile | None = None) -> None:
            if zf is not None:
                try:
                    zf.close()
                except Exception:
                    pass
            if tmp is not None:
                tmp.close()

        def _zip_file() -> zipfile.ZipFile:
            return zipfile.ZipFile(source)

        try:
            zf = await asyncio.to_thread(_zip_file)
        except BaseException:
            _cleanup()
            raise

        files: list[FileForPath] = []
        for info in zf.infolist():
            name = info.filename
            i = name.find("/")
            entry_path = name[i + 1 :] if i >= 0 else name
            files.append(FileForPath(path=entry_path, opener=partial(zf.open, info)))

        def close() -> None:
            _cleanup(zf)

        return files, close


# ---------------------------------------------------------------------------
# postgres LOC cacher + migrations
# (port of internal/infrastructure/postgres_loc_cacher and cmd/server/db.go)
# ---------------------------------------------------------------------------

_MIGRATION_RE = re.compile(r"^(\d+)_.+\.up\.sql$")


def connect_and_migrate_db(db_conn: str):
    if not db_conn:
        raise RuntimeError("env var DB_CONN is not provided")
    import psycopg

    migrations_dir = SCRIPT_DIR / "migrations"
    if not migrations_dir.is_dir():
        raise RuntimeError(f"create migrator: no such dir: {migrations_dir}")

    # golang-migrate compatible tracking table
    with psycopg.connect(db_conn, autocommit=True) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (version bigint primary key, dirty boolean)"
        )
        row = conn.execute("SELECT version, dirty FROM schema_migrations LIMIT 1").fetchone()
        if row is not None and row[1]:
            raise RuntimeError(f"migrate up: database is in dirty state at version {row[0]}")
        current = row[0] if row is not None else 0
        for file in sorted(migrations_dir.iterdir(), key=lambda p: p.name):
            m = _MIGRATION_RE.match(file.name)
            if m is None or (version := int(m.group(1))) <= current:
                continue
            for statement in filter(None, (s.strip() for s in file.read_text().split(";"))):
                log_info(f"migration: applying {file.name}")
                conn.execute(statement)
            conn.execute(
                "INSERT INTO schema_migrations VALUES (%s, %s) "
                "ON CONFLICT (version) DO UPDATE SET dirty = EXCLUDED.dirty",
                (version, False),
            )
            log_info(f"migration: applied version {version}")
            current = version

    return PostgresLOCCacher(db_conn)


class PostgresLOCCacher:
    """Port of infrastructure/postgres_loc_cacher. Methods mirror the Go
    interface; they are sync and expected to be called via asyncio.to_thread."""

    def __init__(self, db_conn: str) -> None:
        import psycopg

        self._psycopg = psycopg
        self._conn_str = db_conn
        threading.Thread(target=self._cleanup_loop, daemon=True).start()

    def _connect(self):
        return self._psycopg.connect(self._conn_str, autocommit=True)

    def _cleanup_loop(self) -> None:
        ttl = 3600  # hour
        while True:
            time.sleep(ttl)
            try:
                with self._connect() as conn:
                    conn.execute("DELETE FROM repos WHERE created_at < %s", (int(time.time()) - ttl,))
                log_info("Old cache is deleted")
            except Exception as err:
                log_error("Error deleting old cache", error=str(err))

    def set_locs(self, user: str, repo: str, branch: str, locs: list[LOCForPath]) -> None:
        repo_id = f"{user}/{repo}/{branch}"
        locs_json = json.dumps(
            [{"Path": x.path, "LOC": x.loc} for x in locs], separators=(",", ":")
        )
        start = time.perf_counter()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO repos VALUES (%s, %s, %s, %s)",
                (repo_id, locs_json, False, int(time.time())),
            )
        log_info(
            "Set LOC cache", durationSec=time.perf_counter() - start, sizeBytes=len(locs_json)
        )

    def set_too_large(self, user: str, repo: str, branch: str) -> None:
        repo_id = f"{user}/{repo}/{branch}"
        start = time.perf_counter()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO repos VALUES (%s, %s, %s, %s)",
                (repo_id, None, True, int(time.time())),
            )
        log_info("Marking as too large in cache", durationSec=time.perf_counter() - start)

    def get_locs(self, user: str, repo: str, branch: str) -> list[LOCForPath]:
        repo_id = f"{user}/{repo}/{branch}"
        start = time.perf_counter()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT locs, too_large FROM repos WHERE id = %s", (repo_id,)
            ).fetchone()
        if row is None:
            raise ErrNoData()
        loc_bytes, too_large = row
        if too_large:
            raise RepoTooLarge()
        log_info(
            "Got LOC cache",
            durationSec=time.perf_counter() - start,
            sizeBytes=len(loc_bytes or ""),
        )
        if loc_bytes is None:
            raise RuntimeError("unexpected null locs for non-too-large repo")
        items = json.loads(loc_bytes)
        return [LOCForPath(path=item["Path"], loc=item["LOC"]) for item in items]


# ---------------------------------------------------------------------------
# service
# (port of internal/service/github_stat)
# ---------------------------------------------------------------------------


class Service:
    def __init__(
        self, loc_cacher: PostgresLOCCacher | None, content_provider: GitHubFilesProvider,
        max_parallel_work: int,
    ) -> None:
        self.loc_cacher = loc_cacher  # possibly None
        self.content_provider = content_provider
        self.sem = asyncio.Semaphore(max_parallel_work)

    async def get_stat(
        self,
        user: str,
        repo: str,
        branch: str,
        *,
        filter: str | None = None,  # noqa: A002
        matcher: str | None = None,
        no_loc_provider: bool = False,
        temp_storage: str = TempStorage.File,
    ) -> StatTree:
        async with self.sem:
            locs: list[LOCForPath] | None = None
            if self.loc_cacher is not None:
                if not no_loc_provider:
                    try:
                        locs = await asyncio.to_thread(
                            self.loc_cacher.get_locs, user, repo, branch
                        )
                    except RepoTooLarge:
                        raise
                    except Exception as err:
                        log_warn(
                            "Failed to get LOCs from cache, will proceed without it",
                            error=str(err),
                        )
                else:
                    log_info("Proceeding without LOCs cache - as requested")

            if locs is None:
                try:
                    files, close = await self.content_provider.get_content(
                        user, repo, branch, temp_storage
                    )
                except Exception as err:
                    if isinstance(err, RepoTooLarge) and self.loc_cacher is not None:
                        try:
                            await asyncio.to_thread(
                                self.loc_cacher.set_too_large, user, repo, branch
                            )
                        except Exception as err2:
                            log_error("Error marking too large repo in the cache", error=str(err2))
                    raise
                try:
                    loc_counting_start = time.perf_counter()
                    counter = FilesLOCCounter()
                    await asyncio.to_thread(count_files, counter, files)
                    locs = counter.get_locs_for_paths()
                finally:
                    close()
                log_info("LOCs counted", durationSec=time.perf_counter() - loc_counting_start)

                if self.loc_cacher is not None and not no_loc_provider:
                    try:
                        await asyncio.to_thread(
                            self.loc_cacher.set_locs, user, repo, branch, locs
                        )
                    except Exception as err:
                        log_error("Error saving LOCs to the cache", error=str(err))

            tree_building_start = time.perf_counter()
            stat_tree = build_stat_tree(locs, filter=filter, matcher=matcher)
            log_info("Statistics tree built", durationSec=time.perf_counter() - tree_building_start)
            return stat_tree


# ---------------------------------------------------------------------------
# server mode
# (port of cmd/server)
# ---------------------------------------------------------------------------


def _get_stack() -> list[dict[str, str]]:
    frames = traceback.extract_stack()[:-2]  # skip this frame and the caller frame
    return [{"source": f"{f.filename}:{f.lineno}", "func": f.name} for f in frames]


class _RealIPMiddleware:
    """Port of chi middleware.RealIP."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
            ip = ""
            if (xff := headers.get("x-forwarded-for", "")) :
                ip = xff.split(",")[0].strip()
            if not ip:
                ip = headers.get("x-real-ip", "")
            if ip:
                scope["client"] = (ip, 0)
        await self.app(scope, receive, send)


class _RequestIDMiddleware:
    """Port of chi middleware.RequestID with RequestIDHeader="" (always random)."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            state = scope.setdefault("state", {})
            state["request_id"] = uuid.uuid4().hex[:16]
        await self.app(scope, receive, send)


class _RequestLoggerMiddleware:
    """Port of cmd/server NewRequestLoggerMiddleware (plus the requestId
    binding of NewLoggerMiddleware)."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
        client = scope.get("client") or ("", 0)
        path = scope.get("raw_path") or scope["path"].encode(errors="ascii")
        query = scope.get("query_string", b"")
        if isinstance(path, bytes):
            path = path.decode("ascii", "replace")
        if isinstance(query, bytes):
            query = query.decode("ascii", "replace")
        url = path + f"?{query}" if query else path
        token = _request_id.set(scope.get("state", {}).get("request_id"))
        try:
            log_info(
                "New request",
                **{
                    "method": scope["method"],
                    "tls": scope["scheme"] == "https",
                    "host": headers.get("host", ""),
                    "url": url,
                    "protocol": f'HTTP/{scope.get("http_version", "1.1")}',
                    "from": f"{client[0]}:{client[1]}",
                    "origin": headers.get("origin", ""),
                },
            )
            start = time.perf_counter()
            status: int | None = None
            response_bytes = 0

            async def send_wrapped(message: dict) -> None:
                nonlocal status, response_bytes
                if message["type"] == "http.response.start":
                    status = message["status"]
                if message["type"] == "http.response.body":
                    response_bytes += len(message.get("body", b"") or b"")
                await send(message)

            try:
                await self.app(scope, receive, send_wrapped)
            finally:
                log_info(
                    "Request finished",
                    durationSec=time.perf_counter() - start,
                    status=status if status is not None else 200,
                    responseBytes=response_bytes,
                )
        finally:
            _request_id.reset(token)


class _RecoverMiddleware:
    """Port of cmd/server NewRecoverMiddleware. On panic: log the stack and
    return a bare 500 (empty body, like Go's WriteHeader(500)) if possible."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        sent = False

        async def send_wrapped(message: dict) -> None:
            nonlocal sent
            if message["type"] == "http.response.start":
                sent = True
            await send(message)

        try:
            await self.app(scope, receive, send_wrapped)
        except asyncio.CancelledError:
            raise
        except Exception as err:
            log_error("Panic recovered", stack=_get_stack(), panicValue=repr(err))
            if not sent:
                await send({"type": "http.response.start", "status": 500, "headers": []})
                await send({"type": "http.response.body", "body": b""})


def _build_app(service: Service, debug_token: str) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/")
    def docs() -> Response:
        return Response(
            "<html><body><a href='https://github.com/subtle-byte/ghloc'>Docs</a></body></html>",
            media_type="text/html; charset=utf-8",
        )

    async def serve_stat(
        user: str, repo: str, branch: str, request: Request, *, profile: bool
    ) -> Response:
        qp = request.query_params
        filter_v = qp.getlist("filter")[0] if qp.getlist("filter") else None
        matcher_v = qp.getlist("match")[0] if qp.getlist("match") else None
        no_loc_provider = False
        temp_storage = TempStorage.File
        if debug_token:
            token_in_request = qp.get("debug_token", "")
            if token_in_request == debug_token:
                if "no_cache" in qp:
                    no_loc_provider = True
                if "mem_for_temp" in qp:
                    temp_storage = TempStorage.Ram
            elif token_in_request:
                return error_response(BadRequest("Invalid debug token"), pretty=True)

        if profile:  # mitigation for Go's pprof, see module docstring
            import cProfile
            import pstats

            profiler = cProfile.Profile()
            profiler.enable()
            try:
                await service.get_stat(
                    user,
                    repo,
                    branch,
                    filter=filter_v,
                    matcher=matcher_v,
                    no_loc_provider=no_loc_provider,
                    temp_storage=temp_storage,
                )
            finally:
                profiler.disable()
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                pstats.Stats(profiler, stream=out).sort_stats("cumulative").print_stats()
            return Response(
                out.getvalue().encode(),
                media_type="application/octet-stream",
                headers={"Content-Disposition": 'attachment; filename="profile"'},
            )

        pretty = qp.get("pretty", "true") != "false"
        try:
            stat = await service.get_stat(
                user,
                repo,
                branch,
                filter=filter_v,
                matcher=matcher_v,
                no_loc_provider=no_loc_provider,
                temp_storage=temp_storage,
            )
        except asyncio.CancelledError:
            raise
        except Exception as err:
            return error_response(err, pretty=True)  # like Go: errors always pretty
        compact = sorted_stat_json(stat)
        return Response(
            content=to_pretty_json(compact) if pretty else compact,
            media_type="application/json",
            headers={"Cache-Control": "public, max-age=300"},
        )

    @app.get("/debug/{user}/{repo}/{branch}")  # registered first: wins over /{user}/...
    async def debug_stat(user: str, repo: str, branch: str, request: Request) -> Response:
        if not debug_token or request.query_params.get("debug_token", "") != debug_token:
            return PlainTextResponse("404 page not found\n", status_code=404)
        return await serve_stat(user, repo, branch, request, profile=True)

    @app.get("/{user}/{repo}/{branch}")
    async def get_stat(user: str, repo: str, branch: str, request: Request) -> Response:
        return await serve_stat(user, repo, branch, request, profile=False)

    @app.get("/{user}/{repo}")
    async def redirect_stat(user: str, repo: str, request: Request) -> RedirectResponse:
        try:
            branch = await get_default_branch(user, repo)
        except asyncio.CancelledError:
            raise
        except Exception as err:
            return error_response(err, pretty=True)
        url = request.url.replace(path=request.url.path + "/" + branch)
        return RedirectResponse(str(url), status_code=307)

    async def get_default_branch(user: str, repo: str) -> str:
        import httpx

        url = f"https://api.github.com/repos/{user}/{repo}"
        async with httpx.AsyncClient(follow_redirects=False, timeout=None) as client:
            resp = await client.get(url)
        if resp.status_code == 404:
            raise RepoNotFound()
        if resp.status_code != 200:
            raise RuntimeError(f"status {resp.status_code} (body: {resp.text})")
        repo_info = {}
        try:
            repo_info = resp.json()
        except ValueError as err:
            raise RuntimeError(f"response is not json: {err}") from err
        if not repo_info.get("default_branch"):
            raise RuntimeError(f"empty branch (body: {resp.text})")
        return repo_info["default_branch"]

    # middleware order mirrors chi: RealIP, RequestID, Logger(s), Compress,
    # Recover, CORS - the first added is the outermost
    app.add_middleware(_RealIPMiddleware)
    app.add_middleware(_RequestIDMiddleware)
    app.add_middleware(_RequestLoggerMiddleware)
    app.add_middleware(GZipMiddleware, minimum_size=500, compresslevel=5)
    app.add_middleware(_RecoverMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])
    return app


def _env_json_logs() -> bool:
    raw = os.environ.get("JSON_LOGS", "")
    if raw == "":  # env v9: empty value falls back to the default
        return True
    allowed = {
        "1": True, "t": True, "T": True, "true": True, "True": True, "TRUE": True,
        "0": False, "f": False, "F": False, "false": False, "False": False, "FALSE": False,
    }
    if raw in allowed:
        return allowed[raw]
    log_error("Error parsing config", error=f"invalid JSON_LOGS value: {raw!r}")
    sys.exit(1)


def _env_required_int(name: str) -> int:
    raw = os.environ.get(name)
    if not raw:
        log_error("Error parsing config", error=f"required environment variable {name} is not set")
        sys.exit(1)
    try:
        return int(raw)
    except ValueError:
        log_error("Error parsing config", error=f"invalid {name}: {raw!r}")
        sys.exit(1)


def _server_main() -> int:
    import uvicorn

    _buildTime = "unknown"  # will be replaced during building the docker image
    log_info("Starting up the app", buildTime=_buildTime)

    json_logs = _env_json_logs()
    debug_token = os.environ.get("DEBUG_TOKEN", "")
    max_repo_size_mb = _env_required_int("MAX_REPO_SIZE_MB")
    max_concurrent_work = _env_required_int("MAX_CONCURRENT_WORK")
    db_conn = os.environ.get("DB_CONN", "")
    _json_logs.set() if json_logs else _json_logs.clear()
    log_info(f"Debug token is set: {debug_token != ''}")

    github = GitHubFilesProvider(max_repo_size_mb)
    pg: PostgresLOCCacher | None
    try:
        pg = connect_and_migrate_db(db_conn)
        log_info("Connected to DB")
    except Exception as err:
        log_info("Error connecting to DB", error=str(err))
        log_warn("Warning: continue without DB")
        pg = None

    service = Service(pg, github, max_concurrent_work)
    app = _build_app(service, debug_token)

    log_info("Listening on http://localhost:8080")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8080, access_log=False, log_config=None)
    except OSError as err:
        log_error("Fatal", error=str(err))
        return 1
    return 0


# ---------------------------------------------------------------------------
# ghloc CLI mode
# (port of cmd/ghloc)
# ---------------------------------------------------------------------------


def _print_stat_table(locs_for_paths: list[LOCForPath], matcher: str) -> None:
    stat_tree = build_stat_tree(locs_for_paths, None, matcher)
    stats = [(loc, lang) for lang, loc in stat_tree.loc_by_langs.items()]
    stats.sort(key=lambda v: -v[0])
    first_width = 50
    second_width = 20
    width = first_width + second_width + 3

    def print_pair(a: Any, b: Any, sep: str) -> None:
        a_str = str(a)
        if len(a_str) > first_width:
            a_str = a_str[: first_width - 3] + "..."
        b_str = str(b)
        sep_len = (first_width - len(a_str)) + (second_width - len(b_str))
        print(f" {a_str}{sep * (sep_len + 1)}{b_str} ")

    print("=" * width)
    print_pair("File type", "Lines of code", " ")
    print("=" * width)
    for loc, lang in stats:
        print_pair(lang, loc, ".")
    print("=" * width)
    print_pair("Total", stat_tree.loc, " ")
    print("=" * width)


def _count_locs_for_paths() -> list[LOCForPath]:
    print("Counting lines of code...", end="", flush=True)

    counted = threading.Event()

    def ticker() -> None:
        while not counted.wait(1.0):
            print(".", end="", flush=True)

    tick_thread = threading.Thread(target=ticker, daemon=True)
    tick_thread.start()

    files = get_files_in_dir(".")
    counter = FilesLOCCounter()
    for file in files:
        with file.opener() as reader:
            counter.add_file(file.path, reader)
    locs_for_paths = counter.get_locs_for_paths()

    counted.set()
    tick_thread.join()
    print()
    return locs_for_paths


# go:embed server_static equivalent: the whole local-mode UI bundle
# (index.html + assets), gzipped and base85-encoded; decoded to memory on use.
_EMBEDDED_SERVER_STATIC_B85 = (
      'ABzY8000000{`v3*^=r?(kA*UZFRek)s`U1Oto2GB(s=jknWH~f<S<3iAl8eyPwZJLQ)9Gy>|8a?O2<2T`XIPh8R73_<Yd+{KqniR;TFC@8!jJ'
      'kB5I2KU~dwcMJcFF6)2%_y6{fL9<t|<MMtvzeR&_D_YLIqu20aYq9gq=H}!yN_Sj8dd<<b?|)vrqF3x3+<Rwl#CH4{cdK95PLbL$*WNw|Jgs$E'
      'RW-j``MR1_Q{#Wzx3*dNTAS6|)mn^>W@BiWJMGP^&W_!K6_2Z%d2bQ+dj3~&S#1@%yLDl<_1-6w;krLx$KJMO_Ucw?wbxEv|0|Zy#GbFkxE7n$'
      'mTYHK=uBGg&cs}I_wv1ib?P2M^}RRw-dkr<i<cFRo;!8}OHbND8wGlw#vt1|QTup*-e1VJlJ{i$o&0lR9OHa@vfmf)AAPg(`A21Yy|_jDFjyb|'
      '$3Om`|CVf>fAChfe`<xon^%1Q{^vHLeQP&t-P!lF<=i;8E5V0VFwSmqxb~|0&MEYCXFfZ7mD1MT&o*9TR(Be+lRYnK?F#L2>6hv6&bSv%8o}MG'
      'o<iH-Xf5B?I)|R_Hp>_HFuFhf#=9q{Fw)W1mELs<^&Q#?@wetJTIYVjzPSEo>l8|NuTl=phOsvrql;A(eRmFZ-Cndx)50vW=RM6Sw#rW9dTbSL'
      'p|d|Z)Fx={xsO0Qw8z1ByRi%Vg;L<f<({_~VP3(-*%`iDn8n_H7URA#IGpxQ(L5&W;riu^?U%#$JlMCJ=GkioS9_uv=x4iZ1d(04g?P?y&vRps'
      'c<gxInvZt&eALI70yhq#&SA8%wbI6!PmRGu^Cz{SKaS0PYk%eS#5Kd*>leF;IWEevJ-=P7`OwE&V*J<ed%Qb4h1MPGP_V}vWv_17!%gTkM&0G2'
      'r}h2XwbShB)?tJ;rtjvYY_un*y*DpydES2cOzW{K^e)b!r(vwds;GS=dto+qXY8|P&%JoL7?oPSb#(nw=p06YU2KJ3b7w5ax!+-@!8EbQ*kb&<'
      'I@lMhxZAh<P24%O0vl_L`|-44fQEXd!Tja(+>TXjftGt0cOM*<^EX~AZ_x`n^B(3i3eaz8&2M*eYs8&>{u=fM^w5huTr;;}NcWv$?+$ti?Lu%C'
      'Yh1@VmP*dt(B0Vl6LTY4-w~ZH=k>_FYra#^L$6}Kzq;oZ>L+`wC*$)w&6Z&wW*GM<bP8|lMZ5{y3qBX<e|BvzT7lP`#m@G`zX#*mwXdwbgXd^}'
      '?8Ruitk63f%*l#eKQOT8N8Bq|3v<2nZ~IQ89(3y6XzbqxgT=|{G)BQ-cJeySO)wZ68><j#-;LQ@yZZU=7H@~|M(NXSUU%)qEp{85PuefLc>^!*'
      'eLA2A+}FG!ouQcn8uGrI;l(N5R#>mycO%fSH=9mj_o2>j;%|aZa|rrjy63Thrb7JF+Vh#?1e)R1Pv^ztguNZ<?*46UFLagu_MqPYYsBw@?*yGg'
      'd4qL6d$rQVZcIPCinRw1!aC>|oX;wsw+g+JYo|-?U@Vd22Or+P9-H;gXgR*coyL`E+J>yb*<E9w)?OuuiI$dySqZfBJy?t!H_+bQEog#h)HnZJ'
      'PIPTvY<d0tjqcGrV$6f~665p4d24ph;4x>rI64ltCee5VIy*uR?r08!c|Eoz-*V#;_*LoPHP(N0@5aCI+=lp;wP+o@xlVL@?^MmOzkdrm%~8p1'
      '%!ua%G)`sBi3bFoJ<i*{t~<@q(A^tXH?A4Z0er3!FyC^;9=v!ih-gkk``+FNZOB(T(+<x~e|}!a_3<v_2le669^aylSlhD)dH=8v({^JTfe)R<'
      'GjcDOuh5*P?|A>pZQnz!(>MjINqEM&3G8k7-KiVj%X`S}(%=1k2|WB7Ep)^0Z@=jOK)JT-AMDF=K3sPzx9>%v`!3di{$3}43trr8=~%yuyA2CY'
      'A=sH!ZmZDWgV)cwy+7OWX=6{q@LQqPCDuu_W9@CfRB-mj&K}oZT-(HL(AxJrlc#3g!de9iFP)wxZw4)wa(rT1k>?0>v|M!0ZWDa3*)_0M8t5H*'
      '67h&puxbtsuc@y0*G{Z=@Ufg*N3WWUH^|Q0xb(faf6{a3h5i}w?->6I5%Kn&z1R#u(?NS2-P#ke_I>p{Z%^8Q-!*=9_P4;<d)l&k(%NwpQlIU{'
      'o55`sPL5aCLUW>X8=%v32W%@_uWU??PtQ-YwNdCDvXSV7WUDey^UoHYc{#q^IyZ2=F~!>18_+#y>g_{}YbwSS?>2IeAzyhO+hmO3JA7Wm_~rA3'
      '?%CSc@3wpf`(kvf|7C0N@l#*?+~nkwuC*-c84Wh3L`O!RwrWgU?FwW7y$4zp`)lk-rnJY_ej1mqkd?F!A<p<_O#KzjU+v=UbL`hZ_tx2N%zUQ^'
      'IxNh_Hq!#hOqZTNwrfWN^bL8}y3jt@vliwJ{be7V?Kt@SZuSDN4*H$XF5Y4owrjgEH^v&|m0Vxu0pR6w48}C$g&%F`K=@5^S<C68l9y*RGNkzx'
      'Xug&&*2`l$Y8g0t6PA5g1Giv6CT|ql*=UJ;as1mi^a{vR>}#V7*+TCRnQsqphVkz_z+6PzJL)g!R5she{Ehw|(z>?mpT=C*BGMBf|31EP?%QL2'
      'wlG+j{D1pii$?Kg+i9B2C#gO3y@kC~bV&En?dE6>I%1s-d$CCQSWM3@E0SKShU3(=l&_I2OZXMeLyYqP`s6;3Sc4Mg{f*zDycU2Ci3X2?KVg4B'
      'M=IZOnYM0pW}-!WqxYC+cADMDN_cnUGxGe~vD2JJZsk_&G_=UR>))h{@>oGnkkirft{;@M!TT=GdBp4G+%@PrpPyLsbnl%w`o%hRWuwbA{xvTc'
      '9`->ene6CVIFM)QyLr=Ai*n!;thL9wTk8T{F|eR_4GO#_@2r>4N#|z%L42R+*?)c)&(B+d>$gD1Deuuq)>^E~m0)igkYz&Fcv9AN(%-xHA@o()'
      'w`b_tW%tn9tT5*G1n<<X2>NuY1N09wBY98mqu@Q0P-j}+zFi0U@aT<8JVvZ@!vE1v>6g9OCGY8yF5l6A!qcF8kxjz99s1)<wkghs<wrNr?{P5q'
      '<XTGk(7b&vNv{&^gQnJJ(uJ_6ddhz%`ceKJ+WP4N{UDK@AFxO6_P8w0_7?kEv_6;jQ-Sy)k4N_D1zN~$?n%yT#<Ifw?OTUwW<#NG{u}g$ehO+o'
      '7IZOdelnI-Ndr@t=vD7D*2WTZm^{%e&URJT1<$==Z1|h?IrO<-vz>u6QDB{Car<mHG@?z|8sZ+tdV>C`yM^^}$vUT%u076Ide2MN8MN>6J@lKN'
      '+0UbQ&-8|N(>2`6E>dVSox8bL_@&SL!RLk7p3c_Y5Ob;OckqayO=mE^ksQ1apfj@UALDE`TsoU~j4`EioIU>gS%0+Txf~j93|sin<+UMxfquze'
      'zdz{UptULSNZ3?EtPAsb(hsD}(!mox4ebA=GuFV*nbtCy3;z52G1&(?_jy?f{fg*L%1g-~cIx=S1HltNe|5)XFSf8A+;=42BRe0o*w8}O{V_(!'
      '@ynp0r8+zK(XVzK9Kat>+<!xNpa<gJ+i4EoL}w-^)0%dW=(jo9M!GKP&$MUp4YtMC#}cx_-GV+do*2{lAo;PJv0hDU!#ou08^QKT*9rRB%XJzQ'
      '(Leg){wE>HIo7k8ch^6Kyn-ItR&?0BzF0nGc2Xh>6Fz^9toU~4H50OSXDW6=dww;fyo}`k>AgO%7`HH8lP=_J6&{1VbPx1@37B1rY#hcw5}pp('
      '|4#3?$~&;LbI;Nkz#Cwj)UPh>TTSp*!YxQ1iZgc#9Oxs`4&6cylYM^e$0WO9qwQyJUcDDu`<v8n?E3)gmC`}6`)e>US(meLc44;&JuEH<{lb*='
      'WWrjsA$SDz6wm;z8~AcnkG~U}-Ur^Jtk<A$eaF1d(66;=A@H3B^rm87cYoN6M2C%~{*65epx?fQv_2`;1Y5~rxn&xu4SS$<6LvRbU5{uKI!+?9'
      '$bJQl>>h1kT8B(7k`0Lc%JC#>_wV#Ao+$u#(d}_|ANEdQ6;6R&!8I`s;%#hKE92zf**Nn$%Dkf3_4VkwVBF0-K|aG)ARV*}T&ElWTPz8karbV|'
      'P8#IMIia&bZ2J8}D(BDGCm(ckvY?C6_*c*yfo}q{gRVF+&alCMnTPV5dB~le-Ve6z2ic8;eKMv5EN2DX&h!Jmmh!%>xF_^yGlc#R{UjiJYzvHQ'
      'N^}AqCuFO@j52<3&%H~0gXYftz&5<Qy08uLoDVFO)===K3e&`Qd;SKQJLG+LFkfEN<bAA}>6?;Qz6oC0l+XGB>3QdQam1LxLdUu;p5J!@TqoU9'
      '=mk9X`MvB3jO~;7JAT*Y?>E1_Fb&pw$C&eCNV50pH0oP>yuJ(z^(&nxdpx{!DyC1em3Sj`k;|lE6mbr#7BE2I)r@fribL$3(RGSm;O=@G@FD2J'
      'uqPS;wMRNK;m?FM25sOgWFrutA$tku7yA2-_5irZ@E6f1&f<BX8=%)I<F+HnUGO5tQ=zv)XWVGeIsPO(b?)D&{eW}~V2A~KIx7&?!+1FHmIdtr'
      '<UwH;^n)Yuj1P5<OkR)C2J!)R6)?!cjc{#kJSV=<GG{ft3p!$Z7WDSCR;vc_v*<W(g4a}bSg+PmV)E%TuE3dho%rel;|x_^E%u;wWW9*=5Ai#q'
      'xhlVj=be<_nl*2=$QE2xOrom|%c%z0HYrZ63EnjYCfq8m3Nx~cjZ`*NKYd`b!1{st&g;L-#>}t^bq)OD<9m^v(?|LD-Mm8fXZiu?mHOUvL6aG8'
      'Xl;0`_2?#l<Mi+Tr1J)uScVTOUd~&(2ir%=QDy(?8rcliVWxpMYwahy`^a{;*Vu%F=gkXJmm5c$?o2zvUJg9co2teh_%71|+28&BG_OZAW)iay'
      'cn|Qm1o!v_jNWTEV4E}D4gXBr6!^y@?%B-qB*r^gjwd#+HNIUsO)FZG{m}f1ycl{c^hM|?)lPjEjr^NWbR~3B!FL4S1b%aX_o`vUIXxhgcM>aN'
      'z3iA_9+2mEVSk{{Ew#<MrO@AqU-0u%KaAY1?pUOQS2i2MBnayxJKLi5pPVp;Amy(KKeV=wy0bVV&V={`%Y!@SXtWzO!vby(3{WHfI^(^9-W3>*'
      '#d>8AawjsZiE$Vd+{ErWgRjGHFn#A|?!YA*JofFNNNs*fJ*aC4J+vDN9gg*{df#9B)DG5-$7}BC9FT9vJge84<_J&O8XDUYcm0AjA-|LlofZ4D'
      'dzX3`bQsun`&rSfe;u9i?N^)j^kI(%bJ#&K*(`+P0Andk!RuV~i+L4iX!8@lKT~Ycp*HY7e5nDSD_}5Qqpm@pXP$`P>ADT(5&gzo-wf!Q1vyv7'
      'ny5|QPv9>+F3iPM?H_UfPVcxy^0(2r%(D$XH)_n_SF!`UF2*rBz^0#x{Zr|Q`!043r45pSvM;^gZj%2)^h5GulH7}Leh0kOt1)hc^EjID9H({+'
      'zmCMjHz(y?vIPzFCot;xrYn5{2MK=Po-%i_hB|y&C6XZv@{dp8YjTKZrkLa>@HgY+HjFdzi&-IvoSIIU8~m)vKBR5bmh(}MFgveOQ~M#z8+$vu'
      '!*3*LV^O|1Y{QL0J85r{=*cAOf4>KV497f~Y`?#r({w-E+H<VO)85(nvD7`_-#1{5y@2|h3R;vjecgz8OnFV*!Uki$?S|^wEXq3?W3K%}?}+t)'
      '4_57GcC~3=#@XKDoHPDmPU_H41(wKo2%X`g1>EBG*^sn;mwc;WX78jlXjdVF#F?S%;14%V+Qt(96~C7WUySANxJ0tGZp{g&Rj|(*WP`>uUT59{'
      'O!7NvAN!oxEtr3lU=j}Wdx4ik<YRla?lZT+bo|$}_ru!J-|DlYMzDLiAI$d?>^UD_@Pd|;F&(9RVvLe``7{5Kej_~@7x31hSQGt=d?i&WKVrx='
      'qg%QDdwkdrp7*?w^0`~52ApKqh0d^xtO?dPeGfE{%AAZ>sN+x09mO%2|5a_ipwWKOBilHTJeRPNUu}K&+1)}1DWvTflAlt02k1n?r&>#WOe%Ks'
      'J(#{7CC>dV<PPyaVT0prlHK&?Rn{Tv>98$in-hoZbvnZ}^7$jqXC~=N!64hsFXOk*k2i!pqy1ZBFs~`JTgx!SnZ>`!9!+h&LM@f&ncwXVGW~XC'
      'ySr9L974U+*#~!t<sL`S6Gz+78Nm*HsZ;U%TV!KM9bhW4khSE+w}k!^j9bKXNWP5si>W-*<UZUb79wS*&HAh^=!NX)WWI+dp7p2H4}K-sx#Ur)'
      'uEn;G;6qi$iV|Lw$NTT?NpuALL1%@m*h3an2rsRHE(58Tua6_rdn<y!iMd{nF5#c4KF)Yn(%&}K^TxFe<Y+uEnySBX+nyzI684wy2ci!iTlk&V'
      'aZ3^>0gvYMy#=4yv(6NwKh`H+`h5tallo(-kBa}Rx&w@lY}gBIqoZPD9AP_u0Ds$Bv?nx{akd6@FBvP=Px_!JHY1N&3=6OQDv3^x_1JXCuR=I|'
      'dd}BGQ<>h8=;WYR#uLe26ZXvAgHMC#Xi2nEAuLMqFOvOUJzWRz0cH5s7vY@Hq1U9FV=rK1K*u%-KdGrUPm65oJK0p`g5{4sC{m1rLF>`CfH`yQ'
      '0PyF|p0MHBRN|CIH>NnEsUh|nw!fg;6vO`}KM(m8@_q{=#R4<C;#KS!<0BK{Yo|CE&^T;5j7!8$3@KJ9VErZWtJO*q#G3t`m_zpG{AFA4PmzxR'
      'wi;pOw4cd&-x43^x#i|!F@_I}Yb)b5Fo&AFb|rr^$xZOGwY0})9|8x}V?3Y6h7dL+cr&jx@t+MalLDP@!P|PD-!<s<zrhd~pRb_LIq_P^zC|xu'
      '&1mlf!ban9ebk#&t{W>Bc9cR-(#JzKfuKXTN|=*EGmGvcofXI~#{3AcLcA8_nSQzg$3GMOh&U7V{hs62wZ!K7Xg3IZsfB`1lKoI)J%xCQ*U%9g'
      'r*HTSH%BLuAwnnGXb$k{0k^Hr34UtAs?=*${gUb_vJJw&U{}-m5SVOZO+HWXl`iQ$q2TfC$5io~E8{WGGA5FKRp}Qjg{~%K4dh)oD1N=u`A~3Q'
      '@`niwg=ljg94H=f37r~oXTnzvyvYp+-w37t2zgflKd*j?F*eQAA1Ls<F9SGw>SuK;+NmE0d@dDTjbg6^FPNORM3?5l-Z;&f)RoqB{;%wx6MGYy'
      '4{<}}^Mda}-k(9Q7V(U<CWtQ!++vsb)W*dcOWuXGN_{?8ZVxdu0yoAw?0}(+ufFWhXS^c4e`OupM0^$OhhH{uCc-F}>|^^P`o>wAZJ`6~+GEwH'
      '4nK?>XNE^x`w8uHrIo&8`ZvT%TFNurU#Q^wK_0{LR9_IwnbgPkB7TMOB*L2Zh=nCR@4&G<TT8_HeUm>)$Y=6tJ;^}-Wqg~!YxW(;9!J=fI4c(R'
      'w{#LVBE|JwkI*rK)DKENr)&(sLZH_dIVMxYBFs6C1en0>AjTqn-thYYqxqKiw{0b+g?Pr|g<UvH#}V-Q{Pi<yU}Qb;N!SgDVUV&OK2MY4n}9o%'
      'e8PPPGv)<DLGL%V6?lGpE(E`VT!c?*_qkBwOqOHB8q5r5In%I4@Zr??Z`!{USF)uK-h99vaX!vik3i~Yss3v3r%CJ(>mWk+l5sc*{pI~{6LlYz'
      '_$S5zMZEDS_BgJY;*yBAa_^fz#X0tQy*_#W@jvSykQX831;sc1h^dmlgK@k)d<1);M^d~*>dU0Knwm>kDdI}kn+13e`4B)4Lie1)-l(U3a>t&L'
      'KRqD16ivv^{cX(4RkXP^C|>k^nT!W=*xm;*)<f7Q)huSxt31d>85c%+mN}y}c>B{q%<rHeVtAg~s<16XdSy)d<M$k~POL*B_82<74S6SG4Hf$}'
      '!&!6juF^jEU>^n6tS|K}iXRm^DxJAHV}$L+C|<E|XHBv`e8libCj;Lmd#FaSSipUEQ?}V=K=O>!2zCnSLbcCA;R6$PAjPKti0Mq}jFTUnZK;Uj'
      'u8B^GMxOB^aSl}s4!+Ds+r27z7vhj1vw^`qw5#fLJpa)|&(1jH7U_#tQP-AakHgmrqbRZv4~96Xq0oy*7oh_)c0ZB$q^je~xJmY#!k_SFNNk?$'
      '+iaZ<;(U>f%rZwjBk@i{#8@(x7%G2DTv49Q6ubP(*j&VXLI-Kr-j0YlCSPPE`;vM+;fv6#cW>}x;M~9lQRH#jx95{g8>-_zrm-i`_ot90^ep)&'
      'I`ywSp0NjQ?t4Lk%?#!_8cux<WJ}U`)Ahn$n3FH~!r(heuceqGjdj!(YXMzJi>AU>0iR;O1^JvPo)G-Pil%qAuUfp093RB{v*<K;(F*=v@TLlM'
      '$!Li_DK>04;pfk^w<2!o7AJp0|GC&X+w}IGJ_p;1|IXxJE<Sx6=j_H~VmD$wCG0~q6nrciW@AMBAJeZ%`p}AayS6q~I?i<fK1lJ&)zDciw6I^y'
      '^n!tmw@PC@2q*P9&Q{=H%@f4Ag_tqojm%@gC-t1oxZ;@hEN%m10zQOzk6I}ZJW7qN%*CNFAA$_*wCgVP?<;(gtQUNDpkt8UB;tL$S-i>YbRNLg'
      '#oXT~kYR~VM{Cu|#bBDr*@~px?GJSEKlyk20srlE>o~WF2}FNK$N=h}V<D`V<0Zv=v(r|_s5sr3?#aKmdHzK_1hO7jlk|y{Iu&3vhbiSz(EmiE'
      'GG1k8V1Mq*2KK!%sw%uG!4yA!b6;Kmql^ia@oRO_KjSp(G&d>j8yK-%Q>GQM_jZW5dBW~g{&Z3IRM;v#L+;O=eT2wm%HgE2H&!Wa)!9y&9JDaw'
      'rS%Z<IZS0CpVb7T+R3#+9LJ=l=oj5H*#)C{o%r?R_-HJmpG2o<S^}HW$>um+6#rIVVxoN|W>sL~$@62j*=V1j!;)XGwz241<uAug`fj>TytZ2D'
      '^IOnJK34vL=MSdQafVo}PPH5M<GSD(pgRv344*-5O}Z=5{yB-Y!u61_V|l)F{1rU=!B=~ezS{J>gLh4%Sq^JjXZH`Mtw*^`z{vai8L*+7vv|OV'
      'X5*jc;HrG3m|%8e=&0?<r<jj!;Y_cYkFbs*_{mYui|$hn$<w-xfO9>@V}p18viXbzd(yv!?y_M$cS`YhVtu3!f11)dY^}_1Q5z<@@AJK8Eq{-2'
      'w?%66P+WE(<|i;zk_8FAM*3!ouSvh~vn<;sILw*kma`YJ@|}k813W+93Lpbw&N&+27K8LW4~;f**<?&U>ndZ!_mg~FQEby<mtd_&kz=(}{-*1n'
      'Kb<e)&u4MAI9@@<>C(EsC9+^XnAReB%@DMi*bX@NljBih%xWLCcH&cdTyJHZgvqbAdI>a`$<JzDf5P}KympLP?$Y&;Ymww%5gnxXuG}xh4+Z_O'
      'Fims?jqUGAjr14B2C=6&+myH2(}r{83(o7jC(ug~n*!XR0=<U#@n1eSCI0cawnFwn56JB`=dvO;e9@XN3DYaiG?HbVMvwL%SSIor=C`u54^E&('
      'oR1RpYs%%@kX=B!X7(Oqz_5cBxqJuc!`B-yMBsPSCflHYHFi8t@%QjC%vt}=*6}AFAo^V)wqiL>=*BP!_gj)5P@HpMMtV7ohn2p?nZJv2G$b+a'
      '*;X%qlYhH;G*(GWJ<)}bvoCb&;a_6vBKF~lCKG)9oX{q0e2wCu9x?V*HVM4~_Ct-X8|_676VYB_^b~uyWcpkS8z{x}1F}IhkqaX61K7ujkC)|9'
      '-5?tHPJ9Zy2ATet(+McnXGyWo$Vuloo-D^GJs*xQbv<!D&cEGPvLB0V2iEBC%pRin&!MrzplJ??u1dI(<P!zbjpP`%;N2;1fwjB=f4c+AhmS><'
      'In8z~@%o&Ng4j|;j~t44CVA#p*uMnd=@);-4A{;Q^H1?WjPc0)2II=dOxN;#Oa#xjBxXY~4$to-*qTBo#3v|bPx2hL<5EA{hJE_4=#}mDOs1YW'
      'o_Gd*eDM(54L$p*&&<}-o@23T?U0Mn)R_m(z`qb*u3|O2AL2!Y-vBwgGR`;RSUJ$5Q_b>Efe&c&!VUSSJ<0*1{L^S5^M7HR3apXv#HR2c_4nO~'
      '{GX8h5ykWM4<_X}{>t(tKb8DQ^IhajPL*rJ_jE7ov_DxMCTt!VOPt3ma`~4<>%BA9r?BntcMm+WwWYW%df%m7edKq3Px5iCuW}&3r|E1V@66ty'
      'Pff;%gfh2Yu9X>DI<1G6!+F~cu~u$?durY)c!9_Ms;R(7#otHR<B^+-$u18;KOYXs0$N+dIt&>Tir%r0JH$n95JL_b8lJy%xt+tKiP&;r$Ru|*'
      'i05TrS>LDpBZ|?St}Pim-j}(Y;5EoGa4#)gQ|{?hPvYDg1Mj8z?x*j<zpfwHS3Q|tyA{;gSilEZ4bx|!zwKaqWBqPfoc1={!T&}67met((>!ev'
      '6MC2Npzo&e^&<wFuysv`EJLm#VjvghyC-r1sm24vEQH8&NBsH>`L^O6Tt6~CRVk}$f;b!-@oeGQiFIA%3W~YIj~^qim*W;X^A=(Iryl%o7%%t~'
      '$L9V5Mi<$@G!R>gGd+BF@^cBu$NGSUmT9kCU^t*pxmO$ZgCRBrv^g?J-_6xJU|vAyHjNLy;cWhT#Q6ua2=V{Mh5Roa|LqJ+6PV%J_vY8<b=->>'
      ')(&a{ICx%-H9$YmFTRd#KL5zG+7FLY@Pt)CybHZ0n?G!b!K8E+c@4^%Q;stGwGwPWyFq4!?dSEV(R$?Q5B7&@f6^SNHVDz_3cfVw(8c&S@Wp+-'
      'cVYt`QuplL;m<3!4xPJs1x;M!^SL?WjB~9Q&^OgjK&<<%u$<3|!@cp=_f6Dn`C50kL;8pHqPTVAmpwW9sh@#jO(?F^zZHQaV*ieQcRoXm@$E?d'
      'uVr)iZf@&=Glu^QveJP+2C+#y;P~&|XBrpvU@)iT*$L%ZK(4xImwckmUKz)fYaILwIc8^!KPKOUjeWE6>}29%%=Ke;Qr=K~6`n`+D@N^=d4K$-'
      'btfbEguZU_`l+)uOWjyI_wHXsr=sbd?eNx1{;$>WTy0G?Y`WE2+}TdwP@CpHu6k<QT*t-R)-;Q1zxY(|2G8T#%12)7lZ+yNNS^*|Jfp}rS`Us7'
      '0G+-$^V`|IAAR^aHLS&@gB|wbQn4GCz&G4WcZ<F%hIqcwi{M|mPg33oJOeqE$dv>xOujffv=?5*DXy_MUo@|{$N#cD!g{8*FJglt*qV6HL!Alu'
      'hJEv9&o%V33AjE`w&#?md4cR&NLm0UpyW^D%pey7I?bBqlswxd-h{Y}tGb8z{SP4*-oWQLf0_JyJ3QO##$QFO_w{kTm(|BT&``T4c-SkAYVH)U'
      '*F+O$Z(hSaQ|xht>-3;M?J3W8Tv?wxh~v(k#c`$^BX9qYEI)9V%Uwk-1Ji1|Vy+SAoX_87o9(5(!g^nt<FX}R#+2hY^7M55@%?;`OpedA(s-PN'
      '?)G0h#P}O#WxZR)YqE>}WseQmfhEn+>~IA>M!6T?<7sH+{b4;>|3l$Xs9&OCyt`5G$ZN;MgJVsHZ~6oo34FWtE0IY!(>m5{gT2wvzt1?k-(o(M'
      '$eiw-WX==ctmMZ~`an)Te8QffE98x>AY-k{=mTdh!1(i64s4TE0eKI_F7d^S6Z8V`?JfCHAPbQzB7b8ICXG6D*V)Eqy-Q2*pKV$L2DyQ=@fz_h'
      '+8^@XUBQC});<g@%6Ad|cGOZrPUc*#EAf`r9zfjAJA4o!`+b`;l66PQ1)#RA@d+^u6pwbj5|%$N%sloLQl5~cli?xgQoIOaH_nvvL%HejB_d9#'
      '208V%As?WL7@Tf3?6&C|;_aT^8yjI8XRQlgDV`&|^K3VJ3)B*#nhEsIWjQ_)y^uZ{4W>!l1LL&F)8IL7f1G1M%(4H#|EMmM&?nP+NGy-%1=7>j'
      '@e4go>L#~?*|0C9n?YxVo&&#GW4Lza=BYc*%XV`|{Bd@t^WGu-4|?=dySw!<;Xlx6PzTM0u7?;8_(@2wLaj5zo9HF{jeOhaJ-OGRJOKC#t@-uT'
      'Bbt@>p&voc;#@r5C+s@SGo>86FQR?a3>#i4A50y$Q%Q7^6ZW$15j~Jz;52rxbels5wV#j&6>{yZ#pu$3{b7?oajf5$#oehpLH|v<-uty0YqXb`'
      'ql@$Sd7?Zd=tih*L}LZc$XLr8a&23vPjos+*^IRgkh7<)L8q~mJahj#_nJK~o%=(doqH`^7q5Z73h)n1Kx>uHTUW?}l7o6PLtuigFX$FD{#W|('
      'qb>%_h~j@#xp)NsA>MI=9b%(4*LrP322tHM%q?&h-4kkQP)scByz(1ecbgdRxSZ(5Kp&NL%jtPo>xEb+=(H?DlldijIj!sCJXM|9oT8o(#o?k}'
      '6U`m?x;kG+=zrx6;!*N73U=hP<aasF{NcI${838>@`L;*eojswdW2i>`NR0nFV5J>sQ{Y*4Z*%Y(t1%J@VA3jj3Vrt>x2D5a)$WSrcJTO&~w;k'
      'AK%#T*@pk@fsg;|S^u4~{_ZuObR^U)E9<1+=jRuiq))COL$H6e&TB!VsEbt5Pj;nD`v*O;ex|XDn3=LqHL`3v7sAd-eycL6C}k4-A#aHJ96gSG'
      '{$D6l7i`0V7sH<&{3L&1&nmL&Ms|~K%{{8GL^eZG!|q$*$s!JG(Ixu{HT8Ox)s)}Eeo)@gNVh2WlJO0yp=bU=jPRA^$}7I!c#t#kE^kl6FPLC0'
      '#FrlFcv&@nv}uXY0++<z=#T|E^1cnwbopfY+U<eH9COF^z+KcC>=7=ohOULU-79o2<SEPZ@XGcheMF&;2sJPjT9CYYvHKleB==!UL&t7RU(n4i'
      't20Kiw!k_d$H<2EA9x!4Yx>3oo{o5H<Z@8WkJ66oHI5(7Jn7YY<UxXWKc90x2gMHQij6DQ@`hZ`@&R%KI&hiJN1U9G*StyUX8R4*Y>NeNOXb4_'
      'yw9h)O0a7Y?`rw4-+g|EV&tFfW!X+r1IB%jcP#S{y{k2UiGjp)Za9Y}PZ#m;_#om<AxCGL6PJriu8&83P;J5zVvUg_@PGx(AcqKNQ}|f?10Q?;'
      'j~)O2)sC-~%qr?^!k+KNS`q*IFYu%Bv+c3=>D<1MT~vdUaCOw%EaP10o7KbqN;(t#wDL?ZZUeenrF?aL)U5sAkS*4tOuAIezN@h=_Um_gqkm<)'
      'Goj@l=C-lO$I_(u|1#ovaK<a;m~C`A=MCtu#3RNU^cB<r1=i{|ZY;N-be_jOZIb*s87a-@F#dWP!@|BwRp!iv&mDZva2i>B3gRZl?FyfJjwMa7'
      'hQS;)mXJdY%r~0)(thF`$6iEhHfsc^!Dv9<Q%%h@_Epvxm9$(<>kg3*AgjMNR{DAH0k%ob<>F^4HvsL5G2*PvHg2H$470Y_?kWyhKSA6h`GI5|'
      '&{+WA0OV_zZ6a09_7BLT$k%Ue3ccaP_0uMTR^EXp_#$o&=V2d2v^P1qUOyqOH}D1v#5;%_oR4hX<Tx5#k#k(D6Afyd>(wNC=!O^<)K?X`%0<fQ'
      'almh8vli??=qyF#8txId7qgBad{KT$)<`WG9q7x;`Il<5okh)|9{Fpx%R141{e<{4#ErvGz;a3#@rbC+02z?fN4--`2*gK%CQAqQBNcn7?F!yj'
      '4Jocj{M{rQcm#g*h8!odeM|p>p9px$E_Z#&dgd<TJ>4S~7q*{^cqU;VmpDdV(kEivreg089LtILs*GM1_Y>@b;hN&19%>m&9b=lr7;V*Bp@=uz'
      'z~11x8;Cnn>aAq8k%=}^tblUc)#rlKUDPLCZzdI*OExCTtH5|R3dX(VxD?cq%GdYI&kH$`X&de~oj=vrQSEV#|2F46^1*TZKkJ^THIa?mOlwQS'
      'ca`E8<gY^h^%iSO`!Cj;>&@{PlKv3e(&M^L9`efOMH?{}Uk2t#HSdZ8*gT#(hvpag#ft+isqwZcw#RxOxjjB%ZK%cd?%a{X9&5M$wr(i<x>0;`'
      '$+e59;oJ?a8P`DDQk|%jb`KHp*9W|^&M`k!*XA+qXI$VIoa7AR+>ZDRnxggt={FALgrJ|dHrGD6ITT;azk!#4$B_??-{HDt(uP;gG3xB;#1o=N'
      'o*-pC;f?Q0s+S>o>UVDKz>@{<in+h4tAhtoT(-bC5JTd<)P?GAorQ}U<m8`&4{KgWeT~nx%ko3o$*e=_nEN%cne+Bza3UQ5I<WNuhsd=dY(>(!'
      '=K3AApvk7dJ}3I=vs@k`HfmL54EddMG@(NtITvcm7+ryQb-QVWzmb!f{hGJ^pXX6;1rkpX{Y<zI;0abHE9xY>kxN@i^Ix9z3B(M`{E>+8K+4S_'
      'tcr6Ka<ca5i`o{g#0F3EY%mVdKJAk@mk&4=W0#~KNn2+FJM(GoMS&&VfZ>Eee_M-$4lp~S1|!oK<6#{C2)m<DYO#G?<#~Vq^}MAHJma{B%b@Tx'
      'opH@-*ddT5yAH|AW|y!P-goF$A><t7hE8$aG4Wex66T+MK^rro4T;}6$nk}qM|Njmt2XhWOLLmYd$lFW*ezs9j5>#tl0z~axupdee}et@X#Yw6'
      '!7u+Axw2M0Y-F}|y?LEz;!Zxxq>jC6n_E&}5;ocoXKt~U_Q#9;1YQT<0eI7!+}FSuQ%+iC)1kfPeOB@vIcFfvn{1WFbLtoQ*cIf(O``VmcLYA('
      'zl*ES)j!=xd`Q5*z9&xXPBq-vbRZ8WC=P1|_eem)po0=X1ftXRk5p-0NC(mOOSvYBOZ7=Hg>xH)f|7T^KyE1Hd_-`;Ea(+x{`Vwxgd>Tiz9(g8'
      '6>BH2>7Fc+pbq8k2<J)55D9-Lc5HX%sUzwJ=iCRz5m@^ZoZ7kogBJ|^pRkz!@~D_T0ofHiheG5lUq=xnQvjEfqt=g+ypM!V6sDC2WQZa`bzipe'
      '-lKx3NZf}(4TcsV;3d?T05b{f!!uM&WDmop1j7f$l8$?1T$WM9X?k*tTukcJsfmv+Ob~1}mDffLa@;_e9)kGFv+?*g#)kor4DcC^8?y+;H<az('
      'hJst51Vl)5M{)dXG@-!#xIE$hn+V7~bP4c=B4tJ9jtcD|F(EPTYBPLHAw+P))d#U25Gc5wO?L<=)i29EK;wEZunQ<QTc5SedJo0)p(LXCg6yNp'
      '`{kGG2(bFs5)2A^yhHHQi3*>SgAPG65Bm{<csH~r2nyQ7t3tX@qF6o!o*+SS$w@9lg=SFT8^S>vqBpm>F`&fxVDLqH7DCYxz$ly@i|FKmc9tlT'
      'kM|ffLK|DquTWUG;X?p}p%H+Fps2t7?DrhO1&uY7gEb|0BZzBFNyG(89uf`)qMdB6eR&<@g5tW-vN?_ni<JhVd*%7pV^P*FF}h2K)l=J~$kQ36'
      '^8}Epaq^K^0eaN06i{R@UeYV*0E)Z-ZKJ7X_vm*Ag`vT?z?@7Jw?QBsXb(khpg{L79WE+Cdse@9@R1C1CQ5*RP0xVgR$;gkAQ&@^qBxb($43I<'
      'i8BQtVWHn03bbg-@yYYmuYMtU2p_s2XY!)_@Sdx;CQoM`2_N_%3HG0i(LNc36i9-ksXhVq^bv_dJ))ovJP)OgiayM^4ep0|8#xX%1U<dQy*Wjc'
      '(ML{BWI)q39m<J8JOK3eA%l3xFd~LaG@pWjmT;86r$IH3_Zn0f141GF8^zL)IE?_TGmJP2x&>{KqQDNzYYO;QZp_E1n5Ol-pHFlrc^QFi!Oyff'
      '{NO^CV?70UlyN`PXhA1oA3{N)_1@SG7EJF*icB0Ysc>9LdS>AB(Bq<WeGWcQ=376B!l-izTn1!Bi-N^aE+}x<OP`^A+T-ILbzSA{Y*b=RAl!QZ'
      ')?_=|lW$Q{aZxzySYhA*x4c89|KPZyU`ux9g5Ka9ewVUgPX!FzB02EA4+<jYHJgO2Jbq7(C-B1e&SDmJ<^el%!Lx{uL5U`X5ymru?+D=e#>ZbA'
      'IHCbA-dDe)%gOOwnJ4<DVud=!9-KSLdO=S2eZ7dH?!T~?lz2-90b~pnicI#T^ntyW`%a3yVWewZ0H6V*b)oXlITvOnqa8+}3|K?(2$UtrZGbLZ'
      'q!6HX!QSJ2e3V9ui8uGRM}-GV<G-TAQA!_t5y8`|I2&sY0Gr75$<LV#902ZA{Lt6{R7>acH};>;hstwNG!6Syr$Ai_a-)KZ2(IK0{Yc<TXN7uM'
      '#0N8q#6w3g&PF@%_?LVifiE!y)qq#VRMbQA*-Q>LY5X_~R9L^6laE&_M9DhTwhTaQcAz3}tNeMoP?2i<qmPZ~P-uCai4&CRbMkYWV$X=i>tqZH'
      'zIn*#4FxWk&(NL;+AT)}b_4v6SoSvQ?2=+X+x+y*s$dX6P9F(+?@mss9@mF{(k;~YSs$S&UOC?g8RaW{4EvooJX$zkW&YUB3cM@T5BzS)D%2p)'
      'Y?XJ!R4_i{9r_0wB?UU?<qLo4XwfqI`^+i3qH|2Z1i;KBAQj_0*)S9kG!?M}fBp#tyiVtsjBy>t;{|+qFA@q!scuoy;zfWS2qJ;QmgWE+Ea(S?'
      ')2X;H>1kwOf>#_M$H7}hJB$Gd?kKEJ!B+^(SrrI)&U75Xt5dy6@R#usdax{_G=^S2w`pD<2+q~uyiR3{kg-%yRtMORBF>4<oax}aU!boO))oO!'
      '(FwXzF>z8mT%0NYuH@YjIaJ{IdZ!}pvR_K(C8x1^?G6A4?X6#2%e{Li8kBwAlD^ymC6Z0W*e1!?$XUbBCGV2sho4E`13yGDpVJBfo{oCG&?EYY'
      'Fz+)Nlr95{NW{l^KoH%Ul5AH1lIcTO$vd=`pc?>=YCABTu_xd^?^J9>8OxarF380#<gnlaO8*pWMRL@NiDx4KiO%Nhcd`BgL|5)J2ukuq@Yu)e'
      '#@?nx%ZzrATvUEb$IOL=@xigZ4H-I2brQ4E+uM8Vp{?YMfqxauV{fv<`gi$5eF$*8n3Xqt2qD=I4q{7re<t6K95iQEy5yX?K8yniX>|gX-`N3G'
      '`P$Mm8U6jH4FmyGG1_FD92oJc9SRB~;3E4ioF0u56)sV(bHFzA)}Feq9FQ$(P<eI(KtrYV<T}ZDd|)m|Cv^hd^3Rbi*a$XCKRZesSLd9aQwmap'
      'eh-HY1#yFyzP@*8u`EyM_~kPK+y~$@RM&%a9nn7p0WAV?4kxeMSOT2tP;ntns{i4cd5L+QGC!erQILNHA5tT_r}Opt{@r3Q`semRV{k6i-p!_#'
      '({r+C%Kzbm|KWrG;e-F-ga7yN!F@LV)nD<ps>0jmDc+i%!3SO>0u>4r6fw=eo9Bxql`m4k5aNlsYY6o49(b?3mVY<lb5)hk^{6nR<gv9puTyCZ'
      ';JSDw<&X4n&|_zTk6Z%7QRof@2XW@H##E4$aA@K+=~>`pJAmzTvNw{oqO}AsonO^<2&_L+(C-TL?%ZGUH`sfgP6=EYt&!7`@UU(smwtE`>o0-;'
      '6nd9@1Hp~pan+~mDPQws5RO?zFjw^l9+mR9uPxr&KeUzDZwO-X{C0JUfScN#lR~Q2x)URK7mmdrc<*ocnixmIzxpB}V{P8)8AU$+9p0FX6NQw1'
      'Xvc?BEmqGN2P!m2XTn^R>s)vjYkpVP6?oBRC23H3_DIPF3auyCu`gs(4^=wZcp~UoeGYq64lU(uC+9w!^DEx;#K+ZpKk#(*nV0-s<vTy{dbN$8'
      '_`dqgvkXvqmtxQVz}sHP3$^`Jc9abin#4zXjfjd+u2j5Nx&F<;t1`H2ak}Co!J)*`mFpUPWCYIM@(2A7%;dUiYG*+!LpZ)qvBZ_nJ^FYNo(GR5'
      '+~Gv;B!4IGrOvxzipd~ob+%7z_!Kww^*C?amr^J6DtiX_shwhV2mtjvllli?SM;G{;^mBYOpfR7c19o4HMk%qVUjs)E4ikf+0ouq?~%iMgZC%^'
      'EQevRgB9aKvcW9F4YxI=FA7vGtM}-m+$X%lPGt%h68=B+IxzDjP>}F-xu!|*z4}hKg5a-%OIY1odR@ghDL7%CCf%m(N~{*^zgGN&C#e^P`RlZ2'
      'N|3_?4x@Dsa?f7`>HQAFgzpeEe|&wfkjGXf&U_Y*aOQ1y+H3in0wS(b?$SNTlG2@g8`K7Z(|}77E=*%zUpv)0(#t5|AB75l{|P7gN&Y>e;2kMn'
      '!lW(iB?ms7IKVY|c0oGP*SoleGx0-kAkJST0%S$t1`?q~;Ko5aC(lxmHDHb(;HRLKx1^1C8TcU_`_1h6wC#ex!(X*6f;3tmHYW?d>5|4o&_MG<'
      '|G#%7eJMfPe~|A10fSDGTn*>`t81x#ngnmJ##9J|_<T?3IhpU8gDVsIBOTI0V87d3uiFU7!ko8lB#d^(WlHit{#Cne{<xj$PiJ4$Pi_B=_E0oO'
      '35Et%OZqAH7xH(%!Sh&ed<a-1NvZb`bkqpolL|!fdA4^&AT+O)qzMlJ3iN?3mp@8`b5SIk0Wan2Q>p7PeRFUR_D6nzjquPLo`a56(Zw|c(ku5c'
      'ZZ*NmQ~bd#=nsJoT|ec?xq#bd=4V5nSeF^({oB2FPxzKx-$V9X%99r5HyYzpz;5&J`H|$iX(CXjm%BOn2b}ze_F4zaU&k^yo03p<B;z8nhLQ%?'
      'iA*G&E5Y2UXy2LIC!GcB!UYI8aq95c2a4iwA_xaqqdk`Uu<yB`Ecx_haPUo~E9DtFr=r*)@fIpvnFaEfKVCe~&#<03Det=|Ql#ov%2=U)|1fVV'
      '#Pi}Emp=O4B)wf-7bT%>P9JWeB4y>D=7Rb1*|_{>4jSqBW?_1@x2?S`*IG@2TRkSd$?J4ZK&ygBH%ayeWPgxe8gzgsbirpukbbw+Ep&BzKf-vv'
      'Nax6WI2m8#<P60(B=rh;1}51%q-V1{R^P?^)OT6e@umD#-+iT194O&!L<JdkWDkM9g<i5Hn%C6piLP`@`lY1Z7h_KEQ_#G6U+P`&iQJX`^~f&f'
      ';{u>N+{E~_C%-lB6%)Sjlq{6S7kEj$Kmq$sbns9#Bc+9eH>N%=vPTpA=(UZMqq)cT`Mf5<+lb!~!F32$RNi6RrJa`-Is4_|US`8oc55m|)2=+*'
      'F%R$LY?+69i9HkM&X;1-JUlP#n(;&5dE16~hQt<xuHKp5oLoY-jO#$Htq+N_3Vq$zxyUNvD&$+s@iajXkI$oho#{yl(tU?d&?oyFu`2Hrm(s4h'
      'X(+TFq1Z#zr2=XBHg>`mVLSZXoy_hSe|F8+T@=c6JKNFy#dve?BH5K>j{Nw^^_^^l?(BN*rSWaEk4#SupZ3)jA9#AOCom>_9+J+{vrUoRV|gU{'
      'aAgsiG@ef+1c<m5b>GslAbB9OHBbl!&ta{8?lZLuvbjDLxvBm;b53z5k^GmJpGiy#W3-U9MVvtt>G%R)SpTgvn5@I18m4^`O)e^)+tIIw1wPQ8'
      'uGP!9!%x5e+cgYGw+^9;X^U*`Zl-&MqUaBFeo+kj-=Yb{juY`T9hsQNeJOfKlCbyxmp&qr2U{*alFr40Wt@yimOlI@Mh|u`#l7U;O?1=|ih~$w'
      'GP!J5iPNJs$o%|r9rC(dHOz_|{e8VId#9e#bruhmk1cqJWgI<@NvuNvUI0BpEy|Q7Rx0f&cBbq%7fXY)NpUzyEXZH2lZdUE;$zX!ldSRA5k4QB'
      'Rm4gpc$!bXs7=7Z`63A|T*W0$++0+DOl1Avwy7Rz@G*8y$X81<lYf=Zyo^Z;Cp^ekO1#ytKPJ#U;pLC=J-wbg2T0laO8z~bfy8D@F@Cna_Oym4'
      'my=$Zw{&m)YJJn4w_J*~dU1Ya|2ess=XHD*E&OvI`5e*u&;4h5u1iH09`)ALCzj~Cllta+obLUjzAxH%p__9-he?7To1&Ny71P;HdH*`wTKIpQ'
      '9gwV3-X8dIj(<7U%G*t)!MqGa(r)wLkZUMJg?8^KP8E0W)4!)%=KGcQ#_-?jw?6@WO)Q<})P1D&$M?CoL`o-eOv-tyX6H?gN%ezev0dVf-Fn+D'
      '^?k=VD4?j*zteZhS96Ky^(!Tr%VHa?AI{b>wNblsBzh@45<a1Gn#<th?2~T=#Uzo~c0!-|M3lox{AL`hEaC+C9%2rW%$?YVI2RGMOW0!@I%pSk'
      '${!^j9{&d=Wr%Ey&1oM{7RI1buT8YxbUrKB%?5uG>5t0x#-B|>8EZs5x+u4ZGJ#uUmrW=|yS^K&k1hNSuhM9P@&`w(5P66UGnh?Bl&TC*wtR-O'
      '4@^ooLVD?vM(d;O2}y=6%6b)$t_81=8ykBIT${9#&bC{$?&V!?-;J>f^r@}2{A8amBWF1+y<OY@)i&3iJH|EKeZm8#S>rN)E{Z-YPbum6>?Bfp'
      'kOrM<J4kUAXbe8xF)PsLy_41N&}Yy0WIfcQ#6yU3#OJQkhb)ou2P|MpwFOs0nTiTN1+f^xvn!sVNQmX4tckK%C_TT22d#++%`wISO#tapGOZQq'
      'k;N811!i6-MF(qn?~vz?(9tbU<aV^{-cinVAKtO{pZM$;yo|xstIGL8>Mnt**!YB1^7(j<oLCHf>?5vQhUpuWj)ln3?nQh?Z+-4o-GaRZ(ng;u'
      'FZS+YU#}Nv6C1nAd+Z~+ALWHa>JZj4E3Gh#w%y74W=-n;;-QolTZ3bbK9T0jy{EK1q^5!{L|V`vSSz~r2da~_@b9w&G}3PFKyT=uvTV@*KmOPM'
      '3*+Z}_d1pU00'
)


def _load_server_static() -> dict[str, bytes]:
    import base64
    data = json.loads(gzip.decompress(base64.b85decode(_EMBEDDED_SERVER_STATIC_B85)))
    return {k: base64.b64decode(v) for k, v in data.items()}


def _run_web_ui(locs_for_paths: list[LOCForPath]) -> None:
    """Port of cmd/ghloc runServer: static web UI (bundled in the script, like
    go:embed) plus the /api endpoint, on a random port, opened in a browser."""
    static_files = _load_server_static()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            target, _, raw_query = self.path.partition("?")
            if target == "/api":
                query = urllib.parse.parse_qs(raw_query, keep_blank_values=True)
                stat_tree = build_stat_tree(
                    locs_for_paths, filter=(query.get("filter") or [None])[0],
                    matcher=(query.get("match") or [None])[0],
                )
                body = to_pretty_json(sorted_stat_json(stat_tree)).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.serve_static(target)

        def serve_static(self, target: str) -> None:
            rel = urllib.parse.unquote(target.lstrip("/"))
            if rel == "":
                rel = "index.html"
            body = static_files.get(rel)
            if body is None:
                self.send_response(404)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", "19")
                self.end_headers()
                self.wfile.write(b"404 page not found\n")
                return
            content_type = mimetypes.guess_type(rel)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # silence stderr logs
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    url = f"http://{httpd.server_address[0]}:{httpd.server_port}"
    print("Web UI:", url, flush=True)
    print("API:   ", url + "/api", flush=True)
    webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def _ghloc_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="rewrite.py [ghloc]",
        description="Count lines of code in the current directory.",
    )
    parser.add_argument(
        "-m",
        metavar="match expression",
        help="sets match expression used to filter results, sets flag -c implicitly",
    )
    parser.add_argument("-c", action="store_true", help="output to console")
    args = parser.parse_args(argv)

    if args.m is not None:
        _print_stat_table(_count_locs_for_paths(), args.m)
    elif args.c:
        _print_stat_table(_count_locs_for_paths(), "")
    else:
        _run_web_ui(_count_locs_for_paths())
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in ("serve", "ghloc"):
        mode, rest = argv[0], argv[1:]
    else:
        mode, rest = "ghloc", argv  # Go ships two binaries; here the default is ghloc
    if mode == "serve":
        return _server_main()
    return _ghloc_main(rest)


if __name__ == "__main__":
    sys.exit(main())


