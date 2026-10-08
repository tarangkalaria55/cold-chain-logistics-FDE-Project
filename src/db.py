"""Shared SQL Server engine factory.

Builds the connection through ``URL.create`` so credentials containing
characters such as ``;`` or ``}`` can't corrupt a hand-built ODBC string, and
gives every connection a query timeout so one runaway query can't hang a worker.
"""
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import URL

from src.config import settings

__all__ = ["DEFAULT_QUERY_TIMEOUT_SECONDS", "make_engine"]

DEFAULT_QUERY_TIMEOUT_SECONDS = 30


def make_engine(
    user: str | None,
    password: str | None,
    *,
    fast_executemany: bool = False,
    query_timeout: int | None = DEFAULT_QUERY_TIMEOUT_SECONDS,
) -> Engine:
    """Create a pooled engine against the ``master`` database for the given credentials.

    ``query_timeout`` is in seconds; pass ``None`` for long-running bulk loads.
    """
    if not user or not password:
        raise ValueError("SQL credentials are missing; check the .env file.")

    url = URL.create(
        "mssql+pyodbc",
        username=user,
        password=password,
        host=settings.sql_server_host,
        port=settings.sql_server_port,
        database="master",
        query={
            "driver": "ODBC Driver 18 for SQL Server",
            "Encrypt": "yes" if settings.sql_encrypt else "no",
            "TrustServerCertificate": "yes",
        },
    )
    engine = create_engine(url, pool_pre_ping=True, fast_executemany=fast_executemany)

    if query_timeout is not None:
        timeout = query_timeout

        @event.listens_for(engine, "connect")
        def _set_query_timeout(dbapi_connection: Any, _connection_record: Any) -> None:
            dbapi_connection.timeout = timeout

    return engine
