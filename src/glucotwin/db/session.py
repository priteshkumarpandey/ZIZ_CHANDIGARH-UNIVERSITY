import os
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from src.glucotwin.config.settings import settings
from src.glucotwin.utils.logging import logger


class Base(DeclarativeBase):
    pass


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is not None:
        return _engine

    db_url = os.environ.get("DATABASE_URL", settings.DATABASE_URL)
    use_sqlite = os.environ.get("USE_SQLITE_FALLBACK", "").lower() in ("true", "1")

    # SQLite fallback URL
    sqlite_dir = Path("data")
    sqlite_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = sqlite_dir / "glucotwin_twin.db"
    sqlite_url = f"sqlite+aiosqlite:///{sqlite_path.absolute()}"

    if not use_sqlite and "postgresql" in db_url:
        _engine = create_async_engine(db_url, echo=False, pool_pre_ping=True)
        logger.info(f"Using PostgreSQL database: {db_url}")
        return _engine

    logger.info(f"Using SQLite database fallback: {sqlite_url}")
    _engine = create_async_engine(sqlite_url, echo=False)
    return _engine


def get_session_factory():
    global _session_factory
    if _session_factory is not None:
        return _session_factory
    engine = get_engine()
    _session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return _session_factory


class _AsyncSessionLocalProxy:
    def __call__(self, *args, **kwargs):
        factory = get_session_factory()
        return factory(*args, **kwargs)


class _EngineProxy:
    def __getattr__(self, name: str):
        eng = get_engine()
        return getattr(eng, name)


engine = _EngineProxy()
AsyncSessionLocal = _AsyncSessionLocalProxy()


async def init_db() -> None:
    # Creates all tables in database
    eng = get_engine()
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db_session():
    session_factory = get_session_factory()
    async with session_factory() as session:
        yield session
