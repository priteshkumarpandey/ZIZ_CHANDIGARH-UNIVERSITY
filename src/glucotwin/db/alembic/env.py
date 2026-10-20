import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection

from src.glucotwin.config.settings import settings
from src.glucotwin.db.session import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = settings.DATABASE_URL
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    import os
    from pathlib import Path

    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    db_url = os.environ.get("DATABASE_URL", settings.DATABASE_URL)
    use_sqlite = os.environ.get("USE_SQLITE_FALLBACK", "").lower() in ("true", "1")

    if use_sqlite or "postgresql" not in db_url:
        sqlite_dir = Path("data")
        sqlite_dir.mkdir(parents=True, exist_ok=True)
        sqlite_path = sqlite_dir / "glucotwin_twin.db"
        db_url = f"sqlite+aiosqlite:///{sqlite_path.absolute()}"

    connectable = create_async_engine(db_url, poolclass=NullPool)

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(asyncio.run, run_async_migrations()).result()
    else:
        asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
