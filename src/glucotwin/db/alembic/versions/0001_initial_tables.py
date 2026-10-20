"""Initial database tables migration

Revision ID: 0001_initial_tables
Revises: None
Create Date: 2026-10-11
"""

from collections.abc import Sequence

from alembic import op

from src.glucotwin.db.session import Base

revision: str = "0001_initial_tables"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
