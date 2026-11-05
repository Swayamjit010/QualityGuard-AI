"""username for self-registration + access requests (users ask an admin for the Inspector role)

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-05 18:10:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.add_column(sa.Column('username', sa.String(length=64), nullable=True))
        batch_op.create_index(batch_op.f('ix_users_username'), ['username'], unique=True)

    op.create_table('access_requests',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('requested_role', sa.String(length=32), nullable=False),
    sa.Column('reason', sa.Text(), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('decided_at', sa.DateTime(), nullable=True),
    sa.Column('decided_by', sa.Integer(), nullable=True),
    sa.Column('decision_note', sa.Text(), nullable=False),
    sa.ForeignKeyConstraint(['decided_by'], ['users.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('access_requests', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_access_requests_created_at'), ['created_at'], unique=False)
        batch_op.create_index(batch_op.f('ix_access_requests_status'), ['status'], unique=False)
        batch_op.create_index(batch_op.f('ix_access_requests_user_id'), ['user_id'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('access_requests', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_access_requests_user_id'))
        batch_op.drop_index(batch_op.f('ix_access_requests_status'))
        batch_op.drop_index(batch_op.f('ix_access_requests_created_at'))
    op.drop_table('access_requests')

    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_users_username'))
        batch_op.drop_column('username')
