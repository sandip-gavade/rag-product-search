from django.db import migrations

from pgvector.django import VectorExtension


class Migration(migrations.Migration):
    """Enables the `vector` Postgres extension.

    Must run before any migration that creates a VectorField column —
    the `vector` type doesn't exist in Postgres until this extension is
    installed, so this has to be its own migration ahead of 0002_initial.
    """

    initial = True

    dependencies = []

    operations = [
        VectorExtension(),
    ]
