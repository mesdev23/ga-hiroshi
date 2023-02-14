#!/usr/bin/env python

def init_django():
    import django
    from django.conf import settings
    import os

    if settings.configured:
        return

    # Build paths inside the project like this: os.path.join(BASE_DIR, ...)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    settings.configure(
        INSTALLED_APPS=[
            'db',
        ],
        # DATABASES = {
        #     "default": {
        #         "ENGINE": "django.db.backends.sqlite3",
        #         "NAME": os.path.join(BASE_DIR, "db.sqlite3"),
        #     }
        # },

        DATABASES = {'default': {'ENGINE': 'django.db.backends.postgresql',
                             'NAME': 'postgres16',
                             'USER': 'pso-root',
                             'PASSWORD': 'Root_1421',
                             'HOST': '34.72.142.21',
                             'PORT': 5432,
                            }
                }
    )

    django.setup()


if __name__ == "__main__":
    from django.core.management import execute_from_command_line

    init_django()
    execute_from_command_line()