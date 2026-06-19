import os
import pytest

# Required by protect_app — must be set before app is imported
os.environ.setdefault("JWT_SECRET", "test-jwt-secret-for-pytest-32bytes!!")
os.environ.setdefault("AUTH_LOGIN_URL", "https://apps.streamflows.org/login")
os.environ.setdefault("AUTH_PORTAL_URL", "https://apps.streamflows.org/")
os.environ.setdefault("USE_AUTH", "")
os.environ.setdefault("API_SERVER", "https://wcc.sc.egov.usda.gov/awdbRestApi")


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: marks tests that require network access")
