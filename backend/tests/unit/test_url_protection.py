"""Tests for URL protection utility."""

import pytest

from backend.src.utils.url_protection import protect_urls, restore_urls


def test_protect_single_url():
    text = "Check out https://example.com/page for more info"
    cleaned, url_map = protect_urls(text)

    assert "https://example.com/page" not in cleaned
    assert "{{URL_1}}" in cleaned
    assert url_map["{{URL_1}}"] == "https://example.com/page"


def test_protect_multiple_urls():
    text = "See https://foo.com and https://bar.com for details"
    cleaned, url_map = protect_urls(text)

    assert "https://foo.com" not in cleaned
    assert "https://bar.com" not in cleaned
    assert len(url_map) == 2


def test_restore_urls():
    url_map = {"{{URL_1}}": "https://example.com", "{{URL_2}}": "https://foo.com"}
    text = "Visit {{URL_1}} and {{URL_2}}"
    restored = restore_urls(text, url_map)

    assert restored == "Visit https://example.com and https://foo.com"


def test_duplicate_url():
    text = "https://example.com appears twice: https://example.com"
    cleaned, url_map = protect_urls(text)

    assert len(url_map) == 1
    assert cleaned.count("{{URL_1}}") == 2


def test_no_urls():
    text = "No URLs here at all"
    cleaned, url_map = protect_urls(text)

    assert cleaned == text
    assert len(url_map) == 0


def test_extend_existing_map():
    existing_map = {"{{URL_1}}": "https://existing.com"}
    text = "New: https://new.com"
    cleaned, url_map = protect_urls(text, existing_map)

    assert "{{URL_2}}" in cleaned
    assert url_map["{{URL_1}}"] == "https://existing.com"
    assert url_map["{{URL_2}}"] == "https://new.com"
