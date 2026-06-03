from forge.llm import _chat_url


def test_chat_url_adds_openai_compatible_v1_path():
    assert _chat_url("https://vip.yi-zhan.top") == "https://vip.yi-zhan.top/v1/chat/completions"


def test_chat_url_preserves_v1_base_url():
    assert _chat_url("https://vip.yi-zhan.top/v1") == "https://vip.yi-zhan.top/v1/chat/completions"


def test_chat_url_preserves_full_chat_completions_url():
    full_url = "https://vip.yi-zhan.top/v1/chat/completions"
    assert _chat_url(full_url) == full_url
