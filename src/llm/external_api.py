from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat
from src.llm.render import load_template, map_content_params_to_template, map_test_params_to_template
try: 
    chat = GigaChat(
        credentials="",
        verify_ssl_certs=False,
        timeout=30
    )
except Exception as e:
    raise 






