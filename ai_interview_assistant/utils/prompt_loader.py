from ai_interview_assistant.utils.config_handler import prompts_conf
from ai_interview_assistant.utils.path_tool import get_abs_path
from ai_interview_assistant.utils.logger_handler import logger

def _load_prompt_by_key(config_key: str, fn_name: str) -> str:
    """
    按 prompts.yml 中的键名读取提示词文本。
    """
    try:
        prompt_rel_path = prompts_conf[config_key]
    except KeyError as e:
        logger.error(f"[{fn_name}]在 prompts.yml 中缺少配置项: {config_key}")
        raise e

    prompt_abs_path = get_abs_path(prompt_rel_path)

    try:
        with open(prompt_abs_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"[{fn_name}]读取提示词失败: path={prompt_abs_path}, err={str(e)}")
        raise e


def load_system_prompts() -> str:
    return _load_prompt_by_key("main_prompt_path", "load_system_prompts")


def load_rag_query_prompt() -> str:
    return _load_prompt_by_key("rag_query_prompt_path", "load_rag_query_prompt")


def load_resume_parse_prompt() -> str:
    return _load_prompt_by_key("resume_parse_prompt_path", "load_resume_parse_prompt")


def load_jd_parse_prompt() -> str:
    return _load_prompt_by_key("jd_parse_prompt_path", "load_jd_parse_prompt")


def load_question_generation_prompt() -> str:
    return _load_prompt_by_key(
        "question_generation_prompt_path",
        "load_question_generation_prompt",
    )


def load_answer_scoring_prompt() -> str:
    return _load_prompt_by_key(
        "answer_scoring_prompt_path",
        "load_answer_scoring_prompt",
    )


def load_followup_prompt() -> str:
    return _load_prompt_by_key("followup_prompt_path", "load_followup_prompt")


def load_reference_answer_prompt() -> str:
    return _load_prompt_by_key(
        "reference_answer_prompt_path",
        "load_reference_answer_prompt",
    )


def load_query_rewrite_prompt() -> str:
    return _load_prompt_by_key(
        "query_rewrite_prompt_path",
        "load_query_rewrite_prompt",
    )


def load_intent_classify_prompt() -> str:
    return _load_prompt_by_key(
        "intent_classify_prompt_path",
        "load_intent_classify_prompt",
    )


def load_mock_interview_prompt() -> str:
    return _load_prompt_by_key(
        "mock_interview_prompt_path",
        "load_mock_interview_prompt",
    )


def load_final_evaluation_prompt() -> str:
    return _load_prompt_by_key(
        "final_evaluation_prompt_path",
        "load_final_evaluation_prompt",
    )


# 兼容旧调用名：后续可逐步替换为 load_rag_query_prompt
def load_rag_prompts() -> str:
    return load_rag_query_prompt()