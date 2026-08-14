import argparse
import os
import time
import pandas as pd
from openai import OpenAI
import openai
import sys

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

print("程序启动成功", flush=True)


def _get_token_encoder():
    """Best-effort tokenizer for local enforcement.

    NOTE: Tokenization will not exactly match Claude/Yunwu, but it gives a
    consistent local cap so your saved Excel never exceeds --max_tokens.
    """

    try:
        import tiktoken  # type: ignore

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _count_tokens(text: str, encoder) -> int:
    if not text:
        return 0
    if encoder is not None:
        return len(encoder.encode(text))
    # Fallback (rough): treat "tokens" as words
    return len(text.split())


def _truncate_to_max_tokens(text: str, max_tokens: int, encoder):
    """Return (truncated_text, token_count, was_truncated)."""

    if max_tokens <= 0:
        return "", 0, bool(text)
    if not text:
        return "", 0, False

    if encoder is None:
        parts = text.split()
        if len(parts) <= max_tokens:
            return text, len(parts), False
        return " ".join(parts[:max_tokens]), max_tokens, True

    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text, len(tokens), False

    truncated = encoder.decode(tokens[:max_tokens]).strip()
    return truncated, max_tokens, True


def _normalize_col(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def resolve_question_column(df: pd.DataFrame) -> str:
    normalized_map = {_normalize_col(c): c for c in df.columns}
    question_col = next(
        (normalized_map[k] for k in ["question", "questions", "prompt", "query"] if k in normalized_map),
        None,
    )
    if question_col is None:
        raise ValueError(f"输入文件缺少问题列。检测到的列: {list(df.columns)}")
    return question_col


OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
SYSTEM_PROMPT = "Answer the question below:"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_API_KEY = os.environ.get(DEFAULT_API_KEY_ENV) or ""
DEFAULT_TIMEOUT = 3000
DEFAULT_INPUT = "examples/1487_questions.xlsx"
DEFAULT_OUTPUT = "output/output_qwen3.5_1487.xlsx"
DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_BASE_URL = OLLAMA_BASE_URL
MAX_TOKENS = 8192


def _extract_answer(resp):
    if not resp or not getattr(resp, "choices", None):
        return ""
    choice = resp.choices[0]
    # Support both object-like and dict-like choice bodies.
    message = None
    if hasattr(choice, "message"):
        message = choice.message
    elif isinstance(choice, dict):
        message = choice.get("message")
    if not message:
        return ""
    if hasattr(message, "content"):
        return (message.content or "").strip()
    if isinstance(message, dict):
        return (message.get("content") or "").strip()
    return ""


def main():
    parser = argparse.ArgumentParser(description="Read Question column, call local Ollama via its HTTP API, and save result to new xlsx.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--key",
        type=str,
        default=DEFAULT_API_KEY,
        help=f"API key for OpenAI-compatible clients. Not required for local Ollama. Defaults to env {DEFAULT_API_KEY_ENV}.",
    )
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL, help="Base URL for the Ollama-compatible HTTP API.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS, help="最大 token 数（硬限制）")
    parser.add_argument("--start_row", type=int, default=1, help="从第几行开始读取（1-based）")
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=10,
        help="每隔多少条记录保存一次输出结果",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = pd.read_excel(args.input)
    question_col = resolve_question_column(df)

    if "Result" not in df.columns:
        df["Result"] = ""
    if "Provider_Result_Token_Count" not in df.columns:
        df["Provider_Result_Token_Count"] = 0
    # `Result_Token_Count` removed by user request — we keep provider tokens but
    # do not persist the local token count column anymore.
    if "Result_Was_Truncated" not in df.columns:
        df["Result_Was_Truncated"] = False

    start_pos = max(1, args.start_row) - 1
    if start_pos >= len(df):
        print(f"起始行 {args.start_row} 超出数据总行数 {len(df)}，无需处理。", flush=True)
        df[[question_col, "Result", "Provider_Result_Token_Count", "Result_Was_Truncated"]].rename(
            columns={question_col: "question"}
        ).to_excel(args.output, index=False)
        print(f"处理完成，已保存到 {args.output}", flush=True)
        return

    if args.key:
        client = OpenAI(api_key=args.key, base_url=args.base_url, timeout=DEFAULT_TIMEOUT)
    else:
        client = OpenAI(api_key="dummy", base_url=args.base_url, timeout=DEFAULT_TIMEOUT)
    encoder = _get_token_encoder()
    processed_count = 0
    checkpoint_interval = max(1, args.checkpoint_interval)

    for pos, q in df.loc[start_pos:, question_col].items():
        q_text = "" if pd.isna(q) else str(q)
        human_row = pos + 1

        answer = ""
        provider_completion_tokens = 0
        completion_tokens = 0
        was_truncated = False

        print(f"\n🔄 Running row {human_row} ...", flush=True)
        print(f"😊 Question: {q_text[:80]}{'...' if len(q_text) > 80 else ''}", flush=True)

        # Retry only on transient API errors (network / timeout / server errors).
        # Do NOT retry on auth/permission or invalid-request errors.
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"{q_text}\n\n"
                                f"Please keep the final answer within {args.max_tokens} tokens."
                            ),
                        },
                    ],
                    max_completion_tokens=args.max_tokens,
                    max_tokens=args.max_tokens,
                    temperature=0,
                    stream=False,
                    timeout=DEFAULT_TIMEOUT,
                    extra_body={
                        "options": {
                            "num_ctx": 8192,
                            "num_predict": args.max_tokens,
                        }
                    },
                )

                answer = _extract_answer(resp)
                provider_completion_tokens = int(resp.usage.completion_tokens) if getattr(resp, "usage", None) else 0

                # If provider reports tokens but returned an empty body, try several
                # fallbacks (text field, raw resp) and save whatever we can up to
                # the local max. This prevents storing an empty Result when the
                # provider generated output but placed it in a non-standard field
                # or the SDK wrapped it differently.
                if not answer and provider_completion_tokens > 0:
                    raw_fallback = None
                    # try common text fields on the choice
                    try:
                        choice = resp.choices[0]
                        raw_fallback = getattr(choice, "text", None)
                        if not raw_fallback and isinstance(choice, dict):
                            raw_fallback = choice.get("text") or choice.get("message")
                    except Exception:
                        raw_fallback = None

                    if not raw_fallback:
                        # last resort: stringify the whole response
                        try:
                            raw_fallback = str(resp)
                        except Exception:
                            raw_fallback = ""

                    answer, completion_tokens, was_truncated = _truncate_to_max_tokens(raw_fallback, args.max_tokens, encoder)
                    print(
                        f"⚠️ Provider returned no `message.content`. Saved fallback (truncated) output, tokens_reported={provider_completion_tokens}, saved_tokens={completion_tokens}",
                        flush=True,
                    )
                else:
                    answer, completion_tokens, was_truncated = _truncate_to_max_tokens(answer, args.max_tokens, encoder)

                print(f"✅ Answer: {answer[:120]}{'...' if len(answer) > 120 else ''}", flush=True)
                print(f"🔢 Provider completion tokens: {provider_completion_tokens}", flush=True)
                print(f"🔢 Saved Result tokens (local): {completion_tokens}", flush=True)
                if was_truncated:
                    print(
                        f"⚠️ Provider did not respect max_tokens; locally truncated to {args.max_tokens} tokens.",
                        flush=True,
                    )
                break  # success — exit retry loop

            except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
                print(f"[{human_row}] 认证失败/无权限（不会重试）: {e}", flush=True)
                df[[question_col, "Result", "Provider_Result_Token_Count", "Result_Was_Truncated"]].rename(
                    columns={question_col: "question"}
                ).to_excel(args.output, index=False)
                raise SystemExit(
                    "API key 无效或没有权限。请检查 key / base_url / model，并建议立即更换（rotate）已泄露的 key。"
                )

            except (openai.BadRequestError, openai.NotFoundError, openai.UnprocessableEntityError) as e:
                print(f"[{human_row}] 请求参数错误/资源不存在（不会重试）: {e}", flush=True)
                df[[question_col, "Result", "Provider_Result_Token_Count", "Result_Was_Truncated"]].rename(
                    columns={question_col: "question"}
                ).to_excel(args.output, index=False)
                raise SystemExit("请求不可用（BadRequest/NotFound/UnprocessableEntity）。请修正参数后再试。")

            except openai.RateLimitError as e:
                # If provider explicitly says "wait N seconds", honor it.
                msg = str(e)
                wait_time = 2 * (attempt + 1)
                import re

                m = re.search(r"wait\s+(\d+)\s*seconds|等待\s*(\d+)\s*秒", msg, flags=re.IGNORECASE)
                if m:
                    seconds = next((g for g in m.groups() if g), None)
                    if seconds:
                        wait_time = max(wait_time, int(seconds))
                print(f"[{human_row}] 触发限流: {e}, {wait_time} 秒后重试 ({attempt + 1}/5)...", flush=True)
                time.sleep(wait_time)

            except (openai.APIStatusError, openai.APIError) as e:
                wait_time = 2 * (attempt + 1)
                print(f"[{human_row}] API 错误: {e}, {wait_time} 秒后重试 ({attempt + 1}/5)...", flush=True)
                time.sleep(wait_time)

            except Exception as e:
                wait_time = 2 * (attempt + 1)
                print(f"[{human_row}] 调用失败: {e}, {wait_time} 秒后重试 ({attempt + 1}/5)...", flush=True)
                time.sleep(wait_time)

        df.at[pos, "Result"] = answer
        # `Result_Token_Count` column removed — we still compute `completion_tokens`
        # for logging but we no longer persist it per user request.
        df.at[pos, "Result_Was_Truncated"] = was_truncated
        df.at[pos, "Provider_Result_Token_Count"] = provider_completion_tokens
        processed_count += 1

        if processed_count % checkpoint_interval == 0:
            df[[question_col, "Result", "Provider_Result_Token_Count", "Result_Was_Truncated"]].rename(
                columns={question_col: "question"}
            ).to_excel(args.output, index=False)
            print(
                f"✅ 已分批保存：处理到第 {human_row} 行（本批累计 {processed_count} 条），checkpoint_interval={checkpoint_interval}。",
                flush=True,
            )

    # Final save
    df[[question_col, "Result", "Provider_Result_Token_Count", "Result_Was_Truncated"]].rename(
        columns={question_col: "question"}
    ).to_excel(args.output, index=False)
    print(f"处理完成，已保存到 {args.output}", flush=True)


if __name__ == "__main__":
    main()
