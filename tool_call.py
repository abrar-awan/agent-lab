"""Session 4: one tool call, run by hand.

The model never runs code. It replies with a request to call a tool;
we run the tool ourselves and send the result back in a second request.

    uv run tool_call.py                  # invoice_01, honest
    uv run tool_call.py --tamper         # same invoice, total raised by 9.00
    uv run tool_call.py invoice_02.pdf   # an invoice with no tax line
    uv run tool_call.py --provider groq  # same round trip, OpenAI tool format
"""

import argparse
import json
import os
import re
import sys

import pdfplumber
from google.genai import types

from llm import generate
from providers import PROVIDERS


# 1. The tool: plain Python. The model never sees this code.
def check_totals(subtotal: float, tax: float, total: float) -> dict:
    expected = round(subtotal + tax, 2)
    return {
        "consistent": abs(expected - total) < 0.01,
        "expected_total": expected,
        "difference": round(total - expected, 2),
    }


# 2. What the model does see: a name, a description and a JSON schema.
CHECK_TOTALS = types.FunctionDeclaration(
    name="check_totals",
    description=(
        "Check that an invoice's subtotal plus tax equals its total. "
        "Use this instead of doing the arithmetic yourself."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "subtotal": {"type": "number", "description": "Subtotal before tax"},
            "tax": {"type": "number", "description": "Tax amount; 0 if the invoice shows no tax"},
            "total": {"type": "number", "description": "Total amount due, as printed"},
        },
        "required": ["subtotal", "tax", "total"],
    },
)
TOOLS = {"check_totals": check_totals}  # name the model uses -> function we run

config = types.GenerateContentConfig(
    tools=[types.Tool(function_declarations=[CHECK_TOTALS])],
    # We run tools ourselves. (The SDK can do it for you; that hides the loop.)
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


def show_raw(content: types.Content) -> None:
    """Print a model turn exactly as returned, with long signatures cut short."""
    raw = content.model_dump(mode="json", exclude_none=True)
    for part in raw.get("parts", []):
        if "thought_signature" in part:
            part["thought_signature"] = part["thought_signature"][:20] + "..."
    print(json.dumps(raw, indent=2))


def tokens(response) -> str:
    u = response.usage_metadata
    return f"{u.prompt_token_count} in / {u.candidates_token_count} out / {u.thoughts_token_count or 0} thinking"


# OpenAI-format version of the same tool, for every non-Gemini provider.
CHECK_TOTALS_OPENAI = {
    "type": "function",
    "function": {
        "name": CHECK_TOTALS.name,
        "description": CHECK_TOTALS.description,
        "parameters": CHECK_TOTALS.parameters_json_schema,
    },
}


def run_openai(provider: str, prompt: str) -> None:
    """The same two-request round trip, through the `openai` package."""
    from openai import OpenAI

    cfg = PROVIDERS[provider]
    model = os.getenv(cfg["model_env"])
    api_key = "ollama" if cfg["key_env"] is None else os.getenv(cfg["key_env"])
    if not model or not api_key:
        missing = cfg["model_env"] if not model else cfg["key_env"]
        sys.exit(f"[{provider}] {missing} not set in .env")
    # max_retries=0: a failure should show up as a failure.
    client = OpenAI(base_url=cfg["base_url"], api_key=api_key, timeout=120, max_retries=0)

    messages = [{"role": "user", "content": prompt}]
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, tools=[CHECK_TOTALS_OPENAI])
    except Exception as e:
        sys.exit(f"[{provider}] request 1 FAILED: {type(e).__name__}: {str(e)[:300]} body={getattr(e, "body", None)}")
    message = response.choices[0].message
    print(f"--- turn 1: what the model returned ({provider}, {model}) ---")
    print(json.dumps(message.model_dump(mode="json", exclude_none=True), indent=2))

    if not message.tool_calls:
        sys.exit("\nThe model answered without calling the tool.")
    call = message.tool_calls[0]
    args = json.loads(call.function.arguments)
    result = TOOLS[call.function.name](**args)
    print(f"\n--- we run {call.function.name}({json.dumps(args)}) ---")
    print(result)

    messages.append(message.model_dump(exclude_none=True))  # the model's own turn, call included
    messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, tools=[CHECK_TOTALS_OPENAI])
    except Exception as e:
        sys.exit(f"[{provider}] request 2 FAILED: {type(e).__name__}: {str(e)[:300]} body={getattr(e, "body", None)}")
    print("\n--- turn 2: the model answers ---")
    print(response.choices[0].message.content)


# 3. Load an invoice, optionally with a wrong total.
parser = argparse.ArgumentParser()
parser.add_argument("invoice", nargs="?", default="invoice_01.pdf")
parser.add_argument("--tamper", action="store_true")
parser.add_argument("--provider", default="gemini", choices=list(PROVIDERS))
cli = parser.parse_args()
target = cli.invoice

if cli.provider == "mistral":  # parked: account returns 429
    print("[mistral] SKIPPED (parked)")
    sys.exit(0)
with pdfplumber.open(f"docs/{target}") as pdf:
    text = pdf.pages[0].extract_text()

if cli.tamper:
    *body, last = text.splitlines()  # the last line is always the total
    amount = re.findall(r"[\d,]+\.\d{2}", last)[-1]
    fake = f"{float(amount.replace(',', '')) + 9:,.2f}"
    text = "\n".join(body + [last.replace(amount, fake)])
    print(f"(tampered: total {amount} -> {fake})\n")

prompt = (
    "Is this invoice's arithmetic consistent? "
    "Check it with the tool, then answer in one sentence.\n\n" + text
)

if cli.provider != "gemini":
    run_openai(cli.provider, prompt)
    sys.exit(0)

contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]

# 4. Request 1: the model decides to call the tool.
response = generate(contents=contents, config=config)
print("--- turn 1: what the model returned ---")
show_raw(response.candidates[0].content)
print(f"[tokens: {tokens(response)}]")

if not response.function_calls:
    sys.exit("\nThe model answered without calling the tool.")
call = response.function_calls[0]

# 5. We run the tool. This is the "act" step, and it happens in our code.
result = TOOLS[call.name](**call.args)
print(f"\n--- we run {call.name}({json.dumps(call.args)}) ---")
print(result)

# 6. Request 2: the whole conversation again, plus the tool's result.
contents.append(response.candidates[0].content)  # the model's own turn, call included
contents.append(types.Content(role="user", parts=[
    types.Part(function_response=types.FunctionResponse(
        id=call.id, name=call.name, response=result,
    ))
]))

response = generate(contents=contents, config=config)
print("\n--- turn 2: the model answers ---")
print(response.text)
print(f"[tokens: {tokens(response)}]")
