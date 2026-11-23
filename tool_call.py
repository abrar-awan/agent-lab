"""Session 4: one tool call, run by hand.

The model never runs code. It replies with a request to call a tool;
we run the tool ourselves and send the result back in a second request.

    uv run tool_call.py                  # invoice_01, honest
    uv run tool_call.py --tamper         # same invoice, total raised by 9.00
    uv run tool_call.py invoice_02.pdf   # an invoice with no tax line
"""

import json
import re
import sys

import pdfplumber
from google.genai import types

from llm import generate


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


# 3. Load an invoice, optionally with a wrong total.
names = [a for a in sys.argv[1:] if not a.startswith("--")]
target = names[0] if names else "invoice_01.pdf"
with pdfplumber.open(f"docs/{target}") as pdf:
    text = pdf.pages[0].extract_text()

if "--tamper" in sys.argv:
    *body, last = text.splitlines()  # the last line is always the total
    amount = re.findall(r"[\d,]+\.\d{2}", last)[-1]
    fake = f"{float(amount.replace(',', '')) + 9:,.2f}"
    text = "\n".join(body + [last.replace(amount, fake)])
    print(f"(tampered: total {amount} -> {fake})\n")

contents = [
    types.Content(role="user", parts=[types.Part(text=(
        "Is this invoice's arithmetic consistent? "
        "Check it with the tool, then answer in one sentence.\n\n" + text
    ))])
]

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
