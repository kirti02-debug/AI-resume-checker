import json
import time
from google import genai
from google.genai import errors



client = genai.Client()

# --- Get resume and JD from user ---
resume_text = input("Paste the RESUME text: ")
jd_text = input("Paste the JOB DESCRIPTION text: ")

# --- Build the comparison prompt ---
prompt = f"""
You are an expert technical recruiter and resume screener.

Compare the following RESUME against the JOB DESCRIPTION and provide:
Return ONLY a valid JSON object (no markdown, no code fences, no extra text) with this exact structure:
{{
  "match_score": <integer 0-100>,
  "top_strengths": [<string>, <string>, ...],
  "missing_skills": [<string>, <string>, ...],
 "summary": "<exactly 2 lines of plain text, separated by \\n>"
}}

RESUME:
\"\"\"
{resume_text}
\"\"\"

JOB DESCRIPTION:
\"\"\"
{jd_text}
\"\"\"

"""
def generate_with_retry(client, model, prompt, retries=4, base_delay=5, timeout=30):
    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"http_options": {"timeout": timeout * 1000}}
            )
            return response

        except errors.ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = base_delay * attempt
                print(f"Rate limited (429). Waiting {wait}s before retry {attempt}/{retries}...")
                time.sleep(wait)
            else:
                print(f"Client error (not retryable): {e}")
                raise

        except errors.ServerError as e:
            wait = base_delay * attempt
            print(f"Server error (attempt {attempt}/{retries}). Waiting {wait}s before retry...")
            time.sleep(wait)

        except TimeoutError:
            print(f"Request timed out (attempt {attempt}/{retries}). Retrying...")
            time.sleep(base_delay)

        except Exception as e:
            print(f"Unexpected error (attempt {attempt}/{retries}): {e}")
            time.sleep(base_delay)

    raise RuntimeError(f"Failed to get a response from Gemini after {retries} attempts.")

def parse_gemini_json(response_text: str) -> dict:
    # Clean up in case Gemini wraps it in ```json ... ``` anyway
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini did not return valid JSON: {e}\nRaw response:\n{response_text}")

    # Basic validation of expected keys

    required_keys = {"match_score", "top_strengths", "missing_skills","summary" }
    missing = required_keys - data.keys()
    if missing:
        raise ValueError(f"JSON is missing required keys: {missing}")

    if not isinstance(data["match_score"], int) or not (0 <= data["match_score"] <= 100):
        raise ValueError("match_score must be an integer between 0 and 100")

    if not isinstance(data["summary"], str):
        raise ValueError("summary must be a string")

    line_count = len([line for line in data["summary"].split("\n") if line.strip()])
    if line_count > 2:
        raise ValueError(f"summary must be exactly 2 lines, got {line_count} lines")


    return data


# call the API

response = generate_with_retry(client, "gemini-3.8-flash", prompt)

#pare the output

result_json = parse_gemini_json(response.text)
print("Match Score:", result_json["match_score"])
print("Summary:", result_json["summary"])
print("top_strengths:", result_json["top_strengths"])



