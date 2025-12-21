#!/bin/sh
set -eu

echo "[init] inserting initial data into app database..."

psql -v ON_ERROR_STOP=1 --username "$APP_DB_USER" --dbname "$APP_DB" <<-'EOSQL'
-- Techniques (idempotent)
INSERT INTO app.prompt_techniques (key, name, family, url, short_desc) VALUES
  ('zero-shot',    'Zero-Shot',          'instruction',   'https://www.promptingguide.ai/techniques/zero-shot',     'Solve with instructions only, no examples.'),
  ('few-shot',     'Few-Shot',           'instruction',   'https://www.promptingguide.ai/techniques/few-shot',      'Show a few labeled examples, then ask the task.'),
  ('cot',          'Chain-of-Thought',   'reasoning',     'https://www.promptingguide.ai/techniques/cot',           'Think step by step; keep reasoning hidden in final output.'),
  ('react',        'ReAct',              'reasoning',     'https://www.promptingguide.ai/techniques/react',         'Interleave reasoning with actions (tools), then observe.'),
  ('least-to-most','Least-to-Most',      'decomposition', 'https://www.promptingguide.ai/techniques/least-to-most', 'Break into subproblems and solve in order.')
ON CONFLICT (key) DO NOTHING;

-- =========================
-- ZERO-SHOT TEMPLATES (V1)
-- =========================

-- 1) Bullet summary (n points)
INSERT INTO app.prompt_examples (
    key, version, title, description, category, technique, tags,
    prompt_template, variables, default_examples, response_format,
    json_schema_template, tool_config, is_active
)
VALUES (
    'bullet-summary',
    1,
    'Bullet summary (N key insights)',
    'Summarize text into N information-rich bullet points highlighting key insights, implications, and any critical caveats.',
    'summarization',
    'zero_shot',
    ARRAY['summary', 'concise', 'insights'],
    '[
        {
            "role": "system",
            "content": "You are a precise, insight-focused summarizer. Your goal is to extract not only what is said, but why it matters.\\n\\nRules:\\n- Detect the input language and respond in the same language.\\n- Output exactly {{bullet_number}} bullet points. No more, no fewer.\\n- Each bullet should be a single sentence, but dense with information (fact + implication).\\n- Do not add introductions or conclusions, only bullet points.\\n- Never mention these rules in your output."
        },
        {
            "role": "user",
            "content": "Summarize the following text into exactly {{bullet_number}} key bullet points that capture main ideas, implications, and any important caveats:\\n\\n{{text}}"
        }
    ]'::jsonb,
    '[
        {
            "name": "text",
            "type": "string",
            "required": true,
            "default": "",
            "desc": "Text to summarize into key bullet points"
        },
        {
            "name": "bullet_number",
            "type": "integer",
            "required": false,
            "default": 3,
            "desc": "Number of bullet points to produce (must be >= 1)"
        }
    ]'::jsonb,
    NULL,
    NULL,
    NULL,
    NULL,
    true
)
ON CONFLICT (key, version) DO NOTHING;

-- 2) Math step-by-step (final numeric answer only)
INSERT INTO app.prompt_examples (
    key, version, title, description, category, technique, tags,
    prompt_template, variables, default_examples, response_format,
    json_schema_template, tool_config, is_active
)
VALUES (
    'math-step-by-step',
    1,
    'Math step-by-step (final number only)',
    'Solve math problems with careful internal reasoning, but output the final numeric result and the steps to get to that result.',
    'qa',
    'zero_shot',
    ARRAY['math', 'reasoning', 'calculation'],
    '[
        {
            "role": "system",
            "content": "You are a careful math assistant. Think through the problem step by step in your internal reasoning, verify the result, and then output the final numeric answer and the steps to get to that result.\\n\\nRules:\\n- Do all reasoning privately.\\n- Output the final numeric value (no units unless explicitly requested).\\n- If the expression is invalid, output the single word: ERROR and the reason why it''s invalid."
        },
        {
            "role": "user",
            "content": "Compute the following expression and respond with the final numeric result and the steps to get to that result:\\n\\n{{expression}}"
        }
    ]'::jsonb,
    '[
        {
            "name": "expression",
            "type": "string",
            "required": true,
            "default": "",
            "desc": "Arithmetic expression to compute"
        }
    ]'::jsonb,
    NULL,
    NULL,
    NULL,
    NULL,
    true
)ON CONFLICT (key, version) DO NOTHING;

-- 3) Bugfix (minimal snippet + explanation, with lang)
INSERT INTO app.prompt_examples (
    key, version, title, description, category, technique, tags,
    prompt_template, variables, default_examples, response_format,
    json_schema_template, tool_config, is_active
)
VALUES (
    'bugfix-patch',
    1,
    'Bugfix (minimal snippet + explanation)',
    'Fix bugs in code by returning the minimally changed code snippet and a short explanation, instead of a full diff.',
    'coding',
    'zero_shot',
    ARRAY['coding', 'bugfix', 'snippet'],
    '[
        {
            "role": "system",
            "content": "You are an experienced software engineer. Analyze the code and the bug description in your internal reasoning, identify the minimal safe change, and output a JSON object with two fields: `fixed_code` and `short_explanation`.\\n\\nRules for the FINAL output:\\n- `fixed_code` MUST contain only the modified portion of the file (for example, the changed function or relevant block), not the entire file.\\n- The code will be provided inside a fenced code block with a language tag, like ```{{lang}} ...```.\\n- Keep the style and formatting consistent with the original code.\\n- `short_explanation` MUST briefly describe what was wrong and how you fixed it in 1–3 sentences.\\n- Do not include any extra keys beyond those defined in the schema.\\n- Do not include explanations outside of the JSON. The final output must be valid JSON only."
        },
        {
            "role": "user",
            "content": "You are given a file and a bug description. Internally, analyze what is wrong and how to fix it, then output ONLY the JSON object described in the instructions.\\n\\nLanguage: {{lang}}\\n\\nBug description:\\n{{bug_desc}}\\n\\nFile contents (language-specified fenced block):\\n```{{lang}}\\n{{file_text}}\\n```"
        }
    ]'::jsonb,
    '[
        {
            "name": "bug_desc",
            "type": "string",
            "required": true,
            "default": "",
            "desc": "Bug description and symptoms"
        },
        {
            "name": "file_text",
            "type": "string",
            "required": true,
            "default": "",
            "desc": "Full contents of the file to patch"
        },
        {
            "name": "lang",
            "type": "string",
            "required": false,
            "default": "",
            "desc": "Programming language of the code (e.g., python, javascript, sql). If empty, the model may infer."
        }
    ]'::jsonb,
    NULL,
    'json_schema',
    '{
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "fixed_code": {
                "type": "string",
                "description": "Only the modified portion of the file (e.g., function or relevant block) with the fix applied."
            },
            "short_explanation": {
                "type": "string",
                "description": "1–3 sentences explaining what was wrong and how it was fixed."
            }
        },
        "required": ["fixed_code", "short_explanation"]
    }'::jsonb,
    NULL,
    true
)
ON CONFLICT (key, version) DO NOTHING;

-- 4) Decompose then solve (natural language)
INSERT INTO app.prompt_examples (
    key, version, title, description, category, technique, tags,
    prompt_template, variables, default_examples, response_format,
    json_schema_template, tool_config, is_active
)
VALUES (
    'decompose-solve',
    1,
    'Decompose then solve (plan + answer)',
    'Break complex tasks into ordered subproblems, solve them briefly, and provide a concise final answer section.',
    'qa',
    'zero_shot',
    ARRAY['decomposition', 'reasoning', 'planning'],
    '[
        {
            "role": "system",
            "content": "You are a problem-solving assistant that works in two stages: PLAN and ANSWER.\\n\\nOutput format:\\n1) A section starting with `PLAN:` followed by a short numbered list of 3–7 substeps.\\n2) A section starting with `ANSWER:` containing a concise final answer (2–5 sentences).\\n\\nRules:\\n- Keep each plan step short but specific.\\n- In the ANSWER, do not repeat the entire plan; only the conclusions.\\n- If information is missing, explicitly state assumptions in the PLAN section."
        },
        {
            "role": "user",
            "content": "Task:\\n{{task}}"
        }
    ]'::jsonb,
    '[
        {
            "name": "task",
            "type": "string",
            "required": true,
            "default": "",
            "desc": "Complex task or question to decompose and solve"
        }
    ]'::jsonb,
    NULL,
    NULL,
    NULL,
    NULL,
    true
)ON CONFLICT (key, version) DO NOTHING;


-- =========================
-- FEW-SHOT TEMPLATES (V1)
-- =========================

-- 5) Sentiment classifier (json_schema with confidence + explanation)
INSERT INTO app.prompt_examples (
    key, version, title, description, category, technique, tags,
    prompt_template, variables, default_examples, response_format,
    json_schema_template, tool_config, is_active
)
VALUES (
    'sentiment-classifier',
    1,
    'Sentiment (pos/neg/neutral + confidence)',
    'Classify sentiment with few-shot examples and return a structured JSON object containing label, confidence, and explanation.',
    'classification',
    'few_shot',
    ARRAY['sentiment', 'classification', 'json'],
    '[
        {
            "role": "system",
            "content": "You are a robust sentiment classifier. Your job is to map short texts to one of three labels: positive, negative, or neutral, and explain why.\\n\\nRules:\\n- Consider emotional tone, intensity, and context.\\n- If sentiment is mixed, choose the dominant one but reduce confidence.\\n- Your final output MUST follow the provided JSON schema (label, confidence, explanation, evidence_snippets)."
        },
        {
            "role": "user",
            "content": "Classify the sentiment of the following text:\\n\\n{{text}}"
        }
    ]'::jsonb,
    '[
        {
            "name": "text",
            "type": "string",
            "required": true,
            "default": "",
            "desc": "Sentence or short paragraph to classify"
        }
    ]'::jsonb,
    '[
        {
            "input": "I love this product!",
            "output": {
                "label": "positive",
                "confidence": 0.96,
                "explanation": "The word love expresses strong positive emotion and satisfaction.",
                "evidence_snippets": ["love this product"]
            }
        },
        {
            "input": "This is the worst.",
            "output": {
                "label": "negative",
                "confidence": 0.94,
                "explanation": "The phrase the worst clearly indicates a very negative evaluation.",
                "evidence_snippets": ["the worst"]
            }
        },
        {
            "input": "It is okay, I guess.",
            "output": {
                "label": "neutral",
                "confidence": 0.68,
                "explanation": "The wording is mildly positive but softened by uncertainty, resulting in a near-neutral tone.",
                "evidence_snippets": ["okay", "I guess"]
            }
        }
    ]'::jsonb,
    'json_schema',
    '{
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "label": {
                "type": "string",
                "description": "Sentiment label",
                "enum": ["positive", "negative", "neutral"]
            },
            "confidence": {
                "type": "number",
                "description": "Model confidence in the chosen label, between 0 and 1",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "explanation": {
                "type": "string",
                "description": "Short natural-language explanation of why this label was chosen"
            },
            "evidence_snippets": {
                "type": "array",
                "description": "Phrases from the text that strongly support the decision",
                "items": {
                    "type": "string"
                }
            }
        },
        "required": ["label", "confidence", "explanation", "evidence_snippets"]
    }'::jsonb,
    NULL,
    true
)ON CONFLICT (key, version) DO NOTHING;

-- 6) Code review feedback (sections, with lang)
INSERT INTO app.prompt_examples (
    key, version, title, description, category, technique, tags,
    prompt_template, variables, default_examples, response_format,
    json_schema_template, tool_config, is_active
)
VALUES (
    'code-review',
    1,
    'Code review feedback (sections)',
    'Provide structured code review feedback with sections: Summary, Strengths, Issues, Suggestions, and Risk level.',
    'coding',
    'few_shot',
    ARRAY['code-review', 'feedback', 'best-practices'],
    '[
        {
            "role": "system",
            "content": "You are an experienced code reviewer. Provide constructive, actionable feedback focusing on readability, correctness, performance, and maintainability.\\n\\nOutput format (strict):\\nSUMMARY:\\n- One or two sentences summarizing the state of the code.\\n\\nSTRENGTHS:\\n- Bullet list of good aspects.\\n\\nISSUES:\\n- Bullet list of problems, each including WHAT and WHY it matters.\\n\\nSUGGESTIONS:\\n- Bullet list of concrete improvements (including examples or snippets when useful).\\n\\nRISK_LEVEL:\\n- One of: low, medium, high, based on potential impact of the issues.\\n\\nDo not include any other sections or headings."
        },
        {
            "role": "user",
            "content": "Language: {{lang}}\\n\\nPlease review the following code and provide structured feedback. The code is wrapped in a fenced block with the language tag:\\n\\n```{{lang}}\\n{{code}}\\n```"
        }
    ]'::jsonb,
    '[
        {
            "name": "code",
            "type": "string",
            "required": true,
            "default": "",
            "desc": "Code snippet or function to review"
        },
        {
            "name": "lang",
            "type": "string",
            "required": false,
            "default": "",
            "desc": "Programming language of the code (e.g., python, javascript, sql). If empty, the model may infer."
        }
    ]'::jsonb,
    '[
        {
            "input": "def add(a, b):\\n    return a + b",
            "lang": "python",
            "output": "SUMMARY:\\n- Simple, clear implementation of an addition helper.\\n\\nSTRENGTHS:\\n- Very easy to read and understand.\\n- Single responsibility function.\\n\\nISSUES:\\n- No type hints, which can reduce clarity in larger codebases.\\n- No docstring describing behavior or edge cases (e.g., non-numeric inputs).\\n\\nSUGGESTIONS:\\n- Add type hints, e.g., `def add(a: float, b: float) -> float:`.\\n- Add a short docstring explaining purpose and expected types.\\n\\nRISK_LEVEL:\\n- low"
        }
    ]'::jsonb,
    NULL,
    NULL,
    NULL,
    true
)
ON CONFLICT (key, version) DO NOTHING;

-- 7) Email tone converter (with explicit constraints)
INSERT INTO app.prompt_examples (
    key, version, title, description, category, technique, tags,
    prompt_template, variables, default_examples, response_format,
    json_schema_template, tool_config, is_active
)
VALUES (
    'email-tone',
    1,
    'Email tone converter (controlled tone)',
    'Convert email tone while preserving intent and key details, with examples.',
    'writing',
    'few_shot',
    ARRAY['email', 'tone', 'writing', 'rewrite'],
    '[
        {
            "role": "system",
            "content": "You are an assistant that rewrites emails to match a target tone while preserving meaning.\\n\\nRules:\\n- Preserve all concrete details (dates, numbers, names, links, attachments).\\n- Keep roughly the same length (max ±25% change).\\n- Adjust phrasing to match the requested tone: professional, casual, or formal.\\n- Do not invent information.\\n- Output only the rewritten email, no commentary."
        },
        {
            "role": "user",
            "content": "Target tone: {{target_tone}}\\n\\nRewrite this email to the target tone while preserving all important details:\\n\\n{{email_text}}"
        }
    ]'::jsonb,
    '[
        {
            "name": "email_text",
            "type": "string",
            "required": true,
            "default": "",
            "desc": "Original email text to rewrite"
        },
        {
            "name": "target_tone",
            "type": "string",
            "required": false,
            "default": "professional",
            "desc": "Target tone (professional, casual, formal)"
        }
    ]'::jsonb,
    '[
        {
            "input": "Hey, can you send me that file? Thanks!",
            "target_tone": "professional",
            "output": "Hello, could you please send me that file when you have a moment? Thank you in advance."
        },
        {
            "input": "I need this done ASAP.",
            "target_tone": "professional",
            "output": "I would appreciate it if this could be completed as soon as possible."
        }
    ]'::jsonb,
    NULL,
    NULL,
    NULL,
    true
)ON CONFLICT (key, version) DO NOTHING;


-- =========================
-- PROMPT_CHAIN / STRUCTURED
-- =========================

-- Unified Q&A: optional context + confidence + follow-ups
INSERT INTO app.prompt_examples (
    key, version, title, description, category, technique, tags,
    prompt_template, variables, default_examples, response_format,
    json_schema_template, tool_config, is_active
)
VALUES (
    'qa-unified',
    1,
    'Q&A (optional context + confidence + follow-ups)',
    'Answer questions using optional context. If context is provided, rely primarily on it; otherwise use internal knowledge. Return answer, confidence, answerability, sources, and follow-up questions.',
    'qa',
    'zero_shot',
    ARRAY['qa', 'context', 'json', 'confidence', 'followup'],
    '[
        {
            "role": "system",
            "content": "You are a grounded question-answering assistant.\\n\\nContext handling:\\n- If a non-empty context is provided, rely primarily on that context.\\n- If the answer cannot be reliably derived from the context, say so and lower confidence.\\n- If context is empty, answer from your general knowledge.\\n\\nOutput rules:\\n- Provide a clear, direct answer in plain language.\\n- Include a realistic confidence score between 0 and 1.\\n- Indicate whether the question was answerable from the provided context (if any).\\n- When context is used, list the key phrases or identifiers that support your answer in `sources_used`.\\n- Provide 2–4 short follow-up questions the user might ask next.\\n- Do not include chain-of-thought or internal reasoning. Return only the final structured JSON that matches the schema."
        },
        {
            "role": "user",
            "content": "Context (may be empty):\\n{{context}}\\n\\nQuestion: {{question}}"
        }
    ]'::jsonb,
    '[
        {
            "name": "question",
            "type": "string",
            "required": true,
            "default": "",
            "desc": "User question that requires an answer"
        },
        {
            "name": "context",
            "type": "string",
            "required": false,
            "default": "",
            "desc": "Optional context to ground the answer. If empty, answer from internal knowledge."
        }
    ]'::jsonb,
    NULL,
    'json_schema',
    '{
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "answer": {
                "type": "string",
                "description": "Direct, concise answer to the question."
            },
            "confidence": {
                "type": "number",
                "description": "Confidence from 0 to 1 about correctness.",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "answerable": {
                "type": "boolean",
                "description": "True if the available context (if any) clearly contains enough information to answer the question."
            },
            "sources_used": {
                "type": "array",
                "description": "Key phrases or identifiers from the context that support the answer. Empty if no context was used.",
                "items": { "type": "string" }
            },
            "missing_info": {
                "type": "string",
                "description": "If answerable is false or confidence is low, describe what information is missing."
            },
            "follow_up_questions": {
                "type": "array",
                "description": "List of 2–4 suggested follow-up questions.",
                "items": { "type": "string" }
            }
        },
        "required": ["answer", "confidence", "answerable", "sources_used", "missing_info", "follow_up_questions"]
    }'::jsonb,
    NULL,
    true
)
ON CONFLICT (key, version) DO NOTHING;


-- 9) Demographic extract (basic profile + confidence)
INSERT INTO app.prompt_examples (
    key, version, title, description, category, technique, tags,
    prompt_template, variables, default_examples, response_format,
    json_schema_template, tool_config, is_active
)
VALUES (
    'text-extract',
    1,
    'Extract basic demographics (with confidence)',
    'Extract structured demographic information (full_name, age, gender, ethnicity, education_level, employment_status, marital_status) from text, plus overall confidence and explanation.',
    'extraction',
    'zero_shot',
    ARRAY['extraction', 'structured', 'demographics', 'json'],
    '[
        {
            "role": "system",
            "content": "You extract basic demographic information from messy text into a strict JSON object.\\n\\nFields to extract:\\n- full_name\\n- age\\n- gender\\n- ethnicity\\n- education_level\\n- employment_status\\n- marital_status\\n\\nRules:\\n- Do not invent values; if a field is missing or uncertain, use an empty string (for string fields) or null (for age) and lower overall confidence.\\n- Never infer sensitive attributes like gender or ethnicity solely from a name; only use explicit information in the text.\\n- If multiple candidates exist, choose the most likely one based on the text.\\n- Final output MUST follow the provided JSON schema (fields + confidence + explanation)."
        },
        {
            "role": "user",
            "content": "From the following text, extract full_name, age, gender, ethnicity, education_level, employment_status, and marital_status into JSON. If something is not present, leave it as an empty string (or null for age) but explain why in the explanation field.\\n\\nText:\\n{{text}}"
        }
    ]'::jsonb,
    '[
        {
            "name": "text",
            "type": "string",
            "required": true,
            "default": "",
            "desc": "Text to extract demographic information from"
        }
    ]'::jsonb,
    NULL,
    'json_schema',
    '{
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "full_name": {
                "type": "string",
                "description": "Full name (empty string if not found)"
            },
            "age": {
                "type": ["integer", "null"],
                "description": "Age in years (null if not found)",
                "minimum": 0,
                "maximum": 120
            },
            "gender": {
                "type": "string",
                "description": "Self-described gender (empty string if not found)"
            },
            "ethnicity": {
                "type": "string",
                "description": "Self-described ethnicity (empty string if not found)"
            },
            "education_level": {
                "type": "string",
                "description": "Highest education level mentioned (empty string if not found)"
            },
            "employment_status": {
                "type": "string",
                "description": "Employment status or job situation (empty string if not found)"
            },
            "marital_status": {
                "type": "string",
                "description": "Marital status (empty string if not found)"
            },
            "confidence": {
                "type": "number",
                "description": "Global confidence in the extracted information, between 0 and 1.",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "explanation": {
                "type": "string",
                "description": "Short explanation describing how the fields were inferred and which were missing or ambiguous."
            }
        },
        "required": [
            "full_name",
            "age",
            "gender",
            "ethnicity",
            "education_level",
            "employment_status",
            "marital_status",
            "confidence",
            "explanation"
        ]
    }'::jsonb,
    NULL,
    true
)
ON CONFLICT (key, version) DO NOTHING;

-- 10) Explain concept simply (with structure in text)
INSERT INTO app.prompt_examples (
    key, version, title, description, category, technique, tags,
    prompt_template, variables, default_examples, response_format,
    json_schema_template, tool_config, is_active
)
VALUES (
    'explain-concept',
    1,
    'Explain concept simply (sections)',
    'Explain complex concepts in simple terms, including analogy, step-by-step breakdown, and a quick checklist.',
    'education',
    'zero_shot',
    ARRAY['explanation', 'education', 'beginner'],
    '[
        {
            "role": "system",
            "content": "You are a teacher explaining complex ideas to a smart beginner.\\n\\nOutput format:\\n1) ONE-SENTENCE SUMMARY\\n2) EXPLANATION (3–6 short paragraphs)\\n3) ANALOGY (compare to something familiar)\\n4) STEP-BY-STEP (bullet list of how it works)\\n5) QUICK CHECKLIST (3–5 bullets to test understanding)\\n\\nUse clear, simple language and avoid jargon unless you define it."
        },
        {
            "role": "user",
            "content": "Explain the following concept in simple terms:\\n\\n{{concept}}"
        }
    ]'::jsonb,
    '[
        {
            "name": "concept",
            "type": "string",
            "required": true,
            "default": "",
            "desc": "Concept, term, or idea to explain"
        }
    ]'::jsonb,
    NULL,
    NULL,
    NULL,
    NULL,
    true
)ON CONFLICT (key, version) DO NOTHING;

-- 11) Movie details via OMDB tool
INSERT INTO app.prompt_examples (
    key, version, title, description, category, technique, tags,
    prompt_template, variables, default_examples, response_format,
    json_schema_template, tool_config, is_active
)
VALUES (
    'movie-details-tool',
    1,
    'Movie details summary',
    'Look up detailed information about a movie using the get_custom_movie_details tool (OMDB) and present a concise, user-friendly summary.',
    'media',
    'zero_shot',
    ARRAY['movies', 'omdb', 'tools', 'lookup'],
    '[
        {
            "role": "system",
            "content": "You are a movie information assistant with access to a tool called get_custom_movie_details that queries the OMDB API.\\n\\nRules:\\n- When the user asks about a specific movie, you should use the tool to fetch accurate data instead of guessing.\\n- After receiving the tool result, synthesize a clear answer that covers title, year, genre, rating, short plot, and main actors.\\n- If the movie cannot be found, say that you could not find it and suggest the user to check the title or provide more details.\\n- Do not fabricate data that is not present in the tool output."
        },
        {
            "role": "user",
            "content": "Look up detailed information about the following movie and summarize it for me:\\n\\n{{movie_title}}"
        }
    ]'::jsonb,
    '[
        {
            "name": "movie_title",
            "type": "string",
            "required": true,
            "default": "",
            "desc": "Title of the movie to look up (optionally including year, e.g. ''Inception 2010'')"
        }
    ]'::jsonb,
    NULL,
    NULL,
    NULL,
    '{"tool": "get_custom_movie_details"}'::jsonb,
    true
) ON CONFLICT (key, version) DO NOTHING;

-- 12) arXiv paper search with search_arxiv tool
INSERT INTO app.prompt_examples (
    key, version, title, description, category, technique, tags,
    prompt_template, variables, default_examples, response_format,
    json_schema_template, tool_config, is_active
)
VALUES (
    'arxiv-search-tool',
    1,
    'Search arXiv for papers',
    'Search arXiv for academic papers on a topic using the search_arxiv tool and summarize the most relevant results.',
    'research',
    'zero_shot',
    ARRAY['arxiv', 'papers', 'research', 'tools'],
    '[
        {
            "role": "system",
            "content": "You are a research assistant with access to a tool called search_arxiv that searches the arXiv repository.\\n\\nRules:\\n- Always use the search_arxiv tool when the user asks for academic papers, arXiv papers, or research on a topic.\\n- After receiving the tool result, select the most relevant papers (up to the requested limit) and summarize them clearly.\\n- For each paper, ideally include: title, main idea in 1–3 sentences, authors (shortened if the list is very long), and the year.\\n- If the user asks, provide the arXiv or PDF link from the tool output.\\n- If no relevant papers are found, say so explicitly instead of inventing papers."
        },
        {
            "role": "user",
            "content": "Search arXiv for papers related to the following topic or query, and summarize up to {{max_results}} of the most relevant ones:\\n\\n{{query}}"
        }
    ]'::jsonb,
    '[
        {
            "name": "query",
            "type": "string",
            "required": true,
            "default": "",
            "desc": "Search query or topic for arXiv (keywords, question, or short description)"
        },
        {
            "name": "max_results",
            "type": "string",
            "required": false,
            "default": "5",
            "desc": "Maximum number of papers to summarize (as a small integer in string form, e.g. ''3'', ''5'')"
        }
    ]'::jsonb,
    NULL,
    NULL,
    NULL,
    '{"tool": "search_arxiv"}'::jsonb,
    true
) ON CONFLICT (key, version) DO NOTHING;

-- 13) Company fundamentals with get_fmp_company_data tool
INSERT INTO app.prompt_examples (
    key, version, title, description, category, technique, tags,
    prompt_template, variables, default_examples, response_format,
    json_schema_template, tool_config, is_active
)
VALUES (
    'company-fundamentals-tool',
    1,
    'Company fundamentals and ratios',
    'Fetch company profile and optional financial statements using the get_fmp_company_data tool, then summarize key fundamentals and ratios.',
    'finance',
    'zero_shot',
    ARRAY['finance', 'stocks', 'fundamentals', 'tools', 'fmp'],

    -- prompt_template (JSONB)
    $$[
        {
            "role": "system",
            "content": "You are a financial analysis assistant with access to a tool called get_fmp_company_data that queries the FinancialModelingPrep API. The tool can return company profile information and, when requested, income statement and balance sheet data with derived financial metrics.\n\nRules:\n- Use the tool to fetch data rather than relying on memory.\n- If financial statements are requested (statements list is not empty), highlight key metrics such as revenue, net income, margins, leverage, and any notable trends present in the tool output.\n- Explain results in clear language suitable for a non-expert, but keep the answer concise.\n- If the ticker is invalid or the tool returns no data, state that you could not find data for the given ticker and do not fabricate numbers.\n- Do not make investment recommendations; only describe the data and metrics returned by the tool."
        },
        {
            "role": "user",
            "content": "{% set statements_summary = statements | join(', ') if statements else 'profile only' %}\nFetch company information for the following ticker using the financial data tool. Use the statements argument exactly as provided to decide whether to include financial statements when calling get_fmp_company_data, and summarize the key fundamentals and metrics:\n\nTicker: {{ticker}}\nStatements argument to use: {{ statements if statements else [] }}\nStatements requested: {{ statements_summary }}"
        }
    ]$$::jsonb,

    -- variables (JSONB)
    $$[
        {
            "name": "ticker",
            "type": "string",
            "required": true,
            "default": "",
            "desc": "Public company ticker symbol (e.g. 'AAPL', 'MSFT')"
        },
        {
            "name": "statements",
            "type": "string",
            "required": false,
            "default": "income,balance",
            "desc": "Comma-separated list of statements to include (options: 'income', 'balance'). Leave blank for profile only."
        }
    ]$$::jsonb,

    -- default_examples, response_format, json_schema_template
    NULL,
    NULL,
    NULL,

    -- tool_config (JSONB)
    $${
        "tool": "get_fmp_company_data"
    }$$::jsonb,

    true
)
ON CONFLICT (key, version) DO NOTHING;


EOSQL

echo "[init] app data inserted."
