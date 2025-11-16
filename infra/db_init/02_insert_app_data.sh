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

-- ZERO-SHOT examples
INSERT INTO app.prompt_examples (key, version, title, description, category, technique, tags, prompt_template, variables, response_format, json_schema_template, is_active)
VALUES
('bullet-summary-3', 1, 'Bullet summary (3 points)', 'Summarize text into exactly 3 bullet points', 'summarization', 'zero_shot', ARRAY['summary', 'concise'],
 '[{"role": "system", "content": "You are a concise assistant. Output only the result."}, {"role": "user", "content": "Summarize the following text in exactly 3 bullet points:\n{{text}}"}]',
 '[{"name":"text","type":"string","required":true,"default":"","desc":"Text to summarize"}]',
 NULL,
 NULL,
 true)
ON CONFLICT (key, version) DO NOTHING;

INSERT INTO app.prompt_examples (key, version, title, description, category, technique, tags, prompt_template, variables, response_format, json_schema_template, is_active)
VALUES
('math-step-by-step', 1, 'Math step-by-step (final answer only)', 'Solve math problems step by step, output only final numeric result', 'qa', 'zero_shot', ARRAY['math', 'reasoning'],
 '[{"role": "system", "content": "Reason step by step privately. In the final answer, output only the numeric result."}, {"role": "user", "content": "Compute: {{expression}}"}]',
 '[{"name":"expression","type":"string","required":true,"default":"","desc":"Arithmetic expression"}]',
 NULL,
 NULL,
 true)
ON CONFLICT (key, version) DO NOTHING;

INSERT INTO app.prompt_examples (key, version, title, description, category, technique, tags, prompt_template, variables, response_format, json_schema_template, is_active)
VALUES
('bugfix-patch', 1, 'Bugfix patch (final diff only)', 'Fix bugs in code and output unified diff patch', 'coding', 'zero_shot', ARRAY['coding', 'bugfix', 'patch'],
 '[{"role": "system", "content": "Analyze step by step privately. Final output must be a unified diff patch only."}, {"role": "user", "content": "Here is the file; fix the bug described.\n\nDescription:\n{{bug_desc}}\n\nFile:\n{{file_text}}"}]',
 '[{"name":"bug_desc","type":"string","required":true,"default":"","desc":"Bug description"},{"name":"file_text","type":"string","required":true,"default":"","desc":"Full file contents"}]',
 NULL,
 NULL,
 true)
ON CONFLICT (key, version) DO NOTHING;

INSERT INTO app.prompt_examples (key, version, title, description, category, technique, tags, prompt_template, variables, response_format, json_schema_template, is_active)
VALUES
('decompose-solve', 1, 'Decompose then solve', 'Break complex tasks into ordered subproblems', 'qa', 'zero_shot', ARRAY['decomposition', 'reasoning'],
 '[{"role": "system", "content": "Break the task into ordered subproblems, solve each briefly, then provide a concise final answer."}, {"role": "user", "content": "{{task}}"}]',
 '[{"name":"task","type":"string","required":true,"default":"","desc":"Complex task to decompose"}]',
 NULL,
 NULL,
 true)
ON CONFLICT (key, version) DO NOTHING;

-- FEW-SHOT examples
INSERT INTO app.prompt_examples (key, version, title, description, category, technique, tags, prompt_template, variables, default_examples, response_format, json_schema_template, is_active)
VALUES
('sentiment-classifier', 1, 'Sentiment (pos/neg/neutral)', 'Classify sentiment with few-shot examples', 'classification', 'few_shot', ARRAY['sentiment', 'classification'],
 '[{"role": "system", "content": "Classify sentiment as positive, negative, or neutral. Respond with label, confidence of it and brief explanation, justification of your label."}, {"role": "user", "content": "Classify: {{text}}"}]',
 '[{"name":"text","type":"string","required":true,"default":"","desc":"Sentence to classify"}]',
 '[{"input":"I love this product!","output":"positive"},{"input":"This is the worst.","output":"negative"},{"input":"It is okay, I guess.","output":"neutral"}]',
 NULL,
 NULL,
 true)
ON CONFLICT (key, version) DO NOTHING;

INSERT INTO app.prompt_examples (key, version, title, description, category, technique, tags, prompt_template, variables, default_examples, response_format, json_schema_template, is_active)
VALUES
('code-review', 1, 'Code review feedback', 'Provide code review feedback with examples', 'coding', 'few_shot', ARRAY['code-review', 'feedback'],
 '[{"role": "system", "content": "You are an experienced code reviewer. Provide constructive feedback on the following code."}, {"role": "user", "content": "Code to review:\n{{code}}"}]',
 '[{"name":"code","type":"string","required":true,"default":"","desc":"Code snippet to review"}]',
 '[{"input":"def add(a, b):\n    return a + b","output":"Good: Simple and clear. Consider adding type hints and docstring."},{"input":"x = 5\ny = x + 1\nz = y * 2","output":"Good: Clear variable names. Consider using more descriptive names like ''count'' instead of single letters."}]',
 NULL,
 NULL,
 true)
ON CONFLICT (key, version) DO NOTHING;

INSERT INTO app.prompt_examples (key, version, title, description, category, technique, tags, prompt_template, variables, default_examples, response_format, json_schema_template, is_active)
VALUES
('email-tone', 1, 'Email tone converter', 'Convert email tone with examples', 'writing', 'few_shot', ARRAY['email', 'tone', 'writing'],
 '[{"role": "system", "content": "Convert the following email to the specified tone while preserving the core message."}, {"role": "user", "content": "Target tone: {{target_tone}}\n\nEmail:\n{{email_text}}"}]',
 '[{"name":"email_text","type":"string","required":true,"default":"","desc":"Original email text"},{"name":"target_tone","type":"string","required":false,"default":"professional","desc":"Target tone (professional, casual, formal)"}]',
 '[{"input":"Hey, can you send me that file? Thanks!","output":"Hello, could you please send me that file when you have a moment? Thank you!","target_tone":"professional"},{"input":"I need this done ASAP.","output":"I would appreciate it if this could be completed as soon as possible.","target_tone":"professional"}]',
 NULL,
 NULL,
 true)
ON CONFLICT (key, version) DO NOTHING;

-- PROMPT_CHAIN examples
INSERT INTO app.prompt_examples (key, version, title, description, category, technique, tags, prompt_template, variables, response_format, json_schema_template, is_active)
VALUES
('qa-with-context', 1, 'Q&A with context', 'Answer questions using provided context', 'qa', 'prompt_chain', ARRAY['qa', 'context'],
 '[{"role": "system", "content": "Use the following context to answer the question. If the answer cannot be found in the context, say \"I don''t have enough information to answer this question.\""}, {"role": "user", "content": "Context:\n{{context}}\n\nQuestion: {{question}}"}]',
 '[{"name":"context","type":"string","required":true,"default":"","desc":"Contextual information"},{"name":"question","type":"string","required":true,"default":"","desc":"Question to answer"}]',
 NULL,
 NULL,
 true)
ON CONFLICT (key, version) DO NOTHING;

INSERT INTO app.prompt_examples (key, version, title, description, category, technique, tags, prompt_template, variables, response_format, json_schema_template, is_active)
VALUES
('text-extract', 1, 'Extract information', 'Extract structured information from text', 'extraction', 'zero_shot', ARRAY['extraction', 'structured'],
 '[{"role": "system", "content": "Extract the following information from the text and format as JSON: Name, Email, Phone, Company."}, {"role": "user", "content": "Text:\n{{text}}"}]',
 '[{"name":"text","type":"string","required":true,"default":"","desc":"Text to extract information from"}]',
 'json_schema',
 '{"type": "object", "properties": {"name": {"type": "string", "description": "Full name"}, "email": {"type": "string", "description": "Email address"}, "phone": {"type": "string", "description": "Phone number"}, "company": {"type": "string", "description": "Company name"}}, "required": ["name", "email", "phone", "company"]}',
 true)
ON CONFLICT (key, version) DO NOTHING;

INSERT INTO app.prompt_examples (key, version, title, description, category, technique, tags, prompt_template, variables, response_format, json_schema_template, is_active)
VALUES
('explain-concept', 1, 'Explain concept simply', 'Explain complex concepts in simple terms', 'education', 'zero_shot', ARRAY['explanation', 'education'],
 '[{"role": "system", "content": "Explain the following concept in simple terms that a beginner can understand. Use analogies and examples."}, {"role": "user", "content": "Concept: {{concept}}"}]',
 '[{"name":"concept","type":"string","required":true,"default":"","desc":"Concept to explain"}]',
 NULL,
 NULL,
 true)
ON CONFLICT (key, version) DO NOTHING;
EOSQL

echo "[init] app data inserted."

