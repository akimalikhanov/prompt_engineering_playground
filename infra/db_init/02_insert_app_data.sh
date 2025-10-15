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

-- ZERO-SHOT example
INSERT INTO app.prompt_examples (technique_key, title, version, status, language, messages, variables, model_hint)
VALUES
('zero-shot','Bullet summary (3 points)',1,'active','en',
 '[
   {"role":"system","content":"You are a concise assistant. Output only the result."},
   {"role":"user","content":"Summarize the following text in exactly 3 bullet points:\\n{{text}}"}
 ]',
 '[{"name":"text","type":"string","required":true,"desc":"Text to summarize"}]',
 'Any general model')
ON CONFLICT (technique_key, title, version) DO NOTHING;

-- FEW-SHOT example
INSERT INTO app.prompt_examples (technique_key, title, version, status, language, messages, variables)
VALUES
('few-shot','Sentiment (pos/neg/neutral)',1,'active','en',
 '[
   {"role":"system","content":"Classify sentiment as positive, negative, or neutral. Respond with one label only."},
   {"role":"user","content":"I love this product!"},{"role":"assistant","content":"positive"},
   {"role":"user","content":"This is the worst."},{"role":"assistant","content":"negative"},
   {"role":"user","content":"It is okay, I guess."},{"role":"assistant","content":"neutral"},
   {"role":"user","content":"Classify: {{text}}"}
 ]',
 '[{"name":"text","type":"string","required":true,"desc":"Sentence to classify"}]')
ON CONFLICT (technique_key, title, version) DO NOTHING;

-- COT examples (2 variants)
INSERT INTO app.prompt_examples (technique_key, title, version, status, language, messages, variables)
VALUES
('cot','Math step-by-step (final answer only)',1,'active','en',
 '[
   {"role":"system","content":"Reason step by step privately. In the final answer, output only the numeric result."},
   {"role":"user","content":"Compute: {{expression}}"}
 ]',
 '[{"name":"expression","type":"string","required":true,"desc":"Arithmetic expression"}]')
ON CONFLICT (technique_key, title, version) DO NOTHING;

INSERT INTO app.prompt_examples (technique_key, title, version, status, language, messages, variables)
VALUES
('cot','Bugfix patch (final diff only)',1,'active','en',
 '[
   {"role":"system","content":"Analyze step by step privately. Final output must be a unified diff patch only."},
   {"role":"user","content":"Here is the file; fix the bug described.\\n\\nDescription:\\n{{bug_desc}}\\n\\nFile:\\n{{file_text}}"}
 ]',
 '[{"name":"bug_desc","type":"string","required":true,"desc":"Bug description"},
   {"name":"file_text","type":"string","required":true,"desc":"Full file contents"}]')
ON CONFLICT (technique_key, title, version) DO NOTHING;

-- REACT example
INSERT INTO app.prompt_examples (technique_key, title, version, status, language, messages, variables, model_hint)
VALUES
('react','ReAct template (search + calc)',1,'active','en',
 '[
   {"role":"system","content":"Follow the ReAct format. Use cycles of Thought/Action/Observation. Final answer must be under \"Final Answer:\"."},
   {"role":"system","content":"Available tools: Search(query), Calculator(expression)"},
   {"role":"user","content":"{{question}}"}
 ]',
 '[{"name":"question","type":"string","required":true,"desc":"User query needing search and/or calculation"}]',
 'Best with models that support tool use')
ON CONFLICT (technique_key, title, version) DO NOTHING;

-- LEAST-TO-MOST example
INSERT INTO app.prompt_examples (technique_key, title, version, status, language, messages, variables)
VALUES
('least-to-most','Decompose then solve',1,'active','en',
 '[
   {"role":"system","content":"Break the task into ordered subproblems, solve each briefly, then provide a concise final answer."},
   {"role":"user","content":"{{task}}"}
 ]',
 '[{"name":"task","type":"string","required":true,"desc":"Complex task to decompose"}]')
ON CONFLICT (technique_key, title, version) DO NOTHING;
EOSQL

echo "[init] app data inserted."

