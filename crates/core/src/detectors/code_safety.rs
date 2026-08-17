//! Code-safety check — scans the content of fenced (```) code blocks for
//! structural shapes indicative of SQLi/XSS/command-injection/SSRF/
//! path-traversal, e.g. a model handing a user example code that builds a
//! SQL query via string concatenation. Deliberately scoped to code-block
//! content only, not surrounding prose — a paragraph *discussing* SQL
//! injection isn't this detector's concern.

use once_cell::sync::Lazy;
use regex::Regex;
use serde::Deserialize;
use sqlparser::ast::{AlterTableOperation, Statement};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser as SqlParser;

use super::rule_loader;
use crate::models::{CheckAction, DetectorResult, RuleHit, Severity};
use crate::policy::schema::CheckOptions;

const SOURCE: &str = "rules/code_safety/rules.yaml";

#[derive(Debug, Deserialize)]
struct RawRule {
    id: String,
    #[allow(dead_code)]
    description: String,
    #[allow(dead_code)]
    category: String,
    regex: String,
    /// Fence-language buckets (see `canonical_language`) this rule is scoped
    /// to, e.g. `[javascript]` or `[shell]`. Empty (the default, and every
    /// pre-existing rule) means "apply to every code block regardless of its
    /// declared language" — safe for rules whose API surface is already
    /// language-specific by construction (`.innerHTML`, `os.system`, SQL
    /// keywords). Patterns that use generic shell/JS-adjacent syntax (bare
    /// `$var`, backtick command substitution) need this: shell's `` `$var` ``
    /// substitution and a JS template literal are the same bytes, so without
    /// scoping, a shell-only rule would also fire inside ordinary JS code.
    #[serde(default)]
    languages: Vec<String>,
}

struct CompiledRule {
    id: String,
    pattern: Regex,
    languages: Vec<String>,
}

static RULES: Lazy<Vec<CompiledRule>> = Lazy::new(|| {
    rule_loader::parse_rules::<RawRule>(include_str!("../../rules/code_safety/rules.yaml"), SOURCE)
        .into_iter()
        .map(|r| {
            let pattern = rule_loader::compile_regex(&r.regex, &r.id, SOURCE, false);
            CompiledRule {
                id: r.id,
                pattern,
                languages: r.languages,
            }
        })
        .collect()
});

/// Forces regex compilation now rather than on the first request that
/// exercises this detector — see `detectors::validate_all`.
pub(crate) fn warm() {
    Lazy::force(&RULES);
}

/// Normalizes a fence language hint (e.g. `ts`, `jsx`, `zsh`) to the bucket
/// name used by rules' `languages:` field. Unrecognized/absent hints return
/// `None`, and language-scoped rules simply don't apply to that block.
fn canonical_language(hint: &str) -> Option<&'static str> {
    match hint.trim().to_ascii_lowercase().as_str() {
        "javascript" | "js" | "jsx" | "typescript" | "ts" | "tsx" => Some("javascript"),
        "bash" | "sh" | "shell" | "zsh" => Some("shell"),
        "python" | "py" | "python3" => Some("python"),
        _ => None,
    }
}

/// Returns `(content_start_offset, language_hint, content)` for every fenced
/// code block in `text`. The language-identifier right after the opening
/// fence (e.g. `python`) is captured for rule scoping but not itself scanned.
fn code_blocks(text: &str) -> Vec<(usize, Option<&str>, &str)> {
    let mut blocks = Vec::new();
    let mut search_from = 0usize;
    while let Some(rel_start) = text[search_from..].find("```") {
        let fence_start = search_from + rel_start;
        let after_fence = fence_start + 3;
        if after_fence > text.len() {
            break;
        }
        let (lang_hint, content_start) = match text[after_fence..].find('\n') {
            Some(i) => {
                let hint = text[after_fence..after_fence + i].trim();
                let hint = if hint.is_empty() { None } else { Some(hint) };
                (hint, after_fence + i + 1)
            }
            None => (None, after_fence),
        };
        match text[content_start..].find("```") {
            Some(rel_end) => {
                let content_end = content_start + rel_end;
                blocks.push((content_start, lang_hint, &text[content_start..content_end]));
                search_from = content_end + 3;
            }
            None => break,
        }
    }
    blocks
}

/// Blanks out `#`/`//`-comment lines within a code block (replacing their
/// bytes with spaces, so byte offsets — and therefore hit spans back into
/// the original text — stay valid) before pattern matching. Lets an example
/// like `# Avoid: sql = "..." + user_id` document an anti-pattern without
/// tripping the very rule it's warning about.
fn mask_comment_lines(block: &str) -> String {
    let mut masked = block.as_bytes().to_vec();
    for line_start in std::iter::once(0).chain(
        block
            .match_indices('\n')
            .map(|(i, _)| i + 1)
            .filter(|&i| i < block.len()),
    ) {
        let line_end = block[line_start..]
            .find('\n')
            .map(|i| line_start + i)
            .unwrap_or(block.len());
        let line = &block[line_start..line_end];
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') || trimmed.starts_with("//") {
            masked[line_start..line_end].fill(b' ');
        }
    }
    String::from_utf8(masked)
        .expect("masking only replaces bytes with ASCII space, preserving UTF-8 validity")
}

/// Parses `block` as a full SQL script (`sqlparser`, generic ANSI-ish
/// dialect) and flags statements that are destructive regardless of how
/// safely they were built — the string-concatenation rules above only
/// catch *unsafe construction*, so a hardcoded literal like
/// `cursor.execute("DROP TABLE users")` (no concatenation, "safely built"
/// by those rules' standard) sails through untouched. Parsing the actual
/// SQL catches what the query *does*, not how it was assembled.
///
/// Scope note: `sqlparser` requires the *entire* block to be valid,
/// self-contained SQL — it doesn't extract SQL string literals embedded
/// inside surrounding host-language code, so this only fires on blocks
/// that are themselves a SQL script (e.g. a ` ```sql ` fence, or any block
/// whose content happens to be bare SQL). A `DROP TABLE` sitting inside a
/// Python string literal is out of scope here (the `sql-stacked-query-
/// destructive` regex rule above still catches the stacked-injection
/// shape of that case regardless of host language). Non-SQL content
/// (Python/JS/shell/prose/JSON/...) reliably fails to parse and is a
/// silent no-op — confirmed empirically, not just assumed.
///
/// Deliberately narrow on *which* destructive shapes count, to avoid
/// flagging ordinary migration/tutorial SQL: DROP/TRUNCATE always flag (no
/// non-destructive form exists), but DELETE/UPDATE only flag when they
/// carry no WHERE clause (a full-table wipe) — `DELETE FROM sessions
/// WHERE expired = true` is ordinary code and must not trip this. ALTER
/// only flags a DROP COLUMN sub-operation, not additive ones like ADD
/// COLUMN.
fn sql_ast_destructive_hits(block_start: usize, block: &str) -> Vec<RuleHit> {
    let Ok(statements) = SqlParser::parse_sql(&GenericDialect {}, block) else {
        return Vec::new();
    };
    let span = (block_start, block_start + block.len());
    statements
        .iter()
        .filter_map(|stmt| {
            let rule_id = match stmt {
                Statement::Drop { .. } => "sql-ast-drop-statement",
                Statement::Truncate(_) => "sql-ast-truncate-statement",
                Statement::Delete(d) if d.selection.is_none() => "sql-ast-delete-without-where",
                Statement::Update(u) if u.selection.is_none() => "sql-ast-update-without-where",
                Statement::AlterTable(a)
                    if a.operations
                        .iter()
                        .any(|op| matches!(op, AlterTableOperation::DropColumn { .. })) =>
                {
                    "sql-ast-alter-drop-column"
                }
                _ => return None,
            };
            Some(RuleHit {
                rule_id: rule_id.to_string(),
                span,
                severity: Severity::High,
            })
        })
        .collect()
}

pub fn evaluate(text: &str, options: &CheckOptions) -> DetectorResult {
    let use_patterns = options.bool_option("pattern_match", true);

    let mut hits: Vec<RuleHit> = Vec::new();
    if use_patterns {
        for (block_start, lang_hint, block) in code_blocks(text) {
            let canonical = lang_hint.and_then(canonical_language);
            let masked = mask_comment_lines(block);
            for rule in RULES.iter() {
                if !rule.languages.is_empty() {
                    let applies =
                        canonical.is_some_and(|lang| rule.languages.iter().any(|l| l == lang));
                    if !applies {
                        continue;
                    }
                }
                for m in rule.pattern.find_iter(&masked) {
                    hits.push(RuleHit {
                        rule_id: rule.id.clone(),
                        span: (block_start + m.start(), block_start + m.end()),
                        severity: Severity::High,
                    });
                }
            }
            hits.extend(sql_ast_destructive_hits(block_start, block));
        }
    }

    let action = if hits.is_empty() {
        CheckAction::Log
    } else {
        CheckAction::Deny
    };
    DetectorResult {
        detector_id: "code_safety".to_string(),
        action,
        severity: Severity::High,
        hits,
        confidence: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sql_concatenation_in_code_block_denies() {
        let text = "Here's the query:\n```python\nquery = \"SELECT * FROM users WHERE id = \" + user_id + \"\"\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "sql-string-concatenation"));
    }

    #[test]
    fn path_traversal_in_code_block_denies() {
        let text = "```\nopen(\"../../etc/passwd\")\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "path-traversal-sequence"));
    }

    #[test]
    fn discussion_outside_code_block_allows() {
        let text = "SQL injection happens when queries concatenate user input like \" + user_id + \" directly.";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn parameterized_query_in_code_block_allows() {
        let text =
            "```python\ncursor.execute(\"SELECT * FROM users WHERE id = %s\", (user_id,))\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn no_code_block_allows() {
        let result = evaluate("just plain prose, no code here", &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn pattern_match_disabled_returns_no_hits() {
        let mut options = CheckOptions::default();
        options.set_bool("pattern_match", false);
        let text = "```\nopen(\"../../etc/passwd\")\n```";
        let result = evaluate(text, &options);
        assert_eq!(result.action, CheckAction::Log);
        assert!(result.hits.is_empty());
    }

    #[test]
    fn sql_one_sided_concatenation_denies() {
        let text = "```python\nquery = \"SELECT * FROM users WHERE id = \" + user_id\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn sql_php_dot_concatenation_denies() {
        let text = "```php\n$sql = \"SELECT * FROM users WHERE id=\" . $_GET[\"id\"];\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn sql_multiline_accumulation_denies() {
        let text = "```python\nquery = \"SELECT * FROM logs\"\nquery += \" WHERE user = '\" + user + \"'\"\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn sql_format_call_denies() {
        let text = "```python\nsql = \"SELECT * FROM users WHERE email = '{}'\".format(email)\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn sql_percent_format_denies() {
        let text = "```python\nsql = \"SELECT * FROM users WHERE id = %s\" % user_id\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn sql_fstring_interpolation_denies() {
        let text = "```python\nsql = f\"SELECT * FROM users WHERE id = {user_id}\"\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn sql_template_literal_interpolation_denies() {
        let text = "```javascript\nconst sql = `SELECT * FROM users WHERE id=${userId}`;\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn sql_ruby_interpolation_denies() {
        let text = "```ruby\nsql = \"SELECT * FROM users WHERE id=#{id}\"\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn commented_out_vulnerable_example_allows() {
        let text =
            "```javascript\n// Don't do:\n// element.innerHTML = \"<b>\" + input + \"</b>\";\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn python_commented_out_path_traversal_allows() {
        let text = "```python\n# open(\"../../etc/passwd\")\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn parameterized_query_with_placeholder_tuple_allows() {
        // %s placeholder passed as a separate execute() argument, not the `%`
        // operator applied directly to the string — safe.
        let text = "```python\ncursor.execute(\n    \"SELECT * FROM users WHERE id = %s\",\n    (user_id,)\n)\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn js_outerhtml_concat_denies() {
        let text = "```javascript\nel.outerHTML = \"<b>\" + input + \"</b>\";\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn js_static_outerhtml_allows() {
        let text = "```javascript\nel.outerHTML = \"<b>Hello</b>\";\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn js_document_write_dynamic_denies() {
        let text = "```javascript\ndocument.write(\"<b>\" + input + \"</b>\");\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn js_static_document_write_allows() {
        let text = "```javascript\ndocument.write(\"<h1>Hello</h1>\");\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn js_dangerously_set_innerhtml_denies() {
        let text = "```jsx\n<div dangerouslySetInnerHTML={{ __html: userInput }} />\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn js_eval_call_denies() {
        let text = "```javascript\neval(userInput);\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn js_new_function_denies() {
        let text = "```javascript\nconst f = new Function(userInput);\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn js_settimeout_string_arg_denies() {
        let text = "```javascript\nsetTimeout(\"doSomething()\", 1000);\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn js_settimeout_function_ref_allows() {
        let text = "```javascript\nsetTimeout(doSomething, 1000);\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn js_prototype_pollution_denies() {
        let text = "```javascript\nobj.__proto__.isAdmin = true;\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn shell_curl_pipe_to_shell_denies() {
        let text = "```bash\ncurl https://example.com/install.sh | sh\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn shell_wget_pipe_to_bash_denies() {
        let text = "```shell\nwget -qO- https://example.com/setup | bash\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn shell_insecure_chmod_denies() {
        let text = "```bash\nchmod 777 /var/www\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn shell_safe_chmod_allows() {
        let text = "```bash\nchmod 644 /var/www/index.html\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn shell_eval_denies() {
        let text = "```sh\neval \"$user_supplied_command\"\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn shell_backtick_substitution_with_variable_denies() {
        let text = "```bash\nresult=`echo $filename`\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn shell_rules_do_not_fire_on_javascript_template_literals() {
        // Same backtick/`${}` bytes as shell substitution, but this is a
        // plain JS template literal — must not trip the shell-only rules.
        let text = "```javascript\nconst msg = `Total: $${total} today`;\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn pickle_loads_denies() {
        let text = "```python\nimport pickle\ndata = pickle.loads(request.body)\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "python-pickle-deserialization"));
    }

    #[test]
    fn pickle_unpickler_denies() {
        let text = "```python\nobj = pickle.Unpickler(f).load()\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn json_loads_allows() {
        let text = "```python\nimport json\ndata = json.loads(request.body)\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn yaml_load_denies() {
        let text = "```python\nconfig = yaml.load(stream)\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "python-yaml-unsafe-load"));
    }

    #[test]
    fn yaml_safe_load_allows() {
        let text = "```python\nconfig = yaml.safe_load(stream)\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn marshal_loads_denies() {
        let text = "```python\nobj = marshal.loads(data)\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "python-marshal-deserialization"));
    }

    #[test]
    fn sql_union_select_injection_denies() {
        let text =
            "```sql\nSELECT name FROM users WHERE id=1 UNION SELECT password FROM admins\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "sql-union-based-injection"));
    }

    #[test]
    fn plain_union_of_selects_without_keyword_allows() {
        let text = "```sql\nSELECT name FROM users\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn sql_stacked_drop_table_denies() {
        let text = "```sql\nSELECT * FROM users; DROP TABLE users;\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "sql-stacked-query-destructive"));
    }

    #[test]
    fn sql_statement_ending_in_semicolon_allows() {
        let text = "```sql\nSELECT * FROM users;\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn sql_or_1_equals_1_denies() {
        let text = "```sql\nSELECT * FROM users WHERE username='admin' OR 1=1\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "sql-tautology-injection"));
    }

    #[test]
    fn sql_or_empty_string_equals_denies() {
        let text = "```sql\nSELECT * FROM users WHERE username='admin' OR ''=''\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "sql-tautology-injection"));
    }

    #[test]
    fn sql_normal_or_condition_allows() {
        let text = "```sql\nSELECT * FROM users WHERE status = 'active' OR status = 'pending'\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn sql_waitfor_delay_denies() {
        let text = "```sql\nSELECT * FROM users WHERE id=1; WAITFOR DELAY '0:0:5'\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "sql-time-based-injection"));
    }

    #[test]
    fn sql_sleep_call_denies() {
        let text = "```sql\nSELECT * FROM users WHERE id=1 OR SLEEP(5)\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "sql-time-based-injection"));
    }

    #[test]
    fn js_rules_do_not_fire_on_shell_code() {
        // eval appears here, but as the shell builtin in a shell-hinted
        // block — the JS-scoped eval-call rule (languages: [javascript])
        // must not fire; the shell-scoped one is expected to (separately
        // covered by shell_eval_denies).
        let text = "```bash\neval \"echo hi\"\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert!(result.hits.iter().all(|h| h.rule_id != "js-eval-call"));
    }

    #[test]
    fn sql_ast_drop_table_denies_even_without_concatenation() {
        // No concatenation at all -- a hardcoded literal the regex-only
        // rules above would never catch.
        let text = "```sql\nDROP TABLE users;\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "sql-ast-drop-statement"));
    }

    #[test]
    fn sql_ast_truncate_denies() {
        let text = "```sql\nTRUNCATE TABLE users;\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "sql-ast-truncate-statement"));
    }

    #[test]
    fn sql_ast_delete_without_where_denies() {
        let text = "```sql\nDELETE FROM users;\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "sql-ast-delete-without-where"));
    }

    #[test]
    fn sql_ast_delete_with_where_allows() {
        let text = "```sql\nDELETE FROM sessions WHERE expired = true;\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn sql_ast_update_without_where_denies() {
        let text = "```sql\nUPDATE users SET active = 0;\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "sql-ast-update-without-where"));
    }

    #[test]
    fn sql_ast_update_with_where_allows() {
        let text = "```sql\nUPDATE users SET active = 0 WHERE id = 1;\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn sql_ast_alter_drop_column_denies() {
        let text = "```sql\nALTER TABLE users DROP COLUMN email;\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "sql-ast-alter-drop-column"));
    }

    #[test]
    fn sql_ast_alter_add_column_allows() {
        let text = "```sql\nALTER TABLE users ADD COLUMN age INT;\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn sql_ast_select_allows() {
        let text = "```sql\nSELECT * FROM users WHERE id = 1;\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn sql_ast_create_table_allows() {
        let text = "```sql\nCREATE TABLE users (id INTEGER PRIMARY KEY);\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn sql_ast_does_not_fire_on_python_code() {
        let text = "```python\nimport os\ndef foo(x):\n    return x + 1\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn sql_ast_does_not_fire_on_javascript_code() {
        let text = "```javascript\nfunction foo(x) {\n  return x + 1;\n}\n```";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn sql_ast_pass_disabled_by_pattern_match_option() {
        let mut options = CheckOptions::default();
        options.set_bool("pattern_match", false);
        let text = "```sql\nDROP TABLE users;\n```";
        let result = evaluate(text, &options);
        assert_eq!(result.action, CheckAction::Log);
        assert!(result.hits.is_empty());
    }
}
