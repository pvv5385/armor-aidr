//! Malicious-URL check — heuristics over any `http(s)://`, `ftp://`,
//! protocol-relative (`//host/...`), or `data:` URI found in the input:
//! bare-IP host, punycode homoglyph host, embedded userinfo credentials,
//! known link-shortener domains, suspicious TLDs, excessive subdomain
//! depth, homoglyph/typosquat lookalikes of a popular-domain list, and
//! `data:` URIs (an XSS/payload-smuggling vector, flagged outright since
//! they have no host to run the other heuristics against).
//!
//! `domain_allowlist`/`domain_blocklist` are deployment-supplied
//! (no shipped defaults — there's no universal "safe" or "known-bad"
//! domain list) and are checked before the heuristics: an allowlisted host
//! short-circuits with no hits at all, a blocklisted host short-circuits
//! with a single `url-blocklisted-domain` hit, either way skipping the
//! heuristic checks below as moot.
//!
//! Deliberately narrower than a real-time reputation/blocklist lookup (no
//! network calls — `armor-core` has none) — this is the zero-dependency,
//! deterministic slice: structural properties of the URL
//! itself that are cheap to check and rarely appear in benign links.
//! Heuristic regexes match against the URL's *authority* segment
//! (`user:pass@host:port`), not the full URL, so path/query text can't
//! trigger a false positive.
//!
//! The homoglyph/typosquat/suspicious-TLD/subdomain-depth checks are
//! hand-off Rust logic rather than YAML-driven regex, same as the
//! pre-existing shortener check: a confusables table + edit distance isn't
//! expressible as a single regex.

use once_cell::sync::Lazy;
use regex::Regex;
use serde::Deserialize;

use super::rule_loader;
use crate::homoglyphs;
use crate::models::{CheckAction, DetectorResult, RuleHit, Severity};
use crate::policy::schema::CheckOptions;

const SOURCE: &str = "rules/malicious_url/rules.yaml";

#[derive(Debug, Deserialize)]
struct RawRule {
    id: String,
    #[allow(dead_code)]
    description: String,
    provider: String,
    regex: String,
}

struct CompiledRule {
    id: String,
    provider: String,
    pattern: Regex,
}

static RULES: Lazy<Vec<CompiledRule>> = Lazy::new(|| {
    rule_loader::parse_rules::<RawRule>(
        include_str!("../../rules/malicious_url/rules.yaml"),
        SOURCE,
    )
    .into_iter()
    .map(|r| {
        let pattern = rule_loader::compile_regex(&r.regex, &r.id, SOURCE, false);
        CompiledRule {
            id: r.id,
            provider: r.provider,
            pattern,
        }
    })
    .collect()
});

/// Forces regex compilation now rather than on the first request that
/// exercises this detector — see `detectors::validate_all`.
pub(crate) fn warm() {
    Lazy::force(&RULES);
}

static SHORTENERS: Lazy<Vec<String>> = Lazy::new(|| {
    rule_loader::parse_rules(
        include_str!("../../rules/malicious_url/shorteners.yaml"),
        "rules/malicious_url/shorteners.yaml",
    )
});

static SUSPICIOUS_TLDS: Lazy<Vec<String>> = Lazy::new(|| {
    rule_loader::parse_rules(
        include_str!("../../rules/malicious_url/suspicious_tlds.yaml"),
        "rules/malicious_url/suspicious_tlds.yaml",
    )
});

static DEFAULT_POPULAR_DOMAINS: Lazy<Vec<String>> = Lazy::new(|| {
    rule_loader::parse_rules(
        include_str!("../../rules/malicious_url/popular_domains.yaml"),
        "rules/malicious_url/popular_domains.yaml",
    )
});

/// Full Wagner-Fischer edit-distance matrix. Domain labels are short
/// (well under 100 chars), so the O(n*m) memory footprint of the plain
/// matrix isn't worth trading away for the rolling-row optimization.
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (rows, cols) = (a.len() + 1, b.len() + 1);

    let mut dist = vec![vec![0usize; cols]; rows];
    for (i, row) in dist.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in dist[0].iter_mut().enumerate() {
        *cell = j;
    }

    for i in 1..rows {
        for j in 1..cols {
            let sub_cost = usize::from(a[i - 1] != b[j - 1]);
            dist[i][j] = (dist[i - 1][j] + 1)
                .min(dist[i][j - 1] + 1)
                .min(dist[i - 1][j - 1] + sub_cost);
        }
    }
    dist[rows - 1][cols - 1]
}

fn base_domain(host: &str) -> String {
    let labels: Vec<&str> = host.split('.').collect();
    if labels.len() < 2 {
        return host.to_string();
    }
    labels[labels.len() - 2..].join(".")
}

/// Scores `host` against the popular-domain baseline: if it contains
/// confusable characters, find the closest baseline domain by edit distance
/// after homoglyph normalization; otherwise (plain ASCII) look for a
/// single-character typo of a baseline domain. Returns the matching rule id
/// for whichever case fired.
fn homoglyph_or_typosquat_hit(host: &str, popular: &[String]) -> Option<&'static str> {
    let normalized = homoglyphs::fold(host);

    if normalized != host {
        let candidate = base_domain(&normalized);
        let nearest = popular
            .iter()
            .map(|p| {
                if normalized == *p || candidate == *p {
                    0
                } else {
                    levenshtein(&candidate, p)
                }
            })
            .min()
            .unwrap_or(usize::MAX);
        return (nearest <= 1).then_some("url-homoglyph-domain");
    }

    let candidate = base_domain(host);
    let nearest_distinct = popular
        .iter()
        .filter(|p| candidate != **p)
        .map(|p| levenshtein(&candidate, p))
        .min()
        .unwrap_or(usize::MAX);
    (nearest_distinct == 1).then_some("url-typosquat-domain")
}

static URL_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?i)(?:https?://|ftp://|//|data:)[^\s<>"']+"#).unwrap());

/// Whether `matched` (a [`URL_RE`] match) is a `data:` URI rather than a
/// URL with a host — it has no authority segment to run the other checks
/// against, so callers should handle it separately.
fn is_data_uri(matched: &str) -> bool {
    matched
        .get(..5)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("data:"))
}

/// The `user:pass@host:port` segment of a URL, i.e. everything between the
/// scheme (`scheme://`, or a bare `//` for protocol-relative URLs) and the
/// first `/`, `?`, or `#`.
fn authority_of(url: &str) -> &str {
    let after_scheme = url
        .split_once("://")
        .map_or_else(|| url.strip_prefix("//").unwrap_or(url), |(_, rest)| rest);
    let end = after_scheme
        .find(['/', '?', '#'])
        .unwrap_or(after_scheme.len());
    &after_scheme[..end]
}

/// The authority with any `userinfo@` prefix and `:port` suffix stripped.
fn host_of(authority: &str) -> &str {
    let after_userinfo = authority.rsplit('@').next().unwrap_or(authority);
    match after_userinfo.rfind(':') {
        Some(idx) => &after_userinfo[..idx],
        None => after_userinfo,
    }
}

fn is_shortener(host: &str) -> bool {
    domain_list_matches(host, &SHORTENERS)
}

/// True if `host` is exactly one of `list`'s domains, or a subdomain of one
/// (e.g. `accounts.chase.com` matches a `chase.com` entry).
fn domain_list_matches<S: AsRef<str>>(host: &str, list: &[S]) -> bool {
    let host = host.trim_end_matches('.');
    list.iter().any(|domain| {
        let domain = domain.as_ref();
        host.eq_ignore_ascii_case(domain)
            || host.to_ascii_lowercase().ends_with(&format!(".{domain}"))
    })
}

fn suspicious_tld(host: &str) -> bool {
    let host = host.to_ascii_lowercase();
    SUSPICIOUS_TLDS
        .iter()
        .any(|tld| host.ends_with(tld.as_str()))
}

pub fn evaluate(text: &str, options: &CheckOptions) -> DetectorResult {
    let ip_literal_host = options.bool_option("ip_literal_host", true);
    let punycode = options.bool_option("punycode", true);
    let credentials_in_url = options.bool_option("credentials_in_url", true);
    let shorteners = options.bool_option("shorteners", true);
    let homoglyph_typosquat = options.bool_option("homoglyph_typosquat", true);
    let suspicious_tld_check = options.bool_option("suspicious_tld", true);
    let excessive_subdomains = options.bool_option("excessive_subdomains", true);
    let data_uri = options.bool_option("data_uri", true);
    let max_subdomain_depth = options.f64_option("max_subdomain_depth", 3.0) as usize;

    let domain_allowlist = options.str_list_option("domain_allowlist");
    let domain_blocklist = options.str_list_option("domain_blocklist");

    let extra_popular = options.str_list_option("popular_domains");
    let popular: Vec<String> = if extra_popular.is_empty() {
        DEFAULT_POPULAR_DOMAINS.clone()
    } else {
        DEFAULT_POPULAR_DOMAINS
            .iter()
            .cloned()
            .chain(extra_popular)
            .collect()
    };

    let active_rules: Vec<&CompiledRule> = RULES
        .iter()
        .filter(|r| match r.provider.as_str() {
            "ip_literal_host" => ip_literal_host,
            "punycode" => punycode,
            "credentials_in_url" => credentials_in_url,
            _ => false,
        })
        .collect();

    let mut hits: Vec<RuleHit> = Vec::new();
    for m in URL_RE.find_iter(text) {
        if is_data_uri(m.as_str()) {
            if data_uri {
                hits.push(RuleHit {
                    rule_id: "url-data-uri".to_string(),
                    span: (m.start(), m.end()),
                    severity: Severity::High,
                });
            }
            continue;
        }

        let authority = authority_of(m.as_str());
        let host = host_of(authority).to_ascii_lowercase();

        if domain_list_matches(&host, &domain_allowlist) {
            continue;
        }

        if domain_list_matches(&host, &domain_blocklist) {
            hits.push(RuleHit {
                rule_id: "url-blocklisted-domain".to_string(),
                span: (m.start(), m.end()),
                severity: Severity::High,
            });
            continue;
        }

        for rule in &active_rules {
            let subject = if rule.provider == "credentials_in_url" {
                authority
            } else {
                host.as_str()
            };
            if rule.pattern.is_match(subject) {
                hits.push(RuleHit {
                    rule_id: rule.id.clone(),
                    span: (m.start(), m.end()),
                    severity: Severity::High,
                });
            }
        }

        if shorteners && is_shortener(&host) {
            hits.push(RuleHit {
                rule_id: "url-known-shortener".to_string(),
                span: (m.start(), m.end()),
                severity: Severity::Medium,
            });
        }

        if suspicious_tld_check && suspicious_tld(&host) {
            hits.push(RuleHit {
                rule_id: "url-suspicious-tld".to_string(),
                span: (m.start(), m.end()),
                severity: Severity::Low,
            });
        }

        if excessive_subdomains {
            let depth = host.split('.').count();
            if depth > max_subdomain_depth + 2 {
                hits.push(RuleHit {
                    rule_id: "url-excessive-subdomains".to_string(),
                    span: (m.start(), m.end()),
                    severity: Severity::Medium,
                });
            }
        }

        if homoglyph_typosquat {
            if let Some(rule_id) = homoglyph_or_typosquat_hit(&host, &popular) {
                hits.push(RuleHit {
                    rule_id: rule_id.to_string(),
                    span: (m.start(), m.end()),
                    severity: Severity::High,
                });
            }
        }
    }

    let action = if hits.is_empty() {
        CheckAction::Log
    } else {
        CheckAction::Deny
    };
    DetectorResult {
        detector_id: "malicious_url".to_string(),
        action,
        severity: Severity::High,
        hits,
        confidence: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(pairs: &[(&str, bool)]) -> CheckOptions {
        let mut o = CheckOptions::default();
        for (k, v) in pairs {
            o.set_bool(k, *v);
        }
        o
    }

    fn all_enabled() -> CheckOptions {
        opts(&[
            ("ip_literal_host", true),
            ("punycode", true),
            ("credentials_in_url", true),
            ("shorteners", true),
            ("homoglyph_typosquat", true),
            ("suspicious_tld", true),
            ("excessive_subdomains", true),
            ("data_uri", true),
        ])
    }

    #[test]
    fn ip_literal_host_denies() {
        let result = evaluate("click http://203.0.113.5/login", &all_enabled());
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "url-ip-literal-host"));
    }

    #[test]
    fn punycode_host_denies() {
        let result = evaluate("visit https://xn--pple-43d.com/verify", &all_enabled());
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result.hits.iter().any(|h| h.rule_id == "url-punycode-host"));
    }

    #[test]
    fn credentials_in_authority_denies() {
        let result = evaluate(
            "login at https://accounts.google.com@evil.example/reset",
            &all_enabled(),
        );
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "url-credentials-in-authority"));
    }

    #[test]
    fn known_shortener_denies() {
        let result = evaluate("see https://bit.ly/3abcXyz for details", &all_enabled());
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "url-known-shortener"));
    }

    #[test]
    fn ordinary_url_allows() {
        let result = evaluate(
            "see https://docs.rs/regex/latest/regex/ for the crate docs",
            &all_enabled(),
        );
        assert_eq!(result.action, CheckAction::Log);
        assert!(result.hits.is_empty());
    }

    #[test]
    fn disabled_provider_is_not_checked() {
        let result = evaluate(
            "click http://203.0.113.5/login",
            &opts(&[("ip_literal_host", false)]),
        );
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn no_url_present_allows() {
        let result = evaluate("no links here at all", &all_enabled());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn cyrillic_homoglyph_domain_denies() {
        // "gооgle.com" with Cyrillic о (U+043E) instead of Latin o.
        let text = "verify at https://g\u{043E}\u{043E}gle.com/verify";
        let result = evaluate(text, &all_enabled());
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "url-homoglyph-domain"));
    }

    #[test]
    fn typosquat_domain_denies() {
        let result = evaluate("go to https://paypa1.com/login", &all_enabled());
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "url-typosquat-domain"));
    }

    #[test]
    fn suspicious_tld_denies() {
        let result = evaluate("check https://freegift.xyz/claim", &all_enabled());
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "url-suspicious-tld"));
    }

    #[test]
    fn excessive_subdomains_denies() {
        let result = evaluate("see https://a.b.c.d.e.example.com/page", &all_enabled());
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "url-excessive-subdomains"));
    }

    #[test]
    fn homoglyph_check_disabled_by_option() {
        let text = "verify at https://g\u{043E}\u{043E}gle.com/verify";
        let result = evaluate(text, &opts(&[("homoglyph_typosquat", false)]));
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn custom_popular_domain_extends_default_list() {
        let mut o = all_enabled();
        o.set_str_list("popular_domains", &["mybank.com"]);
        let result = evaluate("login at https://mybank1.com/", &o);
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "url-typosquat-domain"));
    }

    #[test]
    fn data_uri_denies() {
        let result = evaluate(
            "click here: data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==",
            &all_enabled(),
        );
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result.hits.iter().any(|h| h.rule_id == "url-data-uri"));
    }

    #[test]
    fn data_uri_check_disabled_by_option() {
        let result = evaluate(
            "click here: data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==",
            &opts(&[("data_uri", false)]),
        );
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn ftp_url_ip_literal_host_denies() {
        let result = evaluate("get it from ftp://203.0.113.5/file.zip", &all_enabled());
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "url-ip-literal-host"));
    }

    #[test]
    fn protocol_relative_known_shortener_denies() {
        let result = evaluate("see //bit.ly/3abcXyz for details", &all_enabled());
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "url-known-shortener"));
    }

    #[test]
    fn no_config_allows_arbitrary_domain() {
        let result = evaluate("see https://example.com/page", &all_enabled());
        assert_eq!(result.action, CheckAction::Log);
        assert!(result.hits.is_empty());
    }

    #[test]
    fn blocklisted_domain_denies() {
        let mut o = all_enabled();
        o.set_str_list("domain_blocklist", &["evil.example"]);
        let result = evaluate("see https://evil.example/phish", &o);
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "url-blocklisted-domain"));
    }

    #[test]
    fn blocklist_matches_subdomains() {
        let mut o = all_enabled();
        o.set_str_list("domain_blocklist", &["evil.example"]);
        let result = evaluate("see https://login.evil.example/phish", &o);
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "url-blocklisted-domain"));
    }

    #[test]
    fn allowlisted_domain_suppresses_all_hits() {
        // bit.ly would otherwise trip url-known-shortener; allowlisting it
        // must skip that (and every other) heuristic, not flag it anyway.
        let mut o = all_enabled();
        o.set_str_list("domain_allowlist", &["bit.ly"]);
        let result = evaluate("see https://bit.ly/3abcXyz for details", &o);
        assert_eq!(result.action, CheckAction::Log);
        assert!(result.hits.is_empty());
    }

    #[test]
    fn allowlist_takes_priority_over_blocklist() {
        let mut o = all_enabled();
        o.set_str_list("domain_allowlist", &["mybank.com"]);
        o.set_str_list("domain_blocklist", &["mybank.com"]);
        let result = evaluate("see https://mybank.com/login", &o);
        assert_eq!(result.action, CheckAction::Log);
        assert!(result.hits.is_empty());
    }
}
