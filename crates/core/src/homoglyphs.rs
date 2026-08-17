//! Shared Unicode confusables table — Cyrillic/Greek/Latin-extended
//! look-alikes of ASCII letters, commonly used both to register homoglyph
//! phishing domains ([`crate::detectors::malicious_url`]) and to obfuscate
//! instruction text past pattern banks that only match ASCII
//! ([`crate::engine::normalize`]'s `homoglyph` view). One table, two
//! consumers, so a new confusable only needs adding once.

use std::collections::HashMap;

use once_cell::sync::Lazy;
use unicode_normalization::UnicodeNormalization;

/// Not exhaustive — the full Unicode confusables table is thousands of
/// entries; this covers the characters that actually show up in
/// registerable domain labels and hand-typed obfuscation. Fullwidth
/// Latin/digits and Roman numerals deliberately have no entries here —
/// [`fold`] runs NFKC first, which already folds those to plain ASCII.
pub(crate) static HOMOGLYPHS: Lazy<HashMap<char, char>> = Lazy::new(|| {
    HashMap::from([
        // Cyrillic -> Latin (lowercase)
        ('а', 'a'),
        ('в', 'b'),
        ('с', 'c'),
        ('ԁ', 'd'),
        ('е', 'e'),
        ('ё', 'e'),
        ('ғ', 'f'),
        ('һ', 'h'),
        ('і', 'i'),
        ('ј', 'j'),
        ('к', 'k'),
        ('ӏ', 'l'),
        ('м', 'm'),
        ('н', 'h'),
        ('о', 'o'),
        ('р', 'p'),
        ('ԛ', 'q'),
        ('г', 'r'),
        ('ѕ', 's'),
        ('т', 't'),
        ('у', 'y'),
        ('х', 'x'),
        ('ԝ', 'w'),
        // Cyrillic -> Latin (uppercase)
        ('А', 'a'),
        ('В', 'b'),
        ('С', 'c'),
        ('Е', 'e'),
        ('Н', 'h'),
        ('І', 'i'),
        ('Ј', 'j'),
        ('К', 'k'),
        ('М', 'm'),
        ('О', 'o'),
        ('Р', 'p'),
        ('Ѕ', 's'),
        ('Т', 't'),
        ('Х', 'x'),
        // Greek -> Latin (lowercase)
        ('α', 'a'),
        ('β', 'b'),
        ('ε', 'e'),
        ('η', 'n'),
        ('ι', 'i'),
        ('κ', 'k'),
        ('μ', 'm'),
        ('ν', 'v'),
        ('ο', 'o'),
        ('ρ', 'p'),
        ('τ', 't'),
        ('υ', 'u'),
        ('χ', 'x'),
        ('ω', 'w'),
        // Greek -> Latin (uppercase)
        ('Α', 'a'),
        ('Β', 'b'),
        ('Ε', 'e'),
        ('Η', 'h'),
        ('Ι', 'i'),
        ('Κ', 'k'),
        ('Μ', 'm'),
        ('Ν', 'n'),
        ('Ρ', 'p'),
        ('Τ', 't'),
        ('Χ', 'x'),
        ('Ζ', 'z'),
        // Latin-extended / IPA look-alikes
        ('ɑ', 'a'),
        ('ƅ', 'b'),
        ('ϲ', 'c'),
        ('ɛ', 'e'),
        ('ƒ', 'f'),
        ('ɡ', 'g'),
        ('ɦ', 'h'),
        ('ɩ', 'i'),
        ('ı', 'i'),
        ('ȷ', 'j'),
        ('ʝ', 'j'),
        ('ƙ', 'k'),
        ('ℓ', 'l'),
        ('ŀ', 'l'),
        ('ɫ', 'l'),
        ('ɱ', 'm'),
        ('ɴ', 'n'),
        ('ɵ', 'o'),
        ('ƿ', 'p'),
        ('ʀ', 'r'),
        ('ꝛ', 'r'),
        ('ʂ', 's'),
        ('ꜱ', 's'),
        ('ƭ', 't'),
        ('ʋ', 'v'),
        ('ɯ', 'w'),
        ('ʏ', 'y'),
        ('ʐ', 'z'),
        ('ᴢ', 'z'),
    ])
});

/// Applies Unicode NFKC normalization (collapses fullwidth/compatibility
/// forms) then maps remaining confusable characters to their ASCII
/// equivalent via [`HOMOGLYPHS`]. Safe to call on arbitrary text — a domain
/// label, a full message, anything — since it's a pure per-char fold with
/// no length or context assumptions.
pub(crate) fn fold(text: &str) -> String {
    text.nfkc()
        .map(|c| *HOMOGLYPHS.get(&c).unwrap_or(&c))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_cyrillic_lookalikes() {
        assert_eq!(fold("іgnore"), "ignore");
    }

    #[test]
    fn leaves_plain_ascii_unchanged() {
        assert_eq!(
            fold("ignore previous instructions"),
            "ignore previous instructions"
        );
    }
}
