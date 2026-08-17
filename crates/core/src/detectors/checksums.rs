//! Shared numeric-identifier checksum validators, used by [`super::pii`] to
//! cut false positives on bare digit-sequence patterns (a 9-digit number
//! isn't a Brazilian CPF just because it has the right shape — the check
//! digits have to actually verify).
//!
//! Each algorithm here is a standard, publicly documented national
//! identifier check (Luhn, Verhoeff, ISO/national mod-11 and mod-23
//! variants) — implemented from the public specification, not ported from
//! any third party's source.

/// Strips everything but ASCII digits.
fn digits_only(s: &str) -> String {
    s.chars().filter(char::is_ascii_digit).collect()
}

/// Standard Luhn checksum (mod 10), used by payment cards and Canada's SIN.
pub fn luhn(s: &str) -> bool {
    let digits = digits_only(s);
    if digits.is_empty() {
        return false;
    }
    let mut total: u32 = 0;
    for (i, ch) in digits.chars().rev().enumerate() {
        let mut n = ch.to_digit(10).unwrap();
        if i % 2 == 1 {
            n *= 2;
            if n > 9 {
                n -= 9;
            }
        }
        total += n;
    }
    total.is_multiple_of(10)
}

// Verhoeff algorithm multiplication (d) and permutation (p) tables — the
// standard public tables (Jacobus Verhoeff, 1969).
const VERHOEFF_D: [[u8; 10]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
];

const VERHOEFF_P: [[u8; 10]; 8] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
];

/// Verhoeff check digit validation (used by India's Aadhaar).
pub fn verhoeff(s: &str) -> bool {
    let digits = digits_only(s);
    if digits.is_empty() {
        return false;
    }
    let mut c: usize = 0;
    for (i, ch) in digits.chars().rev().enumerate() {
        let n = ch.to_digit(10).unwrap() as usize;
        c = VERHOEFF_D[c][VERHOEFF_P[i % 8][n] as usize] as usize;
    }
    c == 0
}

fn checked_digit_sum(digits: &[u32], weights: &[u32]) -> u32 {
    digits.iter().zip(weights).map(|(d, w)| d * w).sum()
}

fn to_digit_vec(digits: &str) -> Vec<u32> {
    digits.chars().filter_map(|c| c.to_digit(10)).collect()
}

/// True if every digit is identical (e.g. "00000000000"). These sequences
/// satisfy the CPF/CNPJ mod-11 arithmetic trivially (all-zero weighted sums),
/// but every reference implementation explicitly rejects them as invalid.
fn all_same_digit(digits: &str) -> bool {
    let mut chars = digits.chars();
    match chars.next() {
        Some(first) => chars.all(|c| c == first),
        None => false,
    }
}

/// Brazil CPF (11 digits, two mod-11 check digits).
pub fn cpf(s: &str) -> bool {
    let digits = digits_only(s);
    if digits.len() != 11 || all_same_digit(&digits) {
        return false;
    }
    let d = to_digit_vec(&digits);
    let weights1: Vec<u32> = (2..=10).rev().collect();
    let sum1 = checked_digit_sum(&d[0..9], &weights1);
    let rem1 = sum1 % 11;
    let check1 = if rem1 < 2 { 0 } else { 11 - rem1 };
    if check1 != d[9] {
        return false;
    }
    let weights2: Vec<u32> = (2..=11).rev().collect();
    let sum2 = checked_digit_sum(&d[0..10], &weights2);
    let rem2 = sum2 % 11;
    let check2 = if rem2 < 2 { 0 } else { 11 - rem2 };
    check2 == d[10]
}

/// Brazil CNPJ (14 digits, two mod-11 check digits).
pub fn cnpj(s: &str) -> bool {
    let digits = digits_only(s);
    if digits.len() != 14 || all_same_digit(&digits) {
        return false;
    }
    let d = to_digit_vec(&digits);
    const W1: [u32; 12] = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2];
    let sum1 = checked_digit_sum(&d[0..12], &W1);
    let rem1 = sum1 % 11;
    let check1 = if rem1 < 2 { 0 } else { 11 - rem1 };
    if check1 != d[12] {
        return false;
    }
    const W2: [u32; 13] = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2];
    let sum2 = checked_digit_sum(&d[0..13], &W2);
    let rem2 = sum2 % 11;
    let check2 = if rem2 < 2 { 0 } else { 11 - rem2 };
    check2 == d[13]
}

/// Spain NIF (8 digits + letter) / NIE (X/Y/Z + 7 digits + letter), mod-23 letter check.
pub fn es_nif(s: &str) -> bool {
    const LETTERS: &str = "TRWAGMYFPDXBNJZSQVHLCKE";
    let cleaned: String = s.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    if cleaned.len() != 9 {
        return false;
    }
    let chars: Vec<char> = cleaned.chars().collect();
    let last = chars[8].to_ascii_uppercase();
    let mut number_str: String = chars[0..8].iter().collect();
    match chars[0].to_ascii_uppercase() {
        'X' => number_str = format!("0{}", &number_str[1..]),
        'Y' => number_str = format!("1{}", &number_str[1..]),
        'Z' => number_str = format!("2{}", &number_str[1..]),
        _ => {}
    }
    let Ok(n) = number_str.parse::<u32>() else {
        return false;
    };
    LETTERS.chars().nth((n % 23) as usize) == Some(last)
}

/// UK NHS number (10 digits, mod-11 check digit, weights 10..2).
pub fn uk_nhs(s: &str) -> bool {
    let digits = digits_only(s);
    if digits.len() != 10 {
        return false;
    }
    let d = to_digit_vec(&digits);
    let weights: Vec<u32> = (2..=10).rev().collect();
    let sum = checked_digit_sum(&d[0..9], &weights);
    let rem = sum % 11;
    if rem == 1 {
        return false; // no valid check digit exists for this base
    }
    let check = if rem == 0 { 0 } else { 11 - rem };
    check == d[9]
}

/// Netherlands BSN 11-proef: weighted sum (9..2, -1) over 9 digits must be divisible by 11.
pub fn nl_bsn(s: &str) -> bool {
    let digits = digits_only(s);
    if digits.len() != 9 {
        return false;
    }
    let d = to_digit_vec(&digits);
    const WEIGHTS: [i32; 9] = [9, 8, 7, 6, 5, 4, 3, 2, -1];
    let sum: i32 = d.iter().zip(WEIGHTS).map(|(&d, w)| d as i32 * w).sum();
    sum % 11 == 0
}

/// Australia TFN, weighted mod-11 (weights for the 9-digit form).
pub fn au_tfn(s: &str) -> bool {
    let digits = digits_only(s);
    if digits.len() != 8 && digits.len() != 9 {
        return false;
    }
    let d = to_digit_vec(&digits);
    let weights: &[u32] = if digits.len() == 9 {
        &[1, 4, 3, 7, 5, 8, 6, 9, 10]
    } else {
        &[10, 7, 8, 4, 6, 3, 5, 1]
    };
    checked_digit_sum(&d, weights).is_multiple_of(11)
}

/// Australia ABN, weighted mod-89 (first digit reduced by 1).
pub fn au_abn(s: &str) -> bool {
    let digits = digits_only(s);
    if digits.len() != 11 {
        return false;
    }
    let d = to_digit_vec(&digits);
    const WEIGHTS: [i64; 11] = [10, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19];
    let first = d[0] as i64 - 1;
    let rest_sum: i64 = d[1..]
        .iter()
        .zip(&WEIGHTS[1..])
        .map(|(&d, &w)| d as i64 * w)
        .sum();
    let total = first * WEIGHTS[0] + rest_sum;
    total.rem_euclid(89) == 0
}

/// Australia ACN, weighted mod-10 check digit over the first 8 digits.
pub fn au_acn(s: &str) -> bool {
    let digits = digits_only(s);
    if digits.len() != 9 {
        return false;
    }
    let d = to_digit_vec(&digits);
    const WEIGHTS: [u32; 8] = [8, 7, 6, 5, 4, 3, 2, 1];
    let sum = checked_digit_sum(&d[0..8], &WEIGHTS);
    let check = (10 - (sum % 10)) % 10;
    check == d[8]
}

/// IBAN checksum per ISO 7064 mod 97-10 (ISO 13616 §4): move the 4-character
/// country-code+check-digit prefix to the end, expand each letter to its
/// two-digit ordinal (A=10 .. Z=35), and the resulting numeral string must be
/// congruent to 1 mod 97. Computed digit-by-digit so it never has to
/// materialize the (up to ~34-digit) number as an actual integer.
pub fn iban_mod97(s: &str) -> bool {
    let cleaned: String = s
        .chars()
        .filter(|c| !c.is_whitespace() && *c != '-')
        .map(|c| c.to_ascii_uppercase())
        .collect();
    let chars: Vec<char> = cleaned.chars().collect();
    if chars.len() < 5 || !chars[0].is_ascii_alphabetic() || !chars[1].is_ascii_alphabetic() {
        return false;
    }

    let rearranged = chars[4..].iter().chain(chars[0..4].iter());
    let mut remainder: u32 = 0;
    for &c in rearranged {
        let value = if c.is_ascii_digit() {
            c.to_digit(10).unwrap()
        } else if c.is_ascii_alphabetic() {
            c as u32 - 'A' as u32 + 10
        } else {
            return false;
        };
        remainder = if value >= 10 {
            (remainder * 100 + value) % 97
        } else {
            (remainder * 10 + value) % 97
        };
    }
    remainder == 1
}

/// Indonesia KTP/NIK: 16 digits; positions 7-12 (1-indexed) encode DD/MM/YY
/// (women's DOB has 40 added to the day). Sanity-checks that the embedded
/// date is plausible rather than verifying an actual checksum (KTP has none).
pub fn id_ktp_date_sanity(s: &str) -> bool {
    let digits = digits_only(s);
    if digits.len() != 16 {
        return false;
    }
    let d = to_digit_vec(&digits);
    let mut day = (d[6] * 10 + d[7]) as i32;
    let month = (d[8] * 10 + d[9]) as i32;
    if day > 40 {
        day -= 40;
    }
    (1..=31).contains(&day) && (1..=12).contains(&month)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luhn_validates_known_visa() {
        assert!(luhn("4242424242424242"));
        assert!(!luhn("1234567890123456"));
    }

    #[test]
    fn verhoeff_validates_known_number() {
        // 2363 is a canonical Verhoeff-valid test number.
        assert!(verhoeff("2363"));
        assert!(!verhoeff("2364"));
    }

    #[test]
    fn cpf_validates_known_valid_number() {
        assert!(cpf("111.444.777-35"));
        assert!(!cpf("111.444.777-36"));
    }

    #[test]
    fn cnpj_validates_known_valid_number() {
        assert!(cnpj("11.222.333/0001-81"));
        assert!(!cnpj("11.222.333/0001-82"));
    }

    #[test]
    fn cpf_rejects_repeated_digit_sequences() {
        assert!(!cpf("000.000.000-00"));
        assert!(!cpf("111.111.111-11"));
        assert!(!cpf("999.999.999-99"));
    }

    #[test]
    fn cnpj_rejects_repeated_digit_sequences() {
        assert!(!cnpj("00.000.000/0000-00"));
        assert!(!cnpj("11.111.111/1111-11"));
    }

    #[test]
    fn es_nif_validates_known_valid_number() {
        assert!(es_nif("12345678Z"));
        assert!(!es_nif("12345678A"));
    }

    #[test]
    fn nl_bsn_validates_known_valid_number() {
        assert!(nl_bsn("111222333"));
    }

    #[test]
    fn au_acn_validates_known_valid_number() {
        // 004 085 616 is a published example ACN.
        assert!(au_acn("004085616"));
        assert!(!au_acn("004085617"));
    }

    #[test]
    fn iban_mod97_validates_known_valid_numbers() {
        // Published examples: Deutsche Bundesbank's DE sample and the
        // Wikipedia FR sample, both real-format checksum-valid IBANs.
        assert!(iban_mod97("DE89370400440532013000"));
        assert!(iban_mod97("FR1420041010050500013M02606"));
        assert!(iban_mod97("GB29 NWBK 6016 1331 9268 19"));
    }

    #[test]
    fn iban_mod97_rejects_bad_checksum() {
        assert!(!iban_mod97("DE89370400440532013001"));
    }

    #[test]
    fn id_ktp_date_sanity_accepts_plausible_dob() {
        // day=17 (female offset 40+17=57), month=08
        assert!(id_ktp_date_sanity("3171055708990001"));
    }

    #[test]
    fn id_ktp_date_sanity_rejects_bad_month() {
        assert!(!id_ktp_date_sanity("3171051513990001")); // day=15, month=13
    }
}
