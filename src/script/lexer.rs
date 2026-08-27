//! Лексер (порт a1-script): числа, строки, имена. Минус распознаётся только
//! как часть отрицательного литерала (`-5`); бинарное вычитание пишется `sub`.

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Int(i64),
    Float(f32),
    Str(String),
    Ident(String),
}

pub fn lex(src: &str) -> Result<Vec<Tok>, String> {
    let mut out = Vec::new();
    let b = src.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        match c {
            b' ' | b'\t' | b'\r' | b'\n' => i += 1,
            b'#' => {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            }
            b'"' => {
                let start = i + 1;
                let end = src[start..]
                    .find('"')
                    .ok_or("незакрытая строка")?
                    + start;
                out.push(Tok::Str(src[start..end].to_string()));
                i = end + 1;
            }
            _ if c == b'-' || c.is_ascii_digit() => {
                let start = i;
                if c == b'-' {
                    i += 1;
                }
                let mut dot = false;
                while i < b.len()
                    && (b[i].is_ascii_digit() || (b[i] == b'.' && !dot))
                {
                    if b[i] == b'.' {
                        dot = true;
                    }
                    i += 1;
                }
                let text = &src[start..i];
                if text == "-" {
                    return Err("одинокий минус: используй sub".into());
                }
                if dot {
                    out.push(Tok::Float(text.parse().map_err(|e| format!("{text:?}: {e}"))?));
                } else {
                    out.push(Tok::Int(text.parse().map_err(|e| format!("{text:?}: {e}"))?));
                }
            }
            _ if c.is_ascii_alphabetic() || c == b'_' => {
                let start = i;
                while i < b.len()
                    && (b[i].is_ascii_alphanumeric() || b[i] == b'_')
                {
                    i += 1;
                }
                out.push(Tok::Ident(src[start..i].to_ascii_lowercase()));
            }
            _ => return Err(format!("неожиданный символ {:?}", c as char)),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basics() {
        let t = lex("setv va0 -5 # коммент\nout \"привет\"").unwrap();
        assert_eq!(
            t,
            vec![
                Tok::Ident("setv".into()),
                Tok::Ident("va0".into()),
                Tok::Int(-5),
                Tok::Ident("out".into()),
                Tok::Str("привет".into()),
            ]
        );
    }

    #[test]
    fn floats_and_unknown_char() {
        assert!(matches!(lex("1.5").unwrap()[0], Tok::Float(_)));
        assert!(lex("@").is_err());
        assert!(lex("-").is_err());
    }
}
