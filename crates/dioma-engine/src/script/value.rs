//! Значения стека ВМ (порт a1-script). Аналог CAOSVar из C2e.

/// Универсальное значение. `Entity` хранит сырой id агента — крейт не знает
/// про агентов ничего, интерпретация на стороне хоста.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Nul,
    Int(i64),
    Float(f32),
    Str(String),
    Entity(u64),
}

impl Value {
    /// Числовое прочтение: Int/Float конвертируются, Nul читается как 0
    /// (неинициализированные переменные — это ноль), остальное — ошибка.
    pub fn as_num(&self) -> Result<f64, String> {
        match self {
            Value::Int(i) => Ok(*i as f64),
            Value::Float(f) => Ok(*f as f64),
            Value::Nul => Ok(0.0),
            other => Err(format!("ожидалось число, получено {other:?}")),
        }
    }

    pub fn as_int(&self) -> Result<i64, String> {
        match self {
            Value::Int(i) => Ok(*i),
            Value::Float(f) => Ok(*f as i64),
            Value::Nul => Ok(0),
            other => Err(format!("ожидалось целое, получено {other:?}")),
        }
    }

    pub fn as_str(&self) -> Result<&str, String> {
        match self {
            Value::Str(s) => Ok(s),
            other => Err(format!("ожидалась строка, получено {other:?}")),
        }
    }

    pub fn as_entity(&self) -> Result<u64, String> {
        match self {
            Value::Entity(id) => Ok(*id),
            other => Err(format!("ожидался агент, получено {other:?}")),
        }
    }

    /// Истина для всего ненулевого.
    pub fn truthy(&self) -> bool {
        match self {
            Value::Nul => false,
            Value::Int(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
            Value::Str(s) => !s.is_empty(),
            Value::Entity(id) => *id != 0,
        }
    }

    pub fn num(v: f64) -> Value {
        let r = v.round();
        if v == r && v.abs() < 9e15 {
            Value::Int(r as i64)
        } else {
            Value::Float(v as f32)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coercions() {
        assert_eq!(Value::Int(5).as_num().unwrap(), 5.0);
        assert_eq!(Value::Float(2.5).as_num().unwrap(), 2.5);
        assert_eq!(
            Value::Str("x".into()).as_num(),
            Err("ожидалось число, получено Str(\"x\")".to_string())
        );
        assert_eq!(Value::num(3.0), Value::Int(3));
        assert_eq!(Value::num(0.5), Value::Float(0.5));
    }
}
