//! PyO3 bindings for .taas v2 — игровой формат: именованные секции + реплей.

#![allow(non_snake_case)]

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use pillars_connectome::{ConnectomeData, NeuronGroup, NtPhenotype};

use crate::taas_format;

use super::connectome_brain::PyConnectomeData;

// ─── Connectome: секции вместо компонентов ───────────────────

/// Записать коннектом секциями: row_ptr/col_indices/values/nt_types/
/// receptor_types + JSON-секции groups и phenotype.
pub fn save_connectome(writer: &mut taas_format::TaasWriter, conn: &ConnectomeData) {
    writer
        .i32("row_ptr", &conn.row_ptr)
        .i32("col_indices", &conn.col_indices)
        .f32("values", &conn.values)
        .f32("nt_types", &conn.nt_types)
        .u8("receptor_types", &conn.receptor_types);
    let groups: Vec<serde_json::Value> = conn
        .groups
        .iter()
        .map(|g| serde_json::json!({"name": g.name, "indices": g.indices}))
        .collect();
    writer.json("groups", &groups);
    writer.json("phenotype", &conn.phenotype);
}

/// Прочитать коннектом из секций.
pub fn load_connectome(reader: &taas_format::TaasReader) -> Result<ConnectomeData, String> {
    let row_ptr = reader.get_i32("row_ptr")?;
    let col_indices = reader.get_i32("col_indices")?;
    let values = reader.get_f32("values")?;
    let nt_types = reader.get_f32("nt_types")?;
    let receptor_types = if reader.has_section("receptor_types") {
        reader.get_u8("receptor_types")?
    } else {
        Vec::new()
    };
    let groups: Vec<NeuronGroup> = reader.get_json("groups")?;
    let phenotype: NtPhenotype = reader.get_json("phenotype").unwrap_or_default();
    let n_neurons = row_ptr.len().saturating_sub(1);
    Ok(ConnectomeData {
        n_neurons,
        n_synapses: values.len(),
        row_ptr,
        col_indices,
        values,
        nt_types,
        receptor_types,
        groups,
        phenotype,
    })
}

/// Save game state (sections) to .taas file.
#[pyfunction]
#[pyo3(name = "save_taas", signature = (path, metadata=None))]
pub fn py_save_taas(path: &str, metadata: Option<Vec<(String, String)>>) -> PyResult<()> {
    let mut writer = taas_format::TaasWriter::new();
    writer.game("dioma");
    if let Some(md) = metadata {
        for (k, v) in md {
            writer.add_metadata(&k, &v);
        }
    }
    writer
        .to_file(path)
        .map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)
}

/// Load game state from .taas file: возвращает dict с метаданными и
/// всеми секциями (по kind: f32/i32/u8/str/json).
#[pyfunction]
#[pyo3(name = "load_taas")]
pub fn py_load_taas(path: &str, py: Python<'_>) -> PyResult<PyObject> {
    let reader = taas_format::TaasReader::from_file(path)
        .map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)?;
    let dict = PyDict::new(py);

    for sec in reader.list_sections() {
        let name = sec.name.as_str();
        let value: PyObject = match sec.kind.as_str() {
            "f32" => reader.get_f32(name).unwrap_or_default().into_pyobject(py)?.into_any().unbind(),
            "i32" => reader.get_i32(name).unwrap_or_default().into_pyobject(py)?.into_any().unbind(),
            "u8" => reader.get_u8(name).unwrap_or_default().into_pyobject(py)?.into_any().unbind(),
            "str" => reader.get_str(name).unwrap_or_default().into_pyobject(py)?.into_any().unbind(),
            "json" => {
                let v: serde_json::Value = reader.get_json(name).unwrap_or_default();
                serde_json::to_string(&v).unwrap_or_default().into_pyobject(py)?.into_any().unbind()
            }
            _ => PyBytes::new(py, &reader.get_blob(name).unwrap_or_default()).into(),
        };
        dict.set_item(name, value)?;
    }

    let meta = PyDict::new(py);
    meta.set_item("version", reader.manifest.version.as_str())?;
    meta.set_item("created", reader.manifest.created.as_str())?;
    if let Some(game) = &reader.manifest.game {
        meta.set_item("game", game.as_str())?;
    }
    meta.set_item("n_sections", reader.list_sections().len())?;
    for (k, v) in &reader.manifest.metadata {
        meta.set_item(k.as_str(), v.as_str())?;
    }
    dict.set_item("_meta", meta)?;

    Ok(dict.into())
}

/// Save connectome (ConnectomeData) to .taas file.
#[pyfunction]
#[pyo3(name = "save_connectome", signature = (path, connectome, metadata=None))]
pub fn py_save_connectome(
    path: &str,
    connectome: &PyConnectomeData,
    metadata: Option<Vec<(String, String)>>,
) -> PyResult<()> {
    let mut writer = taas_format::TaasWriter::new();
    writer.game("dioma");
    if let Some(md) = metadata {
        for (k, v) in md {
            writer.add_metadata(&k, &v);
        }
    }
    save_connectome(&mut writer, &connectome.inner);
    writer
        .to_file(path)
        .map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)
}

/// Load connectome (ConnectomeData) from .taas file.
#[pyfunction]
#[pyo3(name = "load_connectome")]
pub fn py_load_connectome(path: &str) -> PyResult<PyConnectomeData> {
    let reader = taas_format::TaasReader::from_file(path)
        .map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)?;
    let conn = load_connectome(&reader)
        .map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)?;
    Ok(PyConnectomeData { inner: conn })
}

/// Save full model (connectome + bridge + decoder) to a single .taas file.
#[pyfunction]
#[pyo3(name = "save_model", signature = (path, connectome, bridge_weights, bridge_bias, decoder_weights, decoder_bias, metadata=None))]
pub fn py_save_model(
    path: &str,
    connectome: &super::connectome_brain::PyConnectomeData,
    bridge_weights: Vec<f32>,
    bridge_bias: Vec<f32>,
    decoder_weights: Vec<f32>,
    decoder_bias: Vec<f32>,
    metadata: Option<Vec<(String, String)>>,
) -> PyResult<()> {
    let mut writer = taas_format::TaasWriter::new();
    writer.game("dioma").add_metadata("type", "Model");
    if let Some(md) = metadata {
        for (k, v) in md {
            writer.add_metadata(&k, &v);
        }
    }
    save_connectome(&mut writer, &connectome.inner);
    writer
        .f32("bridge_w", &bridge_weights)
        .f32("bridge_b", &bridge_bias)
        .f32("decoder_w", &decoder_weights)
        .f32("decoder_b", &decoder_bias);
    writer
        .to_file(path)
        .map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)
}

/// Inspect .taas file. Returns human-readable string.
#[pyfunction]
#[pyo3(name = "inspect_taas")]
pub fn py_inspect_taas(path: &str) -> PyResult<String> {
    taas_format::inspect_taas(path).map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)
}

// ═══════════════════════════════════════════════════════════
//  Lazy TaasReader — loads sections on demand
// ═══════════════════════════════════════════════════════════

/// Lazy .taas v2 reader — loads sections on demand.
///
/// Usage:
/// ```python
/// reader = TaasReader("model.taas")
/// print(reader.sections)  # list of section names
/// data = reader.get_f32("values")  # loads only this section
/// ```
#[pyclass(name = "TaasReader")]
pub(crate) struct PyTaasReader {
    reader: crate::taas_format::TaasReader,
}

#[pymethods]
impl PyTaasReader {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let reader = crate::taas_format::TaasReader::from_file(path)
            .map_err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>)?;
        Ok(Self { reader })
    }

    /// List all section names in the archive.
    #[getter]
    fn sections(&self) -> Vec<String> {
        self.reader.list_sections().iter().map(|s| s.name.clone()).collect()
    }

    /// Number of sections.
    #[getter]
    fn n_sections(&self) -> usize {
        self.reader.list_sections().len()
    }

    /// Total file size in bytes.
    #[getter]
    fn total_size(&self) -> usize {
        self.reader.total_size()
    }

    /// Metadata as a Python dict.
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for (k, v) in self.reader.metadata() {
            dict.set_item(k.as_str(), v.as_str())?;
        }
        dict.set_item("version", self.reader.manifest.version.as_str())?;
        dict.set_item("created", self.reader.manifest.created.as_str())?;
        Ok(dict.into())
    }

    /// Get a section's raw bytes as `bytes` (Python).
    /// Raises KeyError if section not found.
    fn get_raw(&self, name: &str, py: Python<'_>) -> PyResult<PyObject> {
        self.reader
            .section(name)
            .map_err(PyErr::new::<pyo3::exceptions::PyKeyError, _>)?;
        let data = self
            .reader
            .get_blob(name)
            .map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)?;
        Ok(PyBytes::new(py, &data).into())
    }

    /// Get a section as a list of floats.
    /// Raises KeyError if not found, TypeError if wrong kind.
    fn get_f32(&self, name: &str) -> PyResult<Vec<f32>> {
        self.reader.get_f32(name).map_err(|e| {
            if e.contains("not found") {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>(e)
            } else if e.contains("is not f32") {
                PyErr::new::<pyo3::exceptions::PyTypeError, _>(e)
            } else {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(e)
            }
        })
    }

    /// Get a section as a list of ints.
    fn get_i32(&self, name: &str) -> PyResult<Vec<i32>> {
        self.reader.get_i32(name).map_err(|e| {
            if e.contains("not found") {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>(e)
            } else if e.contains("is not i32") {
                PyErr::new::<pyo3::exceptions::PyTypeError, _>(e)
            } else {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(e)
            }
        })
    }

    /// Section info for a given name: name, kind, count, size, compression.
    fn info(&self, name: &str, py: Python<'_>) -> PyResult<PyObject> {
        let entry = self
            .reader
            .section(name)
            .map_err(PyErr::new::<pyo3::exceptions::PyKeyError, _>)?;
        let dict = PyDict::new(py);
        dict.set_item("name", entry.name.as_str())?;
        dict.set_item("kind", entry.kind.as_str())?;
        dict.set_item("count", entry.count)?;
        dict.set_item("size", entry.size)?;
        dict.set_item("compression", entry.compression.as_deref())?;
        if let Some(us) = entry.uncompressed_size {
            dict.set_item("uncompressed_size", us)?;
        }
        Ok(dict.into())
    }

    /// Section info for all sections: list of dicts.
    fn list_info(&self, py: Python<'_>) -> PyResult<PyObject> {
        let list = PyList::empty(py);
        for sec in self.reader.list_sections() {
            let dict = PyDict::new(py);
            dict.set_item("name", sec.name.as_str())?;
            dict.set_item("kind", sec.kind.as_str())?;
            dict.set_item("count", sec.count)?;
            dict.set_item("size", sec.size)?;
            dict.set_item("compression", sec.compression.as_deref())?;
            list.append(dict)?;
        }
        Ok(list.into())
    }

    /// Validate archive integrity (CRC32 checks).
    /// Raises IOError if validation fails.
    fn validate(&self) -> PyResult<()> {
        self.reader
            .validate()
            .map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)
    }

    /// Load the connectome from this archive (if present).
    fn get_connectome(&self) -> PyResult<Option<PyConnectomeData>> {
        if !self.reader.has_section("row_ptr") {
            return Ok(None);
        }
        match load_connectome(&self.reader) {
            Ok(conn) => Ok(Some(PyConnectomeData { inner: conn })),
            Err(_) => Ok(None),
        }
    }
}
