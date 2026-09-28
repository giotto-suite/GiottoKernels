//! The Arrow C data and C stream interfaces, declared directly.
//!
//! Both are frozen ABIs (https://arrow.apache.org/docs/format/CDataInterface.html),
//! so declaring the three structs here is cheaper than depending on the arrow
//! crate, which would be most of the build. Only what the kernels read is
//! implemented: a struct array of fixed-width, null-free columns.

use std::ffi::{c_char, c_int, c_void, CStr};

#[repr(C)]
pub struct ArrowSchema {
    pub format: *const c_char,
    pub name: *const c_char,
    pub metadata: *const c_char,
    pub flags: i64,
    pub n_children: i64,
    pub children: *mut *mut ArrowSchema,
    pub dictionary: *mut ArrowSchema,
    pub release: Option<unsafe extern "C" fn(*mut ArrowSchema)>,
    pub private_data: *mut c_void,
}

#[repr(C)]
pub struct ArrowArray {
    pub length: i64,
    pub null_count: i64,
    pub offset: i64,
    pub n_buffers: i64,
    pub n_children: i64,
    pub buffers: *mut *const c_void,
    pub children: *mut *mut ArrowArray,
    pub dictionary: *mut ArrowArray,
    pub release: Option<unsafe extern "C" fn(*mut ArrowArray)>,
    pub private_data: *mut c_void,
}

#[repr(C)]
pub struct ArrowArrayStream {
    pub get_schema: Option<unsafe extern "C" fn(*mut ArrowArrayStream, *mut ArrowSchema) -> c_int>,
    pub get_next: Option<unsafe extern "C" fn(*mut ArrowArrayStream, *mut ArrowArray) -> c_int>,
    pub get_last_error: Option<unsafe extern "C" fn(*mut ArrowArrayStream) -> *const c_char>,
    pub release: Option<unsafe extern "C" fn(*mut ArrowArrayStream)>,
    pub private_data: *mut c_void,
}

/// A column the caller requires: its name and Arrow format string
/// (`"i"` int32, `"g"` float64).
pub struct Field {
    pub name: &'static str,
    pub format: &'static str,
}

/// Borrowed view of a stream the R side owns. The stream itself is never
/// released here; every schema and array obtained from it is, on every path.
pub struct Stream {
    ptr: *mut ArrowArrayStream,
}

impl Stream {
    /// # Safety
    /// `ptr` must point to a live, not-yet-released ArrowArrayStream that
    /// outlives this value.
    pub unsafe fn from_ptr(ptr: *mut ArrowArrayStream) -> Result<Self, String> {
        if ptr.is_null() || (*ptr).release.is_none() {
            return Err("stream is null or already released".into());
        }
        if (*ptr).get_schema.is_none() || (*ptr).get_next.is_none() {
            return Err("stream has no get_schema / get_next callbacks".into());
        }
        Ok(Stream { ptr })
    }

    fn last_error(&self) -> String {
        unsafe {
            match (*self.ptr).get_last_error {
                Some(f) => {
                    let m = f(self.ptr);
                    if m.is_null() {
                        "unknown stream error".into()
                    } else {
                        CStr::from_ptr(m).to_string_lossy().into_owned()
                    }
                }
                None => "unknown stream error".into(),
            }
        }
    }

    /// Map each required field to its child index in the stream's struct
    /// schema, checking its type. Extra columns are allowed and ignored.
    pub fn resolve(&self, want: &[Field]) -> Result<Vec<Resolved>, String> {
        unsafe {
            let mut sch: ArrowSchema = std::mem::zeroed();
            if ((*self.ptr).get_schema.unwrap())(self.ptr, &mut sch) != 0 {
                return Err(format!("get_schema failed: {}", self.last_error()));
            }
            let out = resolve_schema(&sch, want);
            if let Some(rel) = sch.release {
                rel(&mut sch);
            }
            out
        }
    }

    /// Next batch, as owned copies of the requested children. `None` at end
    /// of stream. The Arrow array is released before this returns.
    pub fn next_batch(&self, cols: &[Resolved]) -> Result<Option<(usize, Vec<Column>)>, String> {
        unsafe {
            let mut arr: ArrowArray = std::mem::zeroed();
            if ((*self.ptr).get_next.unwrap())(self.ptr, &mut arr) != 0 {
                return Err(format!("get_next failed: {}", self.last_error()));
            }
            if arr.release.is_none() {
                return Ok(None);
            }
            let out = copy_children(&arr, cols);
            (arr.release.unwrap())(&mut arr);
            out.map(Some)
        }
    }
}

/// A required column located in the schema: its child index and its checked
/// Arrow format. Arrays carry no type of their own, so the format travels here.
#[derive(Clone, Copy)]
pub struct Resolved {
    pub index: usize,
    pub format: &'static str,
}

pub enum Column {
    I32(Vec<i32>),
    F64(Vec<f64>),
}

unsafe fn cstr(p: *const c_char) -> &'static [u8] {
    if p.is_null() {
        b""
    } else {
        CStr::from_ptr(p).to_bytes()
    }
}

unsafe fn resolve_schema(sch: &ArrowSchema, want: &[Field]) -> Result<Vec<Resolved>, String> {
    if cstr(sch.format) != b"+s" {
        return Err("stream is not a struct (record batch) stream".into());
    }
    let mut idx = Vec::with_capacity(want.len());
    for f in want {
        let mut found = None;
        for k in 0..sch.n_children as usize {
            let c = &**sch.children.add(k);
            if cstr(c.name) == f.name.as_bytes() {
                let got = cstr(c.format);
                if got != f.format.as_bytes() {
                    return Err(format!(
                        "column '{}' has Arrow type '{}', expected '{}'",
                        f.name,
                        String::from_utf8_lossy(got),
                        f.format
                    ));
                }
                found = Some(k);
                break;
            }
        }
        match found {
            Some(k) => idx.push(Resolved { index: k, format: f.format }),
            None => return Err(format!("stream has no column '{}'", f.name)),
        }
    }
    Ok(idx)
}

unsafe fn copy_children(arr: &ArrowArray, cols: &[Resolved]) -> Result<(usize, Vec<Column>), String> {
    let n = arr.length as usize;
    let mut out = Vec::with_capacity(cols.len());
    for r in cols {
        if r.index >= arr.n_children as usize {
            return Err("batch has fewer columns than its schema".into());
        }
        let c = &**arr.children.add(r.index);
        // No validity bitmap means no nulls. With one, null_count may be -1
        // ("not computed"), which has to be treated as possibly null.
        let has_validity = c.n_buffers >= 1 && !(*c.buffers).is_null();
        if has_validity && c.null_count != 0 {
            return Err("columns must not contain nulls".into());
        }
        if n == 0 {
            out.push(match r.format {
                "g" => Column::F64(Vec::new()),
                _ => Column::I32(Vec::new()),
            });
            continue;
        }
        if c.n_buffers < 2 || (*c.buffers.add(1)).is_null() {
            return Err("column has no data buffer".into());
        }
        let off = (arr.offset + c.offset) as usize;
        let data = *c.buffers.add(1);
        out.push(match r.format {
            "g" => Column::F64(std::slice::from_raw_parts((data as *const f64).add(off), n).to_vec()),
            "i" => Column::I32(std::slice::from_raw_parts((data as *const i32).add(off), n).to_vec()),
            f => return Err(format!("unsupported Arrow format '{}'", f)),
        });
    }
    Ok((n, out))
}
