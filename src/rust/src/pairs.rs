//! The pairs of rows a graph joins: from a stream of edges `(from_id, to_id[,
//! weight])` in node ids, and a map from node id to row, the canonical pair
//! list `(a, b, w)` with `a <= b`, edges with an end that maps to no row
//! dropped. With `dedupe`, a pair listed more than once (a directed graph
//! holding both directions of a mutual pair) becomes one pair at the mean of
//! its weights; without it every edge is kept as listed.

use crate::arrow_c::{Column, Field, Stream};

pub struct Pairs {
    pub a: Vec<i32>,
    pub b: Vec<i32>,
    pub w: Vec<f64>,
}

pub fn graph_pairs_stream(
    stream: &Stream,
    node_row: &[i32],
    weighted: bool,
    dedupe: bool,
) -> Result<Pairs, String> {
    let mut fields = vec![
        Field { name: "from_id", format: "i" },
        Field { name: "to_id", format: "i" },
    ];
    if weighted {
        fields.push(Field { name: "weight", format: "g" });
    }
    let cols = stream.resolve(&fields)?;
    let row_of = |id: i32| -> i32 {
        if id < 1 || id as usize > node_row.len() { 0 } else { node_row[id as usize - 1] }
    };

    let mut key: Vec<u64> = Vec::new();
    let mut wt: Vec<f64> = Vec::new();
    while let Some((n, mut batch)) = stream.next_batch(&cols)? {
        if n == 0 {
            continue;
        }
        let w = if weighted {
            match batch.pop() { Some(Column::F64(v)) => Some(v), _ => unreachable!() }
        } else {
            None
        };
        let to = match batch.pop() { Some(Column::I32(v)) => v, _ => unreachable!() };
        let from = match batch.pop() { Some(Column::I32(v)) => v, _ => unreachable!() };
        for k in 0..n {
            let (ra, rb) = (row_of(from[k]), row_of(to[k]));
            if ra < 1 || rb < 1 {
                continue;
            }
            let (lo, hi) = if ra <= rb { (ra, rb) } else { (rb, ra) };
            key.push(((lo as u64) << 32) | hi as u64);
            wt.push(w.as_ref().map_or(1.0, |w| w[k]));
        }
    }

    if !dedupe {
        let a = key.iter().map(|&k| (k >> 32) as i32).collect();
        let b = key.iter().map(|&k| (k & 0xffff_ffff) as i32).collect();
        return Ok(Pairs { a, b, w: wt });
    }

    let mut idx: Vec<u32> = (0..key.len() as u32).collect();
    idx.sort_unstable_by_key(|&i| (key[i as usize], i));
    let mut out = Pairs { a: Vec::new(), b: Vec::new(), w: Vec::new() };
    let mut s = 0;
    while s < idx.len() {
        let k0 = key[idx[s] as usize];
        let mut e = s;
        let mut sum = 0.0;
        while e < idx.len() && key[idx[e] as usize] == k0 {
            sum += wt[idx[e] as usize];
            e += 1;
        }
        out.a.push((k0 >> 32) as i32);
        out.b.push((k0 & 0xffff_ffff) as i32);
        out.w.push(sum / (e - s) as f64);
        s = e;
    }
    Ok(out)
}
