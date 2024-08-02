use nalgebra::DVector;
use numpy::{PyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyModule};
use std::collections::HashMap;

use pyo3::prepare_freethreaded_python;



#[pyfunction]
fn tag_frames(frame_list: &PyList, players: &PyList, player_features: &PyDict)
-> PyResult<(HashMap<String, Vec<(f64, f64)>>, HashMap<String, Vec<usize>>)> {
    pyo3::prepare_freethreaded_python();
    let mut player_pos: HashMap<String, Vec<(f64, f64)>> = HashMap::new();
    let mut player_idx: HashMap<String, Vec<usize>> = HashMap::new();

    for (i, frame) in frame_list.iter().enumerate() {
        if i % 100 == 0 {
            println!("Tagging frame {} out of {}", i, frame_list.len());
        }

        // Call the Rust function to tag players in the frame
        tag_players_in_frame(frame, players, player_features)?;

        // Update player positions and indices
        let players_in_frame: &PyList = frame.call_method0("players")?.extract()?;
        for player in players_in_frame.iter() {
            let tag: String = player.getattr("tag")?.extract()?;
            let x: f64 = player.getattr("x")?.extract()?;
            let y: f64 = player.getattr("y")?.extract()?;

            player_pos.entry(tag.clone()).or_default().push((x, y));
            player_idx.entry(tag).or_default().push(i);
        }
    }

    Ok((player_pos, player_idx))
}














fn features_distance(features_a: Vec<f32>, features_b: Vec<f32>) -> f32 {

    let vec_a = DVector::from_vec(features_a);
    let vec_b = DVector::from_vec(features_b);
    let dot_product = vec_a.dot(&vec_b);
    let norm_a = vec_a.norm();
    let norm_b = vec_b.norm();

    dot_product / (norm_a * norm_b)
}

#[pyfunction]
fn get_player_features(player_features: &PyDict, tag: &str) -> PyResult<Vec<f32>> {

    let py_array = player_features.get_item(tag)?.ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("KeyError: {}", tag))
    })?;

    
    let array: &PyArray1<f32> = py_array.extract()?;
    let readonly_array = array.readonly();
    Ok(readonly_array.as_slice()?.to_vec())
}


#[pyfunction]
fn tag_players_in_frame(frame: &PyAny, players: &PyList, player_features: &PyDict) -> PyResult<()> {
    let mut matches = Vec::new();
    let mut pairs = HashMap::new();

    // Extract tags and players from Python objects
    let tags: HashMap<String, &PyAny> = players.iter()
        .map(|player| {
            let tag: String = player.getattr("tag")?.extract()?;
            Ok((tag, player))
        })
        .collect::<PyResult<_>>()?;



    let players_from_frame: Vec<&PyAny> = frame.call_method0("players")?.extract()?;

    // Calculate distance pairs
    for tag in tags.keys() {
        for obj in &players_from_frame {


            let tag_value: &PyAny = obj.getattr("tag")?;

            let tag_str: &str = tag_value.str()?.to_str()?;

            let player_in_frame_ft = get_player_features(player_features, tag_str)?;

            let tagf: &PyAny = tags[tag];

            let tagff = tagf.getattr("template_features")?.extract::<Vec<f32>>()?;

            let dist = features_distance(
                tagff,
                player_in_frame_ft,
            );


            pairs.insert((tag.clone(), obj.getattr("tag")?.extract::<String>()?), dist);
        }
    }



    // Find matches with the lowest distances
    while !pairs.is_empty() {
        let (lowest_pair, _) = pairs.iter().min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().clone();
        matches.push(lowest_pair.clone());
        let tag = lowest_pair.0.clone();
        let idx = lowest_pair.1.clone();

        // Remove all pairs that contain the matched tag or idx
        pairs.retain(|(t, i), _| t != &tag && i != &idx);
    }


    // Update frame with matches
    for match_pair in matches {
        frame.call_method1("update_player_tag", (match_pair.1, match_pair.0))?;
    }


    // Remove players with certain tags
    let players = frame.call_method0("players")?.extract::<Vec<&PyAny>>()?;
    for player in players {
        let tag: String = player.getattr("tag")?.extract()?;
        if !["A", "B", "C", "D"].contains(&tag.as_str()) {
            frame.getattr("objects")?.call_method1("remove", (player,))?;
        }
    }

    Ok(())
}

#[pymodule]
fn rust_functions(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tag_frames, m)?)?;
    Ok(())
}
