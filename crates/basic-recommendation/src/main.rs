use std::collections::{HashMap, HashSet};

static DATA: &'static str = include_str!("../../../data.json");

#[derive(Clone, serde::Deserialize)]
struct Document {
    id: String,
    content: String,
}

fn main() {
    let documents: Vec<Document> = serde_json::from_str(&DATA).unwrap();

    let mut input = String::new();
    println!("Enter content for recommendation :");
    let _ = std::io::stdin().read_line(&mut input).unwrap();

    let target_doc = input.trim();
    if target_doc.is_empty() {
        println!("No input provided. Exiting...");
        return;
    }
    let recommendations = recommend(&documents, target_doc, 15);

    println!("Recommendations for '{}':", target_doc);
    for (doc, score) in recommendations {
        println!(
            "  - ID: \"{}\", Content: \"{}\" (score: {:.2})",
            doc.id, doc.content, score
        );
    }
}

fn recommend<'a>(
    documents: &'a [Document],
    target: &'a str,
    top_n: usize,
) -> Vec<(&'a Document, f64)> {
    let all_docs = std::iter::once(target)
        .chain(documents.iter().map(|doc| doc.content.as_str()))
        .collect::<Vec<_>>();

    let tf_idf = compute_tf_idf(&all_docs);
    let map = HashMap::new();
    let target_vector = tf_idf.get(target).unwrap_or(&map);

    let mut similarities = documents
        .iter()
        .map(|doc| {
            let map = HashMap::new();
            let doc_vector = tf_idf.get(doc.content.as_str()).unwrap_or(&map);
            let score = cosine_similarity(target_vector, doc_vector);
            (doc, score)
        })
        .collect::<Vec<_>>();

    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    similarities.into_iter().take(top_n).collect()
}

fn compute_tf_idf<'a>(documents: &'a [&str]) -> HashMap<&'a str, HashMap<String, f64>> {
    let mut term_frequencies = HashMap::new();
    let mut document_frequencies = HashMap::new();
    let mut total_docs = 0;

    for &doc in documents {
        total_docs += 1;
        let words: HashSet<_> = tokenize(doc).into_iter().collect();
        for word in &words {
            *document_frequencies.entry(word.to_string()).or_insert(0) += 1;
        }
        term_frequencies.insert(doc, compute_tf(doc));
    }

    let mut tf_idf = HashMap::new();
    for (&doc, tf) in &term_frequencies {
        let mut doc_tf_idf = HashMap::new();
        for (word, &tf_value) in tf {
            let df = *document_frequencies.get(word).unwrap_or(&1) as f64;
            let idf = (total_docs as f64 / df).ln();
            doc_tf_idf.insert(word.clone(), tf_value * idf);
        }
        tf_idf.insert(doc, doc_tf_idf);
    }

    tf_idf
}

fn compute_tf(document: &str) -> HashMap<String, f64> {
    let tokens = tokenize(document);
    let mut tf = HashMap::new();
    let total_tokens = tokens.len() as f64;

    for token in tokens {
        *tf.entry(token).or_insert(0.0) += 1.0 / total_tokens;
    }

    tf
}

fn tokenize(document: &str) -> Vec<String> {
    document
        .to_lowercase()
        .split_whitespace()
        .map(|word| {
            word.trim_matches(|c: char| !c.is_alphanumeric())
                .to_string()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

fn cosine_similarity(vec1: &HashMap<String, f64>, vec2: &HashMap<String, f64>) -> f64 {
    let mut dot_product = 0.0;
    let mut magnitude1 = 0.0;
    let mut magnitude2 = 0.0;

    let unique_keys: HashSet<_> = vec1.keys().chain(vec2.keys()).collect();
    for key in unique_keys {
        let val1 = vec1.get(key).unwrap_or(&0.0);
        let val2 = vec2.get(key).unwrap_or(&0.0);
        dot_product += val1 * val2;
        magnitude1 += val1 * val1;
        magnitude2 += val2 * val2;
    }

    if magnitude1 == 0.0 || magnitude2 == 0.0 {
        0.0
    } else {
        dot_product / (magnitude1.sqrt() * magnitude2.sqrt())
    }
}
