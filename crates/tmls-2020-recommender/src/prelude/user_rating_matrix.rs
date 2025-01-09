use anyhow::Context;
use diesel::*;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Queryable, Selectable)]
#[diesel(table_name = crate::schema::ratings)]
pub struct UserMovieRating {
    pub user_id: i64,
    pub movie_id: i64,
    pub rating: Option<f64>,
}

/// Create a user-item rating matrix from the database
pub fn create_user_item_matrix(conn: &mut pg::PgConnection) -> anyhow::Result<DMatrix<f64>> {
    use crate::schema::ratings;

    // Fetch all ratings from the database
    let ratings: Vec<UserMovieRating> = ratings::table
        .select(UserMovieRating::as_select())
        .get_results(conn)
        .context("Failed to fetch user ratings from database")?;
    let mut unique_movies: HashSet<i64> = HashSet::new();
    // Build a nested hash map of user_id -> movie_id -> rating
    let mut ratings_map: HashMap<i64, HashMap<i64, f64>> = HashMap::new();
    for row in ratings {
        ratings_map
            .entry(row.user_id)
            .or_insert_with(HashMap::new)
            .insert(row.movie_id, row.rating.unwrap_or(0.0));
        // Add the movie_id to the unique movies set
        unique_movies.insert(row.movie_id);
    }
    let num_movies = unique_movies.len();
    // Create and populate the matrix
    let num_users = ratings_map.len();
    let mut matrix = DMatrix::zeros(num_users, num_movies);
    for (user_index, (_, movie_ratings)) in ratings_map.into_iter().enumerate() {
        for (movie_id, rating) in movie_ratings {
            if (movie_id as usize) < num_movies {
                matrix[(user_index, movie_id as usize - 1)] = rating;
            }
        }
    }

    Ok(matrix)
}

/// Find the top N most frequently rated films
pub fn most_frequently_rated_films(ratings: &DMatrix<f64>, top_n: usize) -> Vec<(usize, usize)> {
    let mut film_rating_count: Vec<(usize, usize)> = (0..ratings.ncols())
        .into_par_iter()
        .map(|j| {
            let count = ratings
                .column(j)
                .iter()
                .filter(|&&rating| rating > 0.0)
                .count();
            (j, count)
        })
        .collect();

    film_rating_count.par_sort_unstable_by(|a, b| b.1.cmp(&a.1));
    film_rating_count.into_par_iter().take(top_n).collect()
}

/// Find the top N highest-rated films by average rating
pub fn highest_rated_films(ratings: &DMatrix<f64>, top_n: usize) -> Vec<(usize, f64)> {
    let film_ratings: Vec<(usize, f64)> = (0..ratings.ncols())
        .into_par_iter()
        .filter_map(|j| {
            let column = ratings.column(j);
            let (sum, count) = column.iter().fold((0.0, 0), |(sum, count), &rating| {
                if rating > 0.0 {
                    (sum + rating, count + 1)
                } else {
                    (sum, count)
                }
            });

            if count > 0 {
                Some((j, sum / count as f64))
            } else {
                None
            }
        })
        .collect();

    let mut sorted_ratings = film_ratings;
    sorted_ratings.par_sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    sorted_ratings.into_iter().take(top_n).collect()
}

/// Find the top-k most similar items for each item
pub fn find_k_nearest_neighbors(
    similarity_matrix: &DMatrix<f64>,
    k: usize,
) -> HashMap<usize, Vec<(usize, f64)>> {
    (0..similarity_matrix.nrows())
        .into_par_iter()
        .map(|i| {
            let mut sims: Vec<(usize, f64)> = similarity_matrix
                .row(i)
                .iter()
                .enumerate()
                .filter(|&(j, _)| i != j)
                .map(|(j, &sim)| (j, sim))
                .collect();

            sims.par_sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            (i, sims.into_iter().take(k).collect())
        })
        .collect()
}

/// Recommend items for a user based on item-item similarities
pub fn recommend_items_for_user(
    user_ratings: &DVector<f64>,
    neighbors: &HashMap<usize, Vec<(usize, f64)>>,
) -> Vec<(usize, f64)> {
    let scores: HashMap<usize, f64> = user_ratings
        .iter()
        .enumerate()
        .filter(|(_, &rating)| rating > 0.0)
        .flat_map(|(item, &rating)| {
            neighbors
                .get(&item)
                .into_iter()
                .flat_map(move |item_neighbors| {
                    item_neighbors
                        .iter()
                        .map(move |&(neighbor, sim)| (neighbor, rating * sim))
                })
        })
        .fold(HashMap::new(), |mut acc, (neighbor, score)| {
            *acc.entry(neighbor).or_insert(0.0) += score;
            acc
        });

    let mut recommendations: Vec<(usize, f64)> = scores.into_iter().collect();
    recommendations
        .par_sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    recommendations
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f64], b: &[f64], mag_a: f64, mag_b: f64) -> f64 {
    if mag_a == 0.0 || mag_b == 0.0 {
        0.0
    } else {
        let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        dot_product / (mag_a * mag_b)
    }
}

/// Compute the item-item similarity matrix (optimized with parallelism)
pub fn compute_item_item_similarity(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    let num_items = matrix.ncols();

    // Precompute magnitudes for each column (item)
    let magnitudes: Vec<f64> = (0..num_items)
        .into_par_iter()
        .map(|i| matrix.column(i).dot(&matrix.column(i)).sqrt())
        .collect();

    // Compute similarity matrix in parallel
    let similarity_data: Vec<f64> = (0..num_items)
        .into_par_iter()
        .flat_map(|i| {
            let magnitudes = &magnitudes; // Create a reference to magnitudes
            (i..num_items).into_par_iter().map(move |j| {
                let sim = cosine_similarity(
                    matrix.column(i).as_slice(),
                    matrix.column(j).as_slice(),
                    magnitudes[i],
                    magnitudes[j],
                );
                sim
            })
        })
        .collect();

    // Construct the similarity matrix
    let mut similarity_matrix = DMatrix::zeros(num_items, num_items);
    for i in 0..num_items {
        for j in i..num_items {
            let sim = similarity_data[i * num_items + j - (i * (i + 1) / 2)];
            similarity_matrix[(i, j)] = sim;
            similarity_matrix[(j, i)] = sim; // Symmetric matrix
        }
    }

    similarity_matrix
}

/// Tests
#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::create_test_pool;

    #[test]
    fn test_user_matrix() {
        let pool = create_test_pool(1);
        let mut conn = pool.get().unwrap();
        let matrix = create_user_item_matrix(&mut conn).unwrap();

        println!("Top 10 Most Frequently Rated Films:");
        for (film_id, count) in most_frequently_rated_films(&matrix, 10) {
            println!("Film ID: {}, Count: {}", film_id, count);
        }

        println!("Top 10 Highest Rated Films:");
        for (film_id, rating) in highest_rated_films(&matrix, 10) {
            println!("Film ID: {}, Rating: {:.2}", film_id, rating);
        }
    }

    #[test]
    fn test_similarity_computation() {
        // Create a small test matrix
        let matrix = DMatrix::from_vec(
            3,
            4,
            vec![4.0, 3.0, 0.0, 5.0, 5.0, 0.0, 4.0, 0.0, 3.0, 1.0, 2.0, 4.0],
        );

        let similarity_matrix = compute_item_item_similarity(&matrix);

        // Check if the similarity matrix is symmetric

        // Check if diagonal elements are 1 (self-similarity)
        for i in 0..similarity_matrix.nrows() {
            assert!((similarity_matrix[(i, i)] - 1.0).abs() < 1e-6);
        }

        // You can add more specific checks here based on expected similarity values
    }
}
