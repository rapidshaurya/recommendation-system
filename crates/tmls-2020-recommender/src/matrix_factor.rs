use ndarray::{Array, Array2, ArrayView2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand::thread_rng;

// Constants for SGD
const LEARNING_RATE: f64 = 0.01;
const REGULARIZATION: f64 = 0.1;
const EPOCHS: usize = 500;

pub fn compute_rmse(ratings: &ArrayView2<f64>, predictions: &ArrayView2<f64>) -> f64 {
    let diff = ratings - predictions;
    let mse = diff.mapv(|x| x.powi(2)).mean().unwrap();
    mse.sqrt()
}

pub fn matrix_factorization(
    ratings: &Array2<f64>,
    latent_features: usize,
) -> (Array2<f64>, Array2<f64>) {
    let (num_users, num_items) = ratings.dim();
    let mut rng = thread_rng();

    // Randomly initialize user and item feature matrices
    let mut user_features = Array::random((num_users, latent_features), Uniform::new(0.0, 1.0));
    let mut item_features = Array::random((latent_features, num_items), Uniform::new(0.0, 1.0));

    for epoch in 0..EPOCHS {
        for ((i, j), &rating) in ratings.indexed_iter() {
            if rating > 0.0 {
                // Compute error for the current rating
                let prediction = user_features.row(i).dot(&item_features.column(j));
                let error = rating - prediction;

                // Update user and item feature matrices
                for k in 0..latent_features {
                    user_features[[i, k]] += LEARNING_RATE
                        * (2.0 * error * item_features[[k, j]]
                            - REGULARIZATION * user_features[[i, k]]);
                    item_features[[k, j]] += LEARNING_RATE
                        * (2.0 * error * user_features[[i, k]]
                            - REGULARIZATION * item_features[[k, j]]);
                }
            }
        }

        // Compute predictions and RMSE
        let predictions = user_features.dot(&item_features);
        let rmse = compute_rmse(&ratings.view(), &predictions.view()); // Convert predictions to a view
        if epoch % 50 == 0 {
            println!("Epoch: {}, RMSE: {:.4}", epoch, rmse);
        }
    }

    (user_features, item_features)
}

pub fn highest_rated_films(ratings: &Array2<f64>, top_n: usize) -> Vec<(usize, f64)> {
    let mut film_ratings: Vec<(usize, f64)> = Vec::new();

    // Iterate over columns (films) and calculate the average rating for each film
    for j in 0..ratings.ncols() {
        let mut sum = 0.0;
        let mut count = 0;

        // Iterate over rows (users) for each film
        for i in 0..ratings.nrows() {
            let rating = ratings[[i, j]];
            if rating > 0.0 {
                // Consider only rated films
                sum += rating;
                count += 1;
            }
        }

        // Compute the average rating for the current film
        if count > 0 {
            let avg_rating = sum / count as f64;
            film_ratings.push((j, avg_rating));
        }
    }

    // Sort films by average rating in descending order
    film_ratings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Return the top N highest-rated films
    film_ratings.into_iter().take(top_n).collect()
}

pub fn most_frequently_rated_films(ratings: &Array2<f64>, top_n: usize) {
    let mut film_rating_count: Vec<(usize, usize)> = Vec::new();

    // Iterate over columns (films) and count the number of ratings for each film
    for j in 0..ratings.ncols() {
        let mut count = 0;

        // Iterate over rows (users) for each film
        for i in 0..ratings.nrows() {
            let rating = ratings[[i, j]];
            if rating > 0.0 {
                // Consider only rated films (non-zero ratings)
                count += 1;
            }
        }

        // Store the count of ratings for the current film (column index)
        film_rating_count.push((j, count));
    }

    // Sort films by rating count in descending order
    film_rating_count.sort_by(|a, b| b.1.cmp(&a.1));

    // Print the top N most frequently rated films with their IDs
    for (film_id, count) in film_rating_count.into_iter().take(top_n) {
        println!("Film ID: {}, Number of Ratings: {}", film_id, count);
    }
}
