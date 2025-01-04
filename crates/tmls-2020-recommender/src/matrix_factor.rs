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


