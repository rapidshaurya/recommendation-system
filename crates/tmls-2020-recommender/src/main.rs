use anyhow::Result;
use datafusion::prelude::*;
use tmls_2020_recommender::{
    matrix_factor::{
        compute_rmse, highest_rated_films, matrix_factorization, most_frequently_rated_films,
    },
    user_item_matrix::create_user_item_matrix,
};
#[tokio::main]
async fn main() -> Result<()> {
    let ctx = SessionContext::new();
    ctx.register_csv("ratings", "ratings.csv", CsvReadOptions::new())
        .await?;

    // Load data and create user-item matrix
    let matrix = create_user_item_matrix(&ctx).await?;

    // // Perform matrix factorization
    // let (user_features, item_features) = matrix_factorization(&matrix, 10);

    // // Generate predictions and evaluate
    // let predictions = user_features.dot(&item_features);
    // let rmse = compute_rmse(&matrix.view(), &predictions.view()); // Use .view() to pass views
    // println!("RMSE: {:.4}", rmse);

    let a = most_frequently_rated_films(&matrix, 10);
    Ok(())
}
