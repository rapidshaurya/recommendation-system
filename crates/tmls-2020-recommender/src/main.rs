use anyhow::Result;

use nalgebra::DVector;
use tmls_2020_recommender::prelude::*;
#[tokio::main]
async fn main() -> Result<()> {
    let pool = create_test_pool(1);
    let mut conn = pool.get().unwrap();

    // Create the user-item matrix
    let matrix = create_user_item_matrix(&mut conn).unwrap();

    // Print matrix dimensions
    dbg!(matrix.ncols(), matrix.nrows());

    // Compute item-item similarity matrix
    let similarity_matrix = compute_item_item_similarity(&matrix);

    // Find the top-10 nearest neighbors for each item
    let neighbors = find_k_nearest_neighbors(&similarity_matrix, 10);
    if let Some(sim) = neighbors.get(&1) {
        for movie in sim {
            println!("Movie ID: {}, Similarity: {}", movie.0, movie.1);
        }
    }

    // Recommend items for user 0
    let user_ratings = DVector::from_vec(matrix.row(1).iter().copied().collect());

    let recommendations = recommend_items_for_user(&user_ratings, &neighbors);

    println!("Recommendations for user {}: {:?}", 0, recommendations);

    Ok(())
}
